import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math

# 尝试导入 thop 用于计算复杂度
try:
    from thop import clever_format
    from thop import profile
except ImportError:
    profile = None
    clever_format = None

# ==========================================
# 0. 辅助工具 (DropPath)
# ==========================================
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Stochastic Depth (DropPath)"""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

# ==========================================
# 1. 基础组件 (LeFF & LayerScale)
# ==========================================

class LeFF(nn.Module):
    """
    Locally-enhanced Feed-Forward (LeFF)
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # x: (B, H, W, C)
        x = self.fc1(x) 
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = self.act(x)
        x = x.permute(0, 2, 3, 1)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return self.gamma * x

# ==========================================
# 2. 预测器 & Attention
# ==========================================

class ParamPredictor(nn.Module):
    """
    通用参数预测器
    """
    def __init__(self, channels, out_params=5, reduction=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.GELU(),
            nn.Linear(channels // reduction, out_params) 
        )

    def forward(self, x):
        # Output: (B, out_params, 1, 1)
        params = self.net(x)
        return params.view(x.shape[0], -1, 1, 1)

class ButterworthTokenInteraction(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # [修改 1] 初始化为 (C, 1, 1) 的向量，实现 Channel-Wise
        # 使用 torch.ones(channels) * init_val
        self.D0 = nn.Parameter(torch.ones(channels, 1, 1) * 0.5) 
        self.n = nn.Parameter(torch.ones(channels, 1, 1) * 2.0)
        
        self.mix = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1)
        )
        self.norm = nn.LayerNorm(channels)
        self.register_buffer('eps', torch.tensor(1e-6))

    def _get_spectral_mask(self, h, w, device):
        # 坐标系构建 (H, W/2+1)
        fx = torch.fft.rfftfreq(w, device=device)
        fy = torch.fft.fftfreq(h, device=device)
        gy, gx = torch.meshgrid(fy, fx, indexing='ij')
        
        # 扩展坐标维度以匹配 Channel (1, H, W)
        gx = gx.unsqueeze(0) 
        gy = gy.unsqueeze(0)
        
        # [修改 2] D0 和 n 已经是 (C, 1, 1)，利用广播机制自动扩展
        # ratio shape: (C, H, W)
        D0 = F.softplus(self.D0) + self.eps
        ratio = (gx ** 2 + gy ** 2).sqrt() / D0
        
        order = F.softplus(self.n)
        mask = 1.0 / (1.0 + ratio ** (2 * order))
        
        # Output: (1, C, H, W) 适配 Batch
        return mask.unsqueeze(0) 

    # forward 函数不需要修改，Mask 的维度 (1, C, H, W) 会自动与 (B, C, H, W) 广播乘法

# ==========================================
# 3. 核心 Block: Channel-Wise + DualMode
# ==========================================

class ButterworthTokenInteraction_Dynamic(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # [修改 1] 动态参数预测器
        self.param_pred = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 4),
            nn.GELU(),
            nn.Linear(channels // 4, channels * 2) # 输出 2*C 个参数 (D0, n)
        )
        
        self.mix = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1)
        )
        self.norm = nn.LayerNorm(channels)
        self.register_buffer('eps', torch.tensor(1e-6))

    def forward(self, x, spatial_size):
        B, N, C = x.shape
        Nh, Nw = spatial_size
        shortcut = x
        x = self.norm(x)
        x = x.transpose(1, 2).view(B, C, Nh, Nw)
        
        # 1. 动态预测参数
        # params: (B, 2*C) -> (B, 2*C, 1, 1)
        params = self.param_pred(x).view(B, 2*C, 1, 1)
        D0, n = torch.split(params, C, dim=1) # (B, C, 1, 1)
        
        # 2. 约束范围
        D0 = 0.01 + 2.0 * torch.sigmoid(D0) # 限制 D0 范围
        n = 0.5 + 4.0 * torch.sigmoid(n)    # 限制 n 范围
        
        # 3. 生成 Mask (Dynamic & Channel-Wise)
        fx = torch.fft.rfftfreq(Nw, device=x.device)
        fy = torch.fft.fftfreq(Nh, device=x.device)
        gy, gx = torch.meshgrid(fy, fx, indexing='ij')
        
        # (1, 1, H, W)
        gx = gx.unsqueeze(0).unsqueeze(0) 
        gy = gy.unsqueeze(0).unsqueeze(0)
        
        ratio = (gx ** 2 + gy ** 2).sqrt() / (D0 + self.eps)
        mask = 1.0 / (1.0 + ratio ** (2 * n))
        
        # 4. 应用
        x = self.mix(x)
        x_fft = torch.fft.rfft2(x.float())
        x_fft = x_fft * mask # (B, C, H, W) * (B, C, H, W)
        x_out = torch.fft.irfft2(x_fft, s=(Nh, Nw))
        
        x_out = x_out.flatten(2).transpose(1, 2)
        return shortcut + x_out

class AnisotropicButterFlowBlock_ChannelWise(nn.Module):
    def __init__(self, channels, window_size=16, mlp_ratio=4., drop=0., drop_path=0.):
        super().__init__()
        self.channels = channels
        self.ws = window_size
        
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.v_proj = nn.Conv2d(channels, channels, 1)
        self.out_proj = nn.Conv2d(channels, channels, 1)
        
        # --- 1. Global Branch (Channel-Wise) ---
        self.global_pred = ParamPredictor(channels, out_params=channels*5)
        
        # --- 2. Local Branch (Channel-Wise) ---
        # [关键修改] out_params 改回 channels*5
        # 每个通道独立预测参数，恢复光谱细节
        self.local_pred = ParamPredictor(channels, out_params=channels*5)
        
        # --- 3. Attention Projectors ---
        # 输入维度变成了 5*C，需要先降维再做 Attention
        # 5*C -> C (Attention) -> 5*C
        self.token_embed = nn.Linear(channels * 5, channels)
        self.token_proj = nn.Linear(channels, channels * 5)
        
        self.param_attn = ButterworthTokenInteraction(channels)
        # Weights
        self.w_g = nn.Parameter(1.0 * torch.ones(channels))
        self.w_intra = nn.Parameter(1.0 * torch.ones(channels))
        self.w_inter = nn.Parameter(1.0 * torch.ones(channels))
        
        self.mlp = LeFF(channels, int(channels * mlp_ratio), drop=drop)
        self.ls1 = LayerScale(channels)
        self.ls2 = LayerScale(channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.register_buffer('eps', torch.tensor(1e-6))

    def _get_dual_mode_mask(self, params, h, w):
        """
        生成双模态 Mask
        params: (B, 5*C, 1, 1) -> Channel-Wise
        """
        # Reshape: (B, 5*C, 1, 1) -> (B, C, 5, 1, 1)
        B = params.shape[0]
        params = params.view(B, self.channels, 5, 1, 1)
        
        # Split along dim 2 (Params dim)
        D0_u, D0_v, theta, n, gate = torch.split(params, 1, dim=2)
        
        # Squeeze dim 2 to get (B, C, 1, 1)
        D0_u, D0_v = D0_u.squeeze(2), D0_v.squeeze(2)
        theta, n, gate = theta.squeeze(2), n.squeeze(2), gate.squeeze(2)

        # 物理约束
        D0_u = 0.01 + 0.79 * torch.sigmoid(D0_u)
        D0_v = 0.01 + 0.79 * torch.sigmoid(D0_v)
        theta = (torch.sigmoid(theta) - 0.5) * math.pi
        n = 0.5 + 3.5 * torch.sigmoid(n)
        g = torch.sigmoid(gate) # Gate: 0=LowPass, 1=HighPass
        
        fx = torch.fft.rfftfreq(w).to(params.device)
        fy = torch.fft.fftfreq(h).to(params.device)
        gy, gx = torch.meshgrid(fy, fx, indexing='ij')
        
        # 扩展维度以支持广播 (1, 1, H, W)
        gx = gx.unsqueeze(0).unsqueeze(0)
        gy = gy.unsqueeze(0).unsqueeze(0)
        
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        gx_rot = gx * cos_t + gy * sin_t
        gy_rot = gy * cos_t - gx * sin_t
        
        term_u = (gx_rot / (D0_u + self.eps)) ** 2
        term_v = (gy_rot / (D0_v + self.eps)) ** 2
        ratio = torch.sqrt(term_u + term_v + self.eps)
        
        base_response = torch.sigmoid(-2.0 * n * torch.log(ratio + self.eps))
        
        # Dual-Mode Mixing
        H_filter = g * (1.0 - base_response) + (1.0 - g) * base_response
        
        return H_filter

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x
        x_norm = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_v = self.v_proj(x_norm)
        
        # --- 1. Global Branch (Channel-Wise) ---
        global_params = self.global_pred(x_norm) # (B, 5*C, 1, 1)
        mask_g = self._get_dual_mode_mask(global_params, H, W) # (B, C, H, W/2+1)
        x_g_fft = torch.fft.rfft2(x_v.float(), dim=(-2, -1))
        out_global = torch.fft.irfft2(x_g_fft * mask_g, s=(H, W), dim=(-2, -1))

        # --- Window Partition ---
        ws = min(self.ws, H, W)
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x_v_pad = F.pad(x_v, (0, pad_w, 0, pad_h))
        x_norm_pad = F.pad(x_norm, (0, pad_w, 0, pad_h))
        
        x_v_wins = x_v_pad.unfold(2, ws, ws).unfold(3, ws, ws)
        x_ctx_wins = x_norm_pad.unfold(2, ws, ws).unfold(3, ws, ws)
        
        Nh, Nw = x_v_wins.shape[2], x_v_wins.shape[3]
        N = Nh * Nw
        
        x_v_flat = x_v_wins.permute(0, 2, 3, 1, 4, 5).reshape(-1, C, ws, ws)
        x_ctx_grouped = x_ctx_wins.permute(0, 2, 3, 1, 4, 5).reshape(B, N, C, ws, ws)
        x_ctx_flat = x_ctx_grouped.reshape(-1, C, ws, ws) 
        
        wins_fft = torch.fft.rfft2(x_v_flat, dim=(-2, -1))

        # ======================================================
        # 1. 预测 Intra 参数 (B*N, 5*C, 1, 1) [Channel-Wise]
        # ======================================================
        local_params_raw = self.local_pred(x_ctx_flat) 
        
        # 生成 Intra Mask (B*N, C, ws, ws/2+1)
        # 注意: _get_dual_mode_mask 内部已处理了 Channel-Wise 的逻辑
        mask_intra = self._get_dual_mode_mask(local_params_raw, ws, ws)
        
        # ======================================================
        # 2. Inter-Window Interaction (Channel-Wise)
        # ======================================================
        # (B*N, 5*C, 1, 1) -> (B, N, 5*C)
        tokens = local_params_raw.view(B, N, 5 * C)
        
        # Grouped/Mixer Embed
        tokens_emb = self.token_embed(tokens)
        
        # [核心修改] 调用巴特沃斯交互，传入空间尺寸
        refined_emb = self.param_attn(tokens_emb, spatial_size=(Nh, Nw))
        
        # Proj Back
        refined_params = self.token_proj(refined_emb)
        refined_params = refined_params.reshape(B*N, 5 * C, 1, 1) #
        
        # 生成 Inter Mask
        refined_params = refined_params.view(B*N, 5 * C, 1, 1)
        mask_inter = self._get_dual_mode_mask(refined_params, ws, ws)

        # ======================================================

        # Fusion (Masks are already B*N, C, ...)
        # Weights: (C) -> (1, C, 1, 1)
        w_intra_reshaped = self.w_intra.view(1, -1, 1, 1)
        w_inter_reshaped = self.w_inter.view(1, -1, 1, 1)
        
        combined_mask = (mask_intra * w_intra_reshaped) + (mask_inter * w_inter_reshaped)
        
        out_wins = torch.fft.irfft2(wins_fft * combined_mask, s=(ws, ws), dim=(-2, -1))
        
        # Restore Shape
        out_windows = out_wins.view(B, Nh, Nw, C, ws, ws)\
                                    .permute(0, 3, 1, 4, 2, 5)\
                                    .reshape(B, C, H+pad_h, W+pad_w)
        out_windows = out_windows[:, :, :H, :W]

        x_fused = (out_global * self.w_g.view(1, -1, 1, 1)) + out_windows
        x_fused = self.out_proj(x_fused)
        
        x = shortcut + self.drop_path(self.ls1(x_fused.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        shortcut = x
        x_norm = self.norm2(x.permute(0, 2, 3, 1))
        x_mlp = self.mlp(x_norm).permute(0, 3, 1, 2)
        x = shortcut + self.drop_path(self.ls2(x_mlp.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        
        return x

class AnisotropicButterFlowBlock_NoGlobal(nn.Module):
    """
    [Ablation Study A] 去除全局分支
    仅保留窗口分支 (Intra + Inter)，验证全局频域滤波的必要性。
    """
    def __init__(self, channels, window_size=16, mlp_ratio=4., drop=0., drop_path=0.):
        super().__init__()
        self.channels = channels
        self.ws = window_size
        
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.v_proj = nn.Conv2d(channels, channels, 1)
        self.out_proj = nn.Conv2d(channels, channels, 1)
        
        # --- 移除 Global Predictor ---
        # self.global_pred = ... (Removed)
        
        # --- 保留 Window Branch 组件 ---
        self.local_pred_latent = ParamPredictor(channels, out_params=channels)
        self.interaction = ButterworthTokenInteraction(channels)
        
        self.param_decoder = nn.Conv1d(
            in_channels=channels, 
            out_channels=channels * 5, 
            kernel_size=1, 
            groups=channels,
            bias=True
        )
        
        # 移除 w_g
        # self.w_g = ... (Removed)
        self.w_intra = nn.Parameter(torch.ones(channels))
        self.w_inter = nn.Parameter(torch.ones(channels))
        
        self.mlp = LeFF(channels, int(channels * mlp_ratio), drop=drop)
        self.ls1 = LayerScale(channels)
        self.ls2 = LayerScale(channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.register_buffer('eps', torch.tensor(1e-6))

    # _get_dual_mode_mask 函数保持不变 (需要复制过来)
    def _get_dual_mode_mask(self, params, h, w):
        # ... (同原版代码) ...
        # 为节省篇幅，此处省略具体实现，请复用原版函数
        B = params.shape[0]
        params = params.view(B, self.channels, 5, 1, 1)
        D0_u, D0_v, theta, n, gate = torch.split(params, 1, dim=2)
        D0_u, D0_v = D0_u.squeeze(2), D0_v.squeeze(2)
        theta, n, gate = theta.squeeze(2), n.squeeze(2), gate.squeeze(2)

        D0_u = 0.01 + 0.79 * torch.sigmoid(D0_u)
        D0_v = 0.01 + 0.79 * torch.sigmoid(D0_v)
        theta = (torch.sigmoid(theta) - 0.5) * math.pi
        n = 0.5 + 3.5 * torch.sigmoid(n)
        g = torch.sigmoid(gate)
        
        fx = torch.fft.rfftfreq(w).to(params.device)
        fy = torch.fft.fftfreq(h).to(params.device)
        gy, gx = torch.meshgrid(fy, fx, indexing='ij')
        
        gx_rot = gx * torch.cos(theta) + gy * torch.sin(theta)
        gy_rot = gy * torch.cos(theta) - gx * torch.sin(theta)
        
        term_u = (gx_rot / (D0_u + self.eps)) ** 2
        term_v = (gy_rot / (D0_v + self.eps)) ** 2
        ratio = torch.sqrt(term_u + term_v + self.eps)
        base_response = torch.sigmoid(-2.0 * n * torch.log(ratio + self.eps))
        H_filter = g * (1.0 - base_response) + (1.0 - g) * base_response
        return H_filter

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x
        
        x_norm = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_v = self.v_proj(x_norm)
        
        # --- 移除 Global Path ---
        # out_global = ... (Removed)

        # --- Window Partition ---
        ws = min(self.ws, H, W)
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x_v_pad = F.pad(x_v, (0, pad_w, 0, pad_h))
        x_norm_pad = F.pad(x_norm, (0, pad_w, 0, pad_h))
        Hp, Wp = x_v_pad.shape[2], x_v_pad.shape[3]
        
        x_v_wins = x_v_pad.unfold(2, ws, ws).unfold(3, ws, ws)
        x_ctx_wins = x_norm_pad.unfold(2, ws, ws).unfold(3, ws, ws)
        Nh, Nw = x_v_wins.shape[2], x_v_wins.shape[3]
        N = Nh * Nw
        
        x_v_flat = x_v_wins.permute(0, 2, 3, 1, 4, 5).reshape(-1, C, ws, ws)
        x_ctx_flat = x_ctx_wins.permute(0, 2, 3, 1, 4, 5).reshape(-1, C, ws, ws)
        wins_fft = torch.fft.rfft2(x_v_flat, dim=(-2, -1))

        # --- Window Processing (Intra + Inter) ---
        latent_tokens = self.local_pred_latent(x_ctx_flat)
        
        # Intra Mask
        intra_params = self.param_decoder(latent_tokens.squeeze(-1)).unsqueeze(-1)
        mask_intra = self._get_dual_mode_mask(intra_params, ws, ws)
        
        # Inter Mask
        tokens_view = latent_tokens.view(B, N, C)
        refined_tokens = self.interaction(tokens_view, spatial_size=(Nh, Nw)) 
        refined_tokens_flat = refined_tokens.view(B * N, C, 1)
        inter_params = self.param_decoder(refined_tokens_flat).unsqueeze(-1)
        mask_inter = self._get_dual_mode_mask(inter_params, ws, ws)

        # Fusion
        combined_mask = (mask_intra * self.w_intra.view(1, -1, 1, 1)) + \
                        (mask_inter * self.w_inter.view(1, -1, 1, 1))
        
        out_wins = torch.fft.irfft2(wins_fft * combined_mask, s=(ws, ws), dim=(-2, -1))
        out_wins = out_wins.view(B, Nh, Nw, C, ws, ws).permute(0, 3, 1, 4, 2, 5).reshape(B, C, Hp, Wp)
        out_windows = out_wins[:, :, :H, :W]

        # --- Final Fusion (仅使用 Window 输出) ---
        # x_fused = (out_global * ...) + out_windows (Removed)
        x_fused = out_windows 
        x_fused = self.out_proj(x_fused)
        
        x = shortcut + self.drop_path(self.ls1(x_fused.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        
        # LeFF
        shortcut = x
        x_norm = self.norm2(x.permute(0, 2, 3, 1))
        x_mlp = self.mlp(x_norm).permute(0, 3, 1, 2)
        x = shortcut + self.drop_path(self.ls2(x_mlp.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        
        return x

class AnisotropicButterFlowBlock_NoIntra(nn.Module):
    """
    [Ablation Study: No Intra-Window Filtering]
    保留：
    1. Global Branch (全局频域滤波)
    2. Window Interaction (窗口间交互 -> 生成 Inter Mask)
    
    移除：
    1. Window Intra Filtering (移除窗口独立预测参数并滤波的分支)
    
    目的：
    验证窗口是否需要"独立思考"的能力，还是仅仅依赖"交互后的上下文信息"就足够了。
    """
    def __init__(self, channels, window_size=16, mlp_ratio=4., drop=0., drop_path=0.):
        super().__init__()
        self.channels = channels
        self.ws = window_size
        
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.v_proj = nn.Conv2d(channels, channels, 1)
        self.out_proj = nn.Conv2d(channels, channels, 1)
        
        # 1. Global Branch (保留)
        self.global_pred = ParamPredictor(channels, out_params=channels*5)
        
        # 2. Local Branch Initializer (保留)
        # 我们仍需要预测初始 Latent，作为交互模块的输入
        self.local_pred_latent = ParamPredictor(channels, out_params=channels)
        
        # 3. Interaction Module (保留)
        self.interaction = ButterworthTokenInteraction_Dynamic(channels)
        
        # 4. Parameter Decoder (保留)
        # 用于将交互后的 Latent 解码为物理参数
        self.param_decoder = nn.Conv1d(
            in_channels=channels, 
            out_channels=channels * 5, 
            kernel_size=1, 
            groups=channels,
            bias=True
        )
        
        self.w_g = nn.Parameter(torch.ones(channels))
        # self.w_intra = ... (已移除：不需要 Intra 权重)
        self.w_inter = nn.Parameter(torch.ones(channels))
        
        self.mlp = LeFF(channels, int(channels * mlp_ratio), drop=drop)
        self.ls1 = LayerScale(channels)
        self.ls2 = LayerScale(channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.register_buffer('eps', torch.tensor(1e-6))

    # _get_dual_mode_mask 函数保持不变 (请复用原版代码)
    def _get_dual_mode_mask(self, params, h, w):
        B = params.shape[0]
        params = params.view(B, self.channels, 5, 1, 1)
        D0_u, D0_v, theta, n, gate = torch.split(params, 1, dim=2)
        D0_u, D0_v = D0_u.squeeze(2), D0_v.squeeze(2)
        theta, n, gate = theta.squeeze(2), n.squeeze(2), gate.squeeze(2)

        #D0_u = 0.01 + 0.79 * torch.sigmoid(D0_u)
        #D0_v = 0.01 + 0.79 * torch.sigmoid(D0_v) #original
        D0_u = torch.sigmoid(D0_u)
        D0_v = torch.sigmoid(D0_v)
        theta = (torch.sigmoid(theta) - 0.5) * math.pi
        n = 4 * torch.sigmoid(n)
        #n = 0.5 + 3.5 * torch.sigmoid(n) # original
        #n = 10 * torch.sigmoid(n)
        g = torch.sigmoid(gate)
        
        fx = torch.fft.rfftfreq(w).to(params.device)
        fy = torch.fft.fftfreq(h).to(params.device)
        gy, gx = torch.meshgrid(fy, fx, indexing='ij')
        gx = gx.unsqueeze(0).unsqueeze(0)
        gy = gy.unsqueeze(0).unsqueeze(0)
        
        gx_rot = gx * torch.cos(theta) + gy * torch.sin(theta)
        gy_rot = gy * torch.cos(theta) - gx * torch.sin(theta)
        
        term_u = (gx_rot / (D0_u + self.eps)) ** 2
        term_v = (gy_rot / (D0_v + self.eps)) ** 2
        ratio = torch.sqrt(term_u + term_v + self.eps)
        base_response = torch.sigmoid(-2.0 * n * torch.log(ratio + self.eps))
        H_filter = g * (1.0 - base_response) + (1.0 - g) * base_response
        return H_filter

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x
        
        x_norm = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_v = self.v_proj(x_norm)
        
        # --- 1. Global Branch (保留) ---
        global_params = self.global_pred(x_norm)
        mask_g = self._get_dual_mode_mask(global_params, H, W)
        x_g_fft = torch.fft.rfft2(x_v.float(), dim=(-2, -1))
        out_global = torch.fft.irfft2(x_g_fft * mask_g, s=(H, W), dim=(-2, -1))

        # --- Window Partition ---
        ws = min(self.ws, H, W)
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x_v_pad = F.pad(x_v, (0, pad_w, 0, pad_h))
        x_norm_pad = F.pad(x_norm, (0, pad_w, 0, pad_h))
        Hp, Wp = x_v_pad.shape[2], x_v_pad.shape[3]
        
        x_v_wins = x_v_pad.unfold(2, ws, ws).unfold(3, ws, ws)
        x_ctx_wins = x_norm_pad.unfold(2, ws, ws).unfold(3, ws, ws)
        Nh, Nw = x_v_wins.shape[2], x_v_wins.shape[3]
        N = Nh * Nw
        
        x_v_flat = x_v_wins.permute(0, 2, 3, 1, 4, 5).reshape(-1, C, ws, ws)
        x_ctx_flat = x_ctx_wins.permute(0, 2, 3, 1, 4, 5).reshape(-1, C, ws, ws)
        wins_fft = torch.fft.rfft2(x_v_flat, dim=(-2, -1))

        # --- Latent Prediction ---
        latent_tokens = self.local_pred_latent(x_ctx_flat)
        #print("latent token",latent_tokens.shape, Nh, Nw)
        # --- [REMOVED] Intra Mask Generation ---
        # intra_params = ... (Deleted)
        # mask_intra = ... (Deleted)
        
        # --- Inter Interaction (保留) ---
        tokens_view = latent_tokens.view(B, N, C)
        # 1. 交互
        refined_tokens = self.interaction(tokens_view, spatial_size=(Nh, Nw)) 
        
        # 2. 解码参数
        refined_tokens_flat = refined_tokens.view(B * N, C, 1)
        inter_params = self.param_decoder(refined_tokens_flat).unsqueeze(-1)
        
        # 3. 生成 Mask
        mask_inter = self._get_dual_mode_mask(inter_params, ws, ws)

        # --- Fusion (仅使用 Inter Mask) ---
        # 原逻辑: combined = intra * w_intra + inter * w_inter
        # 现逻辑: combined = inter * w_inter
        combined_mask = mask_inter * self.w_inter.view(1, -1, 1, 1)
        
        out_wins = torch.fft.irfft2(wins_fft * combined_mask, s=(ws, ws), dim=(-2, -1))
        out_wins = out_wins.view(B, Nh, Nw, C, ws, ws).permute(0, 3, 1, 4, 2, 5).reshape(B, C, Hp, Wp)
        out_windows = out_wins[:, :, :H, :W]

        # --- Final Fusion (Global + Window Inter) ---
        x_fused = (out_global * self.w_g.view(1, -1, 1, 1)) + out_windows
        x_fused = self.out_proj(x_fused)
        
        x = shortcut + self.drop_path(self.ls1(x_fused.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        
        # LeFF
        shortcut = x
        x_norm = self.norm2(x.permute(0, 2, 3, 1))
        x_mlp = self.mlp(x_norm).permute(0, 3, 1, 2)
        x = shortcut + self.drop_path(self.ls2(x_mlp.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        
        return x
# ==========================================
# 4. 主网络架构 (Pansharpening)
# ==========================================

class AnisotropicButterFlowBlock_InterOnly(nn.Module):
    """
    [Ablation Study: Only Window Interaction]
    仅保留窗口交互分支，移除全局分支和窗口内独立滤波分支。
    
    保留：
    1. Latent Prediction (作为交互的输入)
    2. Butterworth Interaction (交互核心，线性复杂度)
    3. Parameter Decoder (解码交互后的参数)
    
    移除：
    1. Global Branch (全局频域滤波)
    2. Intra-Window Filtering (无交互的独立滤波)
    """
    def __init__(self, channels, window_size=16, mlp_ratio=4., drop=0., drop_path=0.):
        super().__init__()
        self.channels = channels
        self.ws = window_size
        
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.v_proj = nn.Conv2d(channels, channels, 1)
        self.out_proj = nn.Conv2d(channels, channels, 1)
        
        # --- [REMOVED] Global Predictor ---
        # self.global_pred = ... 
        
        # 1. Latent Predictor (保留)
        # 即使没有 Intra 分支，我们仍需预测初始 Token 给交互模块
        self.local_pred_latent = ParamPredictor(channels, out_params=channels)
        
        # 2. Interaction Module (保留)
        self.interaction = ButterworthTokenInteraction(channels)
        
        # 3. Parameter Decoder (保留)
        # 用于将交互后的 Token 解码为物理参数
        self.param_decoder = nn.Conv1d(
            in_channels=channels, 
            out_channels=channels * 5, 
            kernel_size=1, 
            groups=channels,
            bias=True
        )
        
        # --- [REMOVED] w_g, w_intra ---
        # self.w_g = ...
        # self.w_intra = ...
        
        # 仅保留 Inter 权重
        self.w_inter = nn.Parameter(torch.ones(channels))
        
        self.mlp = LeFF(channels, int(channels * mlp_ratio), drop=drop)
        self.ls1 = LayerScale(channels)
        self.ls2 = LayerScale(channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.register_buffer('eps', torch.tensor(1e-6))

    def _get_dual_mode_mask(self, params, h, w):
        # 复用原有的 Mask 生成逻辑
        B = params.shape[0]
        params = params.view(B, self.channels, 5, 1, 1)
        D0_u, D0_v, theta, n, gate = torch.split(params, 1, dim=2)
        D0_u, D0_v = D0_u.squeeze(2), D0_v.squeeze(2)
        theta, n, gate = theta.squeeze(2), n.squeeze(2), gate.squeeze(2)

        D0_u = 0.01 + 0.79 * torch.sigmoid(D0_u)
        D0_v = 0.01 + 0.79 * torch.sigmoid(D0_v)
        theta = (torch.sigmoid(theta) - 0.5) * math.pi
        n = 0.5 + 3.5 * torch.sigmoid(n)
        g = torch.sigmoid(gate)
        
        fx = torch.fft.rfftfreq(w).to(params.device)
        fy = torch.fft.fftfreq(h).to(params.device)
        gy, gx = torch.meshgrid(fy, fx, indexing='ij')
        
        gx_rot = gx * torch.cos(theta) + gy * torch.sin(theta)
        gy_rot = gy * torch.cos(theta) - gx * torch.sin(theta)
        
        term_u = (gx_rot / (D0_u + self.eps)) ** 2
        term_v = (gy_rot / (D0_v + self.eps)) ** 2
        ratio = torch.sqrt(term_u + term_v + self.eps)
        
        base_response = torch.sigmoid(-2.0 * n * torch.log(ratio + self.eps))
        H_filter = g * (1.0 - base_response) + (1.0 - g) * base_response
        return H_filter

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x
        
        x_norm = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_v = self.v_proj(x_norm)
        
        # --- [REMOVED] Global Branch Logic ---
        # out_global = ... (Deleted)

        # --- Window Partition ---
        ws = min(self.ws, H, W)
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x_v_pad = F.pad(x_v, (0, pad_w, 0, pad_h))
        x_norm_pad = F.pad(x_norm, (0, pad_w, 0, pad_h))
        Hp, Wp = x_v_pad.shape[2], x_v_pad.shape[3]
        
        x_v_wins = x_v_pad.unfold(2, ws, ws).unfold(3, ws, ws)
        x_ctx_wins = x_norm_pad.unfold(2, ws, ws).unfold(3, ws, ws)
        Nh, Nw = x_v_wins.shape[2], x_v_wins.shape[3]
        N = Nh * Nw
        
        x_v_flat = x_v_wins.permute(0, 2, 3, 1, 4, 5).reshape(-1, C, ws, ws)
        x_ctx_flat = x_ctx_wins.permute(0, 2, 3, 1, 4, 5).reshape(-1, C, ws, ws)
        wins_fft = torch.fft.rfft2(x_v_flat, dim=(-2, -1))

        # --- Latent Prediction ---
        latent_tokens = self.local_pred_latent(x_ctx_flat)
        
        # --- [REMOVED] Intra Branch Logic ---
        # intra_params = ... (Deleted)
        # mask_intra = ... (Deleted)
        
        # --- Inter Interaction (保留核心) ---
        tokens_view = latent_tokens.view(B, N, C)
        
        # 1. 巴特沃斯交互 (O(N log N))
        refined_tokens = self.interaction(tokens_view, spatial_size=(Nh, Nw)) 
        
        # 2. 解码为物理参数
        refined_tokens_flat = refined_tokens.view(B * N, C, 1)
        inter_params = self.param_decoder(refined_tokens_flat).unsqueeze(-1)
        
        # 3. 生成 Mask
        mask_inter = self._get_dual_mode_mask(inter_params, ws, ws)

        # --- Fusion (仅使用 Inter) ---
        # combined = inter * w_inter
        combined_mask = mask_inter * self.w_inter.view(1, -1, 1, 1)
        
        out_wins = torch.fft.irfft2(wins_fft * combined_mask, s=(ws, ws), dim=(-2, -1))
        out_wins = out_wins.view(B, Nh, Nw, C, ws, ws).permute(0, 3, 1, 4, 2, 5).reshape(B, C, Hp, Wp)
        out_windows = out_wins[:, :, :H, :W]

        # --- Final Fusion (仅使用 Window Output) ---
        # x_fused = (out_global * ...) + out_windows -> 变为仅 out_windows
        x_fused = out_windows 
        x_fused = self.out_proj(x_fused)
        
        x = shortcut + self.drop_path(self.ls1(x_fused.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        
        # LeFF Block (保持不变)
        shortcut = x
        x_norm = self.norm2(x.permute(0, 2, 3, 1))
        x_mlp = self.mlp(x_norm).permute(0, 3, 1, 2)
        x = shortcut + self.drop_path(self.ls2(x_mlp.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        
        return x

class AnisotropicButterworthTokenInteraction(nn.Module):
    """
    [Anisotropic Window Interaction]
    
    功能：
    在频域对窗口 Token Map 进行各向异性滤波交互。
    不仅实现了 O(N log N) 的线性全局交互，还支持方向性纹理感知。
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # --- 各向异性参数 ---
        self.D0_u = nn.Parameter(torch.tensor(0.5)) 
        self.D0_v = nn.Parameter(torch.tensor(0.5)) 
        self.theta = nn.Parameter(torch.tensor(0.0)) # 旋转角度
        self.n = nn.Parameter(torch.tensor(2.0))     # 阶数
        
        # 通道混合
        self.mix = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1)
        )
        
        self.norm = nn.LayerNorm(channels)
        self.register_buffer('eps', torch.tensor(1e-6))

    def _get_anisotropic_mask(self, h, w, device):
        """生成各向异性(椭圆)频域 Mask"""
        # 1. 坐标系
        fx = torch.fft.rfftfreq(w, device=device)
        fy = torch.fft.fftfreq(h, device=device)
        gy, gx = torch.meshgrid(fy, fx, indexing='ij')
        
        # 2. 坐标旋转
        cos_t = torch.cos(self.theta)
        sin_t = torch.sin(self.theta)
        
        gx_rot = gx * cos_t + gy * sin_t
        gy_rot = gy * cos_t - gx * sin_t
        
        # 3. 椭圆距离度量
        d0_u = F.softplus(self.D0_u) + self.eps
        d0_v = F.softplus(self.D0_v) + self.eps
        
        term_u = (gx_rot / d0_u) ** 2
        term_v = (gy_rot / d0_v) ** 2
        ratio = torch.sqrt(term_u + term_v + self.eps)
        
        # 4. 巴特沃斯公式
        order = F.softplus(self.n)
        mask = 1.0 / (1.0 + ratio ** (2 * order))
        
        return mask.unsqueeze(0).unsqueeze(0) # (1, 1, H, W/2+1)

    def forward(self, x, spatial_size):
        # x: (B, N, C)
        B, N, C = x.shape
        Nh, Nw = spatial_size
        
        shortcut = x
        x = self.norm(x)
        x = x.transpose(1, 2).view(B, C, Nh, Nw)
        
        # Channel Mixing
        x = self.mix(x)
        
        # FFT
        x_fft = torch.fft.rfft2(x.float())
        
        # Apply Anisotropic Mask
        mask = self._get_anisotropic_mask(Nh, Nw, x.device)
        x_fft = x_fft * mask
        
        # IFFT
        x_out = torch.fft.irfft2(x_fft, s=(Nh, Nw))
        
        # Restore
        x_out = x_out.flatten(2).transpose(1, 2)
        return shortcut + x_out

# ==========================================
# 3. [完整 Block] 全局各向异性 + 窗口各向异性交互
# ==========================================

class AnisotropicButterFlowBlock_Full(nn.Module):
    def __init__(self, channels, window_size=16, mlp_ratio=4., drop=0., drop_path=0.):
        super().__init__()
        self.channels = channels
        self.ws = window_size
        
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.v_proj = nn.Conv2d(channels, channels, 1)
        self.out_proj = nn.Conv2d(channels, channels, 1)
        
        # --- 1. Global Branch Predictor (Anisotropic) ---
        # 预测全局滤波参数
        self.global_pred = ParamPredictor(channels, out_params=channels*5)
        
        # --- 2. Local Branch Components ---
        # A. 隐式预测 (Latent Prediction)
        self.local_pred_latent = ParamPredictor(channels, out_params=channels)
        
        # B. 交互模块 (Anisotropic Interaction)
        # [NEW] 使用各向异性交互模块
        self.interaction = AnisotropicButterworthTokenInteraction(channels)
        
        # C. 参数解码器 (Decoder)
        self.param_decoder = nn.Conv1d(
            in_channels=channels, 
            out_channels=channels * 5, 
            kernel_size=1, 
            groups=channels,
            bias=True
        )
        
        # Fusion Weights
        self.w_g = nn.Parameter(torch.ones(channels))
        self.w_inter = nn.Parameter(torch.ones(channels))
        
        self.mlp = LeFF(channels, int(channels * mlp_ratio), drop=drop)
        self.ls1 = LayerScale(channels)
        self.ls2 = LayerScale(channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.register_buffer('eps', torch.tensor(1e-6))

    def _get_dual_mode_mask(self, params, h, w):
        """
        通用的各向异性双模态 Mask 生成函数
        用于 Global Branch 和 Local Branch 的参数生成
        """
        B = params.shape[0]
        params = params.view(B, self.channels, 5, 1, 1)
        D0_u, D0_v, theta, n, gate = torch.split(params, 1, dim=2)
        
        # Squeeze & Physical Constraints
        D0_u = 0.01 + 0.79 * torch.sigmoid(D0_u.squeeze(2))
        D0_v = 0.01 + 0.79 * torch.sigmoid(D0_v.squeeze(2))
        theta = (torch.sigmoid(theta.squeeze(2)) - 0.5) * math.pi
        n = 0.5 + 3.5 * torch.sigmoid(n.squeeze(2))
        g = torch.sigmoid(gate.squeeze(2))
        
        # Coordinates
        fx = torch.fft.rfftfreq(w).to(params.device)
        fy = torch.fft.fftfreq(h).to(params.device)
        gy, gx = torch.meshgrid(fy, fx, indexing='ij')
        
        # Broadcasting dims
        gx = gx.unsqueeze(0).unsqueeze(0)
        gy = gy.unsqueeze(0).unsqueeze(0)
        
        # Rotation
        gx_rot = gx * torch.cos(theta) + gy * torch.sin(theta)
        gy_rot = gy * torch.cos(theta) - gx * torch.sin(theta)
        
        # Metric
        term_u = (gx_rot / (D0_u + self.eps)) ** 2
        term_v = (gy_rot / (D0_v + self.eps)) ** 2
        ratio = torch.sqrt(term_u + term_v + self.eps)
        
        # Response
        base_response = torch.sigmoid(-2.0 * n * torch.log(ratio + self.eps))
        H_filter = g * (1.0 - base_response) + (1.0 - g) * base_response
        
        return H_filter

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x
        x_norm = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_v = self.v_proj(x_norm)
        
        # ==========================================
        # Branch 1: Global Anisotropic Filtering
        # ==========================================
        global_params = self.global_pred(x_norm)
        mask_g = self._get_dual_mode_mask(global_params, H, W)
        x_g_fft = torch.fft.rfft2(x_v.float(), dim=(-2, -1))
        out_global = torch.fft.irfft2(x_g_fft * mask_g, s=(H, W), dim=(-2, -1))

        # ==========================================
        # Branch 2: Local Window Processing
        # ==========================================
        # 1. Partition
        ws = min(self.ws, H, W)
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x_v_pad = F.pad(x_v, (0, pad_w, 0, pad_h))
        x_norm_pad = F.pad(x_norm, (0, pad_w, 0, pad_h))
        Hp, Wp = x_v_pad.shape[2], x_v_pad.shape[3]
        
        x_v_wins = x_v_pad.unfold(2, ws, ws).unfold(3, ws, ws)
        x_ctx_wins = x_norm_pad.unfold(2, ws, ws).unfold(3, ws, ws)
        Nh, Nw = x_v_wins.shape[2], x_v_wins.shape[3]
        N = Nh * Nw
        
        x_v_flat = x_v_wins.permute(0, 2, 3, 1, 4, 5).reshape(-1, C, ws, ws)
        x_ctx_flat = x_ctx_wins.permute(0, 2, 3, 1, 4, 5).reshape(-1, C, ws, ws)
        wins_fft = torch.fft.rfft2(x_v_flat, dim=(-2, -1))

        # 2. Latent Prediction (Implicit Feature)
        latent_tokens = self.local_pred_latent(x_ctx_flat) # (B*N, C, 1, 1)
        # 3. Anisotropic Interaction (Window <-> Window)
        tokens_view = latent_tokens.view(B, N, C)
        # [关键] 调用各向异性交互
        refined_tokens = self.interaction(tokens_view, spatial_size=(Nh, Nw)) 
        
        # 4. Decode & Generate Mask
        refined_tokens_flat = refined_tokens.view(B * N, C, 1)
        inter_params = self.param_decoder(refined_tokens_flat).unsqueeze(-1)
        mask_inter = self._get_dual_mode_mask(inter_params, ws, ws)

        # 5. Local Fusion (Only Inter result used as discussed in Final ver)
        out_wins = torch.fft.irfft2(wins_fft * mask_inter, s=(ws, ws), dim=(-2, -1))
        
        # Restore Windows
        out_windows = out_wins.view(B, Nh, Nw, C, ws, ws).permute(0, 3, 1, 4, 2, 5).reshape(B, C, Hp, Wp)
        out_windows = out_windows[:, :, :H, :W]

        # ==========================================
        # Final Fusion (Global + Local)
        # ==========================================
        # Note: Local weight (w_inter) applied inside mask generation or here
        # Here we apply weights at feature level
        x_fused = (out_global * self.w_g.view(1, -1, 1, 1)) + \
                  (out_windows * self.w_inter.view(1, -1, 1, 1))
                  
        x_fused = self.out_proj(x_fused)
        
        x = shortcut + self.drop_path(self.ls1(x_fused.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        
        # MLP
        shortcut = x
        x_norm = self.norm2(x.permute(0, 2, 3, 1))
        x_mlp = self.mlp(x_norm).permute(0, 3, 1, 2)
        x = shortcut + self.drop_path(self.ls2(x_mlp.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        
        return x

class ButterFlowNet_Anisotropic(nn.Module):
    def __init__(self, spectral_num=8, channels=64, num_groups=7, window_size=4, drop_path_rate=0.1):
        super().__init__()
        self.input_conv = nn.Conv2d(spectral_num + 1, channels, 3, padding=1)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_groups)]
        
        self.blocks = nn.ModuleList([
            #AnisotropicButterFlowBlock_ChannelWise(channels, window_size=window_size, drop_path=dpr[i]) 
            AnisotropicButterFlowBlock_NoIntra(channels, window_size=window_size, drop_path=dpr[i]) 
            #AnisotropicButterFlowBlock_InterOnly(channels, window_size=window_size, drop_path=dpr[i])
            #AnisotropicButterFlowBlock_Full(channels, window_size=window_size, drop_path=dpr[i])
            for i in range(num_groups)
        ])
        
        self.output_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels // 2, spectral_num, 3, padding=1)
        )
    
    def forward(self, lms, pan):
        x = self.input_conv(torch.cat([lms, pan], dim=1))
        for block in self.blocks:
            x = block(x)
        return self.output_conv(x) + lms

# ==========================================
# 5. 测试脚本
# ==========================================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on {device}...")
    
    # Init Model
    model = ButterFlowNet_Anisotropic(channels=48, window_size=4).to(device)
    
    # Dummy Input (Pansharpening task)
    lms = torch.randn(1, 8, 256, 256).to(device)
    pan = torch.randn(1, 1, 256, 256).to(device)
    
    try:
        out = model(lms, pan)
        print("\n--- ButterFlowNet (Channel-Wise + Dual Mode) ---")
        print(f"Input: LMS={lms.shape}, PAN={pan.shape}")
        print(f"Output: {out.shape}")
        
        if torch.isnan(out).any():
            print("Warning: NaN detected!")
        else:
            print("Success: Stable.")
            
    except Exception as e:
        import traceback
        traceback.print_exc()

    if profile is not None:
        flops, params = profile(model, inputs=(lms, pan), verbose=False)
        fl, pa = clever_format([flops, params], "%.3f")
        print(f"FLOPs: {fl}, Params: {pa}")