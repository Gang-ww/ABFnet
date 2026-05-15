# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 08:50:53 2022

@author: ww
"""

from __future__ import print_function
import argparse

import numpy as np
import os
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
# 把旧的 from uf_former_4070 import  Unfolding_Net_V2 注释掉/删除
from maxpha_v47_token_v1 import ButterFlowNet_Anisotropic as ButterFlowNetFinal
from data_wv3_update import Dataset_GF
import scipy.io as sio
from utils import cal_psnr, compute_ssim, compute_ergas, compute_sam, print_network

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=8, help="super resolution upscale factor")
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', action='store_true', help='use gpu')
parser.add_argument('--no_gpu', action='store_false', dest='gpu_mode', help='use cpu')
parser.set_defaults(gpu_mode=torch.cuda.is_available())
parser.add_argument('--D_kernel_size', type=int, default=14, help='Starting Epoch')
parser.add_argument('--patch_size', type=int, default=64, help='Size of cropped HR image')
parser.add_argument('--data_augmentation', type=bool, default=False)
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--test_dataset', type=str, default=r'D:\AIproject\data\TestData_GF.mat')
#parser.add_argument('--output', default='result/', help='Location to save checkpoint models')
parser.add_argument('--model_type', type=str, default='IBP2')
parser.add_argument('--residual', type=bool, default=False)# 129.pth
parser.add_argument('--model', default='gf.pth', help='sr pretrained base model')#169GF，129QB，149


opt = parser.parse_args()
gpus_list=range(opt.gpus)
print(opt)
device = torch.device(opt.device)
cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")


torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
test_set = Dataset_GF(opt.test_dataset)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)


print('===> Building model')
checkpoint = torch.load(opt.model, map_location=device)
f_model = ButterFlowNetFinal(spectral_num=4, channels=48, num_groups=7, window_size=4).to(device)
#f_model = Unfolding_Net_V2(num_channel=1, base_filter=48, moe_factor=3, num_spectral=8).to(device)
#f_model = InvISPNet(channel_in=16, channel_out=16, block_num=10).to(device)
'''
f_model = TD_Net().to(device)
f_model.load_state_dict(checkpoint)

f_model = GPPNN(ms_channels=4,
           pan_channels=1, 
           n_feat=64,
           n_layer=8).to(device)
f_model.load_state_dict(checkpoint['f_model_state_dict'])
'''
f_model.load_state_dict(checkpoint['f_model_state_dict']) 
print_network(f_model) # 1042007, 968142
print('Pre-trained SR model is loaded.')


if cuda:
    f_model = f_model.cuda(gpus_list[0])

def eval():
    all_global_g = []
    all_local_g = []
    
    def hook_g(module, input, output):
        B = output.shape[0]
        C = output.shape[1] // 5
        params = output.view(B, C, 5, -1)
        gate = params[:, :, 4, :]
        g = torch.sigmoid(gate).detach().cpu().numpy().flatten()
        all_global_g.extend(g.tolist())

    def hook_l(module, input, output):
        B_N = output.shape[0]
        C = output.shape[1] // 5
        params = output.view(B_N, C, 5, -1)
        gate = params[:, :, 4, :]
        g = torch.sigmoid(gate).detach().cpu().numpy().flatten()
        all_local_g.extend(g.tolist())

    hooks = []
    for block in f_model.blocks:
        hooks.append(block.global_pred.register_forward_hook(hook_g))
        hooks.append(block.param_decoder.register_forward_hook(hook_l))

    i=1
    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_ergas = 0.0
    avg_sam = 0.0
    f_model.eval()
    for batch in testing_data_loader:
        with torch.no_grad():
            HSI,_, MS, HS = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), batch[3]
        
        # Scale exactly like Dataset_GF_Pro did before we had to revert to Dataset_GF:
        MS = MS.to(device) / 2047.0
        HS = HS.to(device) / 2047.0
        HSI = HSI.to(device) / 2047.0
        
        if i == 1:
            try:
                from thop import profile, clever_format
                flops, params = profile(f_model, inputs=(MS, HS), verbose=False)
                flops, params = clever_format([flops, params], "%.3f")
                print("===> FLOPs: {}, Params: {}".format(flops, params))
            except ImportError:
                print("===> thop is not installed, cannot calculate FLOPs. If needed, please 'pip install thop'.")
                
        with torch.no_grad():
            #print(MS.shape,HS.shape,HSI.shape)
            #MS = MS.permute(0,3,1,2)
            #k = calc_curr_k(opt, d_model)
            print(i)
            
            out_HSI = f_model(MS, HS)
            out_HSI = out_HSI.cpu().data.squeeze().clamp(0, 1).numpy().transpose(1,2,0)
            HSI = HSI.cpu().data.squeeze().clamp(0, 1).numpy().transpose(1,2,0)
            MS = MS.cpu().data.squeeze().clamp(0, 1).numpy().transpose(1,2,0)
            HS = HS.cpu().data.squeeze().clamp(0, 1).numpy()
            psnr = cal_psnr(out_HSI, HSI)
            avg_psnr = avg_psnr + psnr
            ergas = compute_ergas(HSI, out_HSI)
            avg_ergas += ergas
            ssim = compute_ssim(out_HSI, HSI)
            avg_ssim = avg_ssim + ssim
            sam = compute_sam(out_HSI, HSI)
            avg_sam = avg_sam + sam
            print("===> PSNR: {:.4f} dB || ssim: {:.4f} || ergas: {:.4f}, ||sam: {:.4f}".format(psnr, ssim, ergas, sam))
        
        if not os.path.exists('./result'):
            os.makedirs('./result')
        save_dir = './result/out'+str(i)+'.mat'
        sio.savemat(save_dir, {'out':out_HSI, 'gt':HSI, 'ms':MS, 'hs':HS})
        i = i+1
    print("Avg.PSNR: {:.4f} || Avg.SSIM: {:.4f} || Avg.ERGAS: {:.4f}, || Avg.SAM: {:.4f}".format(avg_psnr/len(testing_data_loader),avg_ssim/len(testing_data_loader),avg_ergas/len(testing_data_loader),avg_sam/len(testing_data_loader)))

    # Remove hooks
    for h in hooks:
        h.remove()

    def print_stats(name, data):
        data = np.array(data)
        print(f"--- {name} Branch g Distribution ---")
        print(f"Mean: {np.mean(data):.4f}")
        print(f"Std:  {np.std(data):.4f}")
        print(f"Min:  {np.min(data):.4f}")
        print(f"25%:  {np.percentile(data, 25):.4f}")
        print(f"50%:  {np.percentile(data, 50):.4f}")
        print(f"75%:  {np.percentile(data, 75):.4f}")
        print(f"Max:  {np.max(data):.4f}")
        print("-" * 30)

    print_stats("Global", all_global_g)
    print_stats("Local", all_local_g)


##Eval Start!!!!

if __name__ == '__main__':
    eval()
