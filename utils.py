import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision.transforms as transforms
from torch import mm
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
#from skimage.measure import compare_ssim
import random
import numpy
#import xlrd
import os

def get_patch1(ms, hs, gt, patch_size, scale, ix=-1, iy=-1):
    ih = ms.shape[-2]
    iw = ms.shape[-1]

    ip = patch_size
    tp = scale*patch_size
    
    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale*ix, scale*iy)    

    ms_patch = ms[:,:,ix:ix + ip,iy:iy + ip].contiguous()
    #if np.any(np.isnan(ms_patch)):
    #    print("image contain non tensor")
    #    sys.exit()
    hs_patch = hs[:,:,tx:tx + tp,ty:ty + tp].contiguous()
    gt_patch = gt[:,:,tx:tx + tp,ty:ty + tp].contiguous()
                
    return ms_patch, hs_patch, gt_patch

def get_patch(img_in, img_in1,img_tar, patch_size):
        h, w = img_in.shape[:2]

        stride = patch_size

        x = random.randint(0, w - stride)
        y = random.randint(0, h - stride)

        img_in = img_in[y:y + stride, x:x + stride, :]
        img_in1 = img_in1[y:y + stride, x:x + stride, :]
        img_tar = img_tar[y:y + stride, x:x + stride, :]

        return img_in,img_in1, img_tar

def mixup_data(ms, pan, gt, alpha=1.0, use_cuda=True):
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam=1.
    batch_size = ms.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    mixed_ms = lam * ms + (1 - lam) * ms[index,:]
    mixed_pan = lam * pan + (1 - lam) * pan[index,:]
    mixed_gt = lam * gt + (1 - lam) * gt[index,:]
    return mixed_ms, mixed_pan, mixed_gt

def determine_conv_functional(n_dim, transposed=False):
    if n_dim == 1:
        if not transposed:
            return nn.functional.conv1d
        else:
            return nn.functional.conv_transposed1d
    elif n_dim == 2:
        if not transposed:
            return nn.functional.conv2d
        else:
            return nn.functional.conv_transposed2d
    elif n_dim == 3:
        if not transposed:
            return nn.functional.conv3d
        else:
            return nn.functional.conv_transposed3d
    else:
        NotImplementedError("No convolution of this dimensionality implemented")

def get_spectral_response(xls_path):
    if not os.path.exists(xls_path):
        raise Exception("Spectral response path does not exist!")
    data = xlrd.open_workbook(xls_path)
    table = data.sheets()[0]
    num_cols = table.ncols
    num_cols_sta = 1
    cols_list = [np.array(table.col_values(i)).reshape(-1,1) for i in range(num_cols_sta,num_cols)]
    sp_data = np.concatenate(cols_list, axis=1)
    sp_data = sp_data / (sp_data.sum(axis=0))
    return sp_data

def mixup(x1, x2, y1, y2, alpha=1.0, use_cuda=True):
    ''' returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        alpha = 1
        
    mixed_x = lam * x1 + (1 - lam) * x2
    mixed_y = lam * y1 + (1 - lam) * y2
    
    return mixed_x, mixed_y

def permute_channel(z):
    index = random.randrange(1,7)
    if index == 1:
        channel_index = [0,1,2]
    elif index == 2:
        channel_index = [0,2,1]
    elif index == 3:
        channel_index = [1,0,2]
    elif index == 4:
        channel_index = [1,2,0]
    elif index == 5:
        channel_index = [2,0,1]
    else:
        channel_index = [2,1,0]
        
    #channel_index = np.arange(0, 3, 1)
    #shuffle_index = np.random.shuffle(channel_index)
    return z[:,channel_index,:,:], index

class norm(nn.Module):
    def __init__(self):
        super(norm, self).__init__()
    def forward(self, x):
        tensor_mean = torch.mean(x, dim=2, keepdim=True)
        tensor_mean = torch.mean(tensor_mean, dim=3, keepdim=True)
        #tensor_max, _ = torch.max(x, dim=2, keepdim=True)
        #tensor_max, _ = torch.max(tensor_max, dim=3, keepdim=True)
        norm_x = x - tensor_mean
        return norm_x

def crop_channel(num_channel, num_spectral, x):
    spectral_index = list(np.arange(0, 31, 1))
    sample_index = random.sample(spectral_index, num_channel)
    sample_index.sort()
    im_patch = x[:, sample_index, :, :]
    return im_patch

class Random_crop_channel(nn.Module):
    
    def __init__(self, num_channel, num_spectral):
        super(Random_crop_channel, self).__init__()
        self.num_channel = num_channel
        self.num_spectral = num_spectral
    def forward(self, x):
        #perm_index = torch.randperm(self.num_channel)
        #x = x[perm_index]
        #spectral_index = np.arange(0, 31, 1)
        #spectral_index = np.random.shuffle(spectral_index)
        #print(spectral_index)
        #x = x[:,spectral_index,:,:]
        
        index = random.randrange(0, self.num_spectral - self.num_channel + 1)
        im_patch = x[:, index:index + self.num_channel, :, :]
        return im_patch

class Random_crop_tensor(nn.Module):
    
    def __init__(self, opt):
        super(Random_crop_tensor, self).__init__()
        self.crop_size = opt.patch_size
        
    def forward(self, x):
        w = x.shape[-2]
        h = x.shape[-1]
        
        ix = random.randrange(0, w - self.crop_size + 1)
        iy = random.randrange(0, h - self.crop_size + 1)
            
        im_patch = x[:, :, ix:ix + self.crop_size, iy:iy + self.crop_size]
        return im_patch
'''
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        alpha = 1
        
    mixed_x = lam * x[:1, :] + (1 - lam) * x[1, :]
    mixed_y = lam * y[:1, :] + (1 - lam) * y[1, :]
    
    #perm = torch.randperm(mixed_x.size(1)).cuda()
    #mixed_x = mixed_x[:, perm, :, :]
    return mixed_x, mixed_y
'''
def mixup_data2(x, y, alpha=1.0, batch_size=4, use_cuda=True):
    ''' returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        alpha = 1
        
    mixed_x = lam * x[:batch_size//2, :] + (1 - lam) * x[batch_size//2, :]
    mixed_y = lam * y[:batch_size//2, :] + (1 - lam) * y[batch_size//2, :]
    
    #perm = torch.randperm(mixed_x.size(1)).cuda()
    #mixed_x = mixed_x[:, perm, :, :]
    return mixed_x, mixed_y

def get_perm_label(x):
    spectral_list = torch.linspace(0,30,31)
    split_size = [3,3,3,3,3,3,3,3,3,4]
    split_list = torch.split(spectral_list, split_size)
    perm_list = []
    for list_patch in split_list:
        perm = torch.randperm(len(list_patch))
        list_patch = list_patch[perm]
        perm_list.append(list_patch)
    perm_list = torch.cat(perm_list, 0).long().cuda()
    perm_x = x[:,perm_list,:,:]
    return perm_x

def perm_channel(x):
    spectral_list = torch.linspace(0,30,31)
    split_size = [3,3,3,3,3,3,3,3,3,4]
    split_list = torch.split(spectral_list, split_size)
    perm_list = []
    for list_patch in split_list:
        perm = torch.randperm(len(list_patch))
        list_patch = list_patch[perm]
        perm_list.append(list_patch)
    perm_list = torch.cat(perm_list, 0).long().cuda()
    perm_x = x[:,perm_list,:,:]
    return perm_x


def compute_ergas(out, gt):
    num_spectral = out.shape[-1]
    out = np.reshape(out, (-1, num_spectral)) 
    gt = np.reshape(gt, (-1, num_spectral))
    diff = gt - out
    mse = np.mean(np.square(diff), axis=0)
    gt_mean = np.mean(gt, axis=0)
    mse = np.reshape(mse, (num_spectral,1))
    gt_mean = np.reshape(gt_mean, (num_spectral,1))
    ergas = 100/4*np.sqrt(np.mean(mse/(gt_mean**2+1e-6)))
    return ergas

def compute_sam(im1, im2):
    num_spectral = im1.shape[-1]
    im1 = np.reshape(im1, (-1, num_spectral))
    im2 = np.reshape(im2, (-1, num_spectral))
    mole = np.sum(np.multiply(im1, im2), axis=1)
    im1_norm = np.sqrt(np.sum(np.square(im1), axis=1))
    im2_norm = np.sqrt(np.sum(np.square(im2), axis=1))
    deno = np.multiply(im1_norm, im2_norm)
    
    sam = np.rad2deg(np.arccos((mole)/(deno+1e-7)))
    sam = np.mean(sam)
    return sam

def compute_ssim(im1, im2):
    n = im1.shape[2]
    ms_ssim=0.0
    for i in range(n):
        single_ssim = compare_ssim(im1[:,:,i], im2[:,:,i], data_range=1)
        ms_ssim += single_ssim
    return ms_ssim/n

def print_network(net):
    num_params = 0
    #for param in net.parameters():
    #    num_params += param.numel()
    num_params = sum([param.nelement() for param in net.parameters()])
    print(net)
    print('Total number of parameters: %d' % num_params)
    

def cal_psnr(im1, im2):
    num_spectral = im1.shape[-1]
    im1 = np.reshape(im1, (-1, num_spectral ))
    im2 = np.reshape(im2, (-1, num_spectral ))
    diff = im1 - im2

    mse = np.mean(np.square(diff), axis=0)
    

    return np.mean(10 * np.log10(1/mse))
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('ConvTranspose2d') != -1:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()

def map2tensor(gray_map):
    """Move gray maps to GPU, no normalization is done"""
    return torch.FloatTensor(gray_map).unsqueeze(0).unsqueeze(0).cuda()
'''
def create_gradient_map(im, window=5, percent=.97):
    """Create a gradient map of the image blurred with a rect of size window and clips extreme values"""
    # Calculate gradients
    gx, gy = np.gradient(rgb2gray(im))
    # Calculate gradient magnitude
    gmag, gx, gy = np.sqrt(gx ** 2 + gy ** 2), np.abs(gx), np.abs(gy)
    # Pad edges to avoid artifacts in the edge of the image
    gx_pad, gy_pad, gmag = pad_edges(gx, int(window)), pad_edges(gy, int(window)), pad_edges(gmag, int(window))
    lm_x, lm_y, lm_gmag = clip_extreme(gx_pad, percent), clip_extreme(gy_pad, percent), clip_extreme(gmag, percent)
    # Sum both gradient maps
    grads_comb = lm_x / lm_x.sum() + lm_y / lm_y.sum() + gmag / gmag.sum()
    # Blur the gradients and normalize to original values
    loss_map = convolve2d(grads_comb, np.ones(shape=(window, window)), 'same') / (window ** 2)
    # Normalizing: sum of map = numel
    return loss_map / np.mean(loss_map)
'''