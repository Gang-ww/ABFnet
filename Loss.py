# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 11:22:31 2020

@author: ww
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
#from util import map2tensor

IS_HIGH_VERSION = tuple(map(int, torch.__version__.split('+')[0].split('.'))) > (1, 7, 1)
if IS_HIGH_VERSION:
    import torch.fft

class L_GT_Grad_PS(nn.Module):
    def __init__(self):
        super(L_GT_Grad_PS, self).__init__()
        self.sobelconv=Sobelxy()
    def forward(self, image_A, image_B): # image_A: PAN, image_B, MS image
        Loss_gradient = 0.0
        for i in range(image_B.shape[1]):
            gradient_A = self.sobelconv(image_A)
            gradient_A = gradient_A
            gradient_A = torch.clamp(gradient_A,0,1)
            gradient_B = self.sobelconv(image_B[:,i,:,:].unsqueeze(1))
            Loss_gradient += F.l1_loss(gradient_A, gradient_B)
            #Loss_gradient += self.mefssim(gradient_A, gradient_B)
        return Loss_gradient

class L_GT_Grad(nn.Module):
    def __init__(self):
        super(L_GT_Grad, self).__init__()
        self.sobelconv=Sobelxy()
    def forward(self, image_A, image_B):
        Loss_gradient = 0.0
        for i in range(image_A.shape[1]):
            gradient_A = self.sobelconv(image_A[:,i,:,:].unsqueeze(1))
            gradient_A = gradient_A
            gradient_A = torch.clamp(gradient_A,0,1)
            gradient_B = self.sobelconv(image_B[:,i,:,:].unsqueeze(1))
            Loss_gradient += F.l1_loss(gradient_A, gradient_B)
            #Loss_gradient += self.mefssim(gradient_A, gradient_B)
        return Loss_gradient

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

class Cbloss(nn.Module):
    def __init__(self, eps=1e-6):
        super(Cbloss, self).__init__()
        self.eps = eps
    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss

class FocalFrequencyLoss(nn.Module):
    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1, ave_spectrum=False, log_matrix=False, batch_matrix=False):
        super(FocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix
        
    def tensor2freq(self, x):
        patch_factor = self.patch_factor
        _,_,h,w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, (
            'patch factor should be divisible by image height and width')
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch_list.append(x[:,:,i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w])
        y = torch.stack(patch_list,1)
        
        if IS_HIGH_VERSION:
            freq = torch.fft.fft2(y, norm='ortho')
            freq = torch.stack([freq.real, freq.imag], -1)
        else:
            freq = torch.rfft(y,2,onesided=False, normalizerd=True)
        return freq
        
    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        matrix_temp = (recon_freq - real_freq) ** 2
        matrix_temp = torch.sqrt(matrix_temp[..., 0] + matrix_temp[..., 1]) ** self.alpha
            
        if self.log_matrix:
            matrix_temp = torch.log(matrix_temp + 1.0)
        if self.batch_matrix:
            matrix_temp = matrix_temp / matrix_temp.max()
        else:
            matrix_temp = matrix_temp / matrix_temp.max(-1).values.max(-1).values[:,:,:,None, None]
        matrix_temp[torch.isnan(matrix_temp)] = 0.0
        matrix_temp = torch.clamp(matrix_temp, min=0.0, max=1.0)
        weight_matrix = matrix_temp.clone().detach()
        
        temp = (recon_freq - real_freq) ** 2
        freq_distance = temp[..., 0] + temp[..., 1]
        loss = weight_matrix * freq_distance
        return torch.mean(loss)
        
    def forward(self, pred, target, matrix=None, **kwargs):
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)
        
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)
        return self.loss_formulation(pred_freq, target_freq, matrix) * self.loss_weight
        
class ergas_loss(torch.nn.Module):
    def __init__(self):
        super(ergas_loss, self).__init__()
        
    def forward(self, out, gt):
        #num_spectral = im1.shape[-1]
        #im1 = torch.reshape(im1, (-1, num_spectral)) 
        #out = torch.reshape(im2, (-1, num_spectral))
        diff = gt - out
        diff = diff.pow(2)
        mse = torch.mean(diff, dim=(2,3), keepdim=True)
        #mse = torch.mean(mse, axis=3, keepdim=True)
        gt_mean = torch.mean(gt, dim=(2,3), keepdim=True)
        #out_mean = torch.mean(out_mean, axis=3, keepdim=True)
        ergas = 100/4*torch.sqrt(torch.mean(mse/(gt_mean**2+1e-8)))
        return ergas     

class My_P_loss(torch.nn.Module):
    def __init__(self):
        super(My_P_loss, self).__init__()
        
    def forward(self, x_u, x_d, y):
        loss1 = torch.mean(((y - x_u).pow(2)).mul(torch.exp(-x_d)) + x_d)
        loss2 = torch.mean(x_d)
        loss = loss1 + loss2
        return loss                                                                                                                  

class My_P_loss1(torch.nn.Module):
    def __init__(self):
        super(My_P_loss1, self).__init__()
        
    def forward(self, x_u, x_d, y):
        loss1 = torch.mean((torch.abs(y - x_u)).mul(torch.exp(-x_d)) + x_d)
        #loss2 = torch.mean(x_d)
        #loss = loss1 + loss2
        return loss1

class Spectral_GANLoss(torch.nn.Module):
    def __init__(self, opt):
        super(Spectral_GANLoss, self).__init__()
        shape = [1, opt.num_spectral, 1, 1]
        self.label = Variable(torch.ones(shape).cuda(), requires_grad=False)
    def forward(self, x, is_real):
        norm_top = torch.mean(torch.mean(x.mul(self.label), -1), -1)
        norm_out = torch.sqrt(torch.mean(torch.mean(x**2,-1),-1))
        norm_label = torch.sqrt(torch.mean(torch.mean(self.label**2,-1),-1))
        if is_real:
            angle_loss = torch.mean(torch.acos(torch.div(norm_top, norm_out.mul(norm_label)+1e-6)))/3.142
        else:
            angle_loss = torch.mean(1 - torch.acos(torch.div(norm_top, norm_out.mul(norm_label)+1e-6))/3.142)
        return angle_loss

class MS_GANLoss(nn.Module):
    
    def __init__(self, opt):
        super(MS_GANLoss, self).__init__()
        self.loss = nn.L1Loss(reduction='mean')
        shape = [1, opt.num_spectral, 1, 1]
        self.label_real = Variable(torch.ones(shape).cuda(), requires_grad=False)
        self.label_fake = Variable(torch.zeros(shape).cuda(), requires_grad=False)
        
    def forward(self, x, is_real):

        if is_real:
            label_tensor = self.label_real
        else:
            label_tensor = self.label_fake
                
        return self.loss(x, label_tensor)


class HS_GANLoss(nn.Module):
    
    def __init__(self, opt):
        super(HS_GANLoss, self).__init__()
        self.loss = nn.L1Loss(reduction='mean')
        shape = [1, 1, opt.patch_size*opt.upscale_factor, opt.patch_size*opt.upscale_factor]
        self.label_real = Variable(torch.ones(shape).cuda(), requires_grad=False)
        self.label_fake = Variable(torch.zeros(shape).cuda(), requires_grad=False)
        
    def forward(self, x, is_real):

        if is_real:
            label_tensor = self.label_real
        else:
            label_tensor = self.label_fake
                
        return self.loss(x, label_tensor)

class GANLoss(nn.Module):
    
    def __init__(self, opt):
        super(GANLoss, self).__init__()
        self.loss = nn.L1Loss(reduction='mean')
        shape_g = [1, 1, opt.patch_size*opt.upscale_factor, opt.patch_size*opt.upscale_factor]
        shape_d_real = [1, 1, opt.patch_size, opt.patch_size]
        self.label_g_real = Variable(torch.ones(shape_g).cuda(), requires_grad=False)
        self.label_g_fake = Variable(torch.zeros(shape_g).cuda(), requires_grad=False)
        self.label_d_real = Variable(torch.ones(shape_d_real).cuda(), requires_grad=False)
    
    def forward(self, x, is_real, is_g):
        if is_g:
            if is_real:
                label_tensor = self.label_g_real
            else:
                label_tensor = self.label_g_fake
        else:
            label_tensor = self.label_d_real
                
        return self.loss(x, label_tensor)

class weight_l1_loss(torch.nn.Module):
    def __init__(self):
        super(weight_l1_loss, self).__init__()
        self.l1 = torch.nn.L1Loss()
        
    def forward(self, x, y):
        mean_y = torch.mean(y, dim=(2,3), keepdim=True)
        #min_y,_ = torch.min(min_y, dim=3, keepdim=True)
        m = torch.mean(y,dim=(1,2,3), keepdim=True)
        loss = self.l1(x*(m/mean_y).pow(2), y*(m/mean_y).pow(2))
        return loss

class weight_l2_loss(torch.nn.Module):
    def __init__(self):
        super(weight_l2_loss, self).__init__()
        self.l2 = torch.nn.MSELoss()
        
    def forward(self, x, y):
        mean_y = torch.mean(y, dim=(2,3), keepdim=True)
        #min_y,_ = torch.min(min_y, dim=3, keepdim=True)
        m = torch.mean(y)
        loss = self.l2(x/mean_y*m, y/mean_y*m)
        return loss
        
class l1_loss(torch.nn.Module):
    def __init__(self):
        super(l1_loss, self).__init__()
        self.l1 = torch.nn.L1Loss()
        
    def forward(self, x, y):
        loss = self.l1(x, y)
        return loss

class sum_loss(torch.nn.Module):
    def __init__(self):
        super(sum_loss, self).__init__()
        
    def forward(self, x):
        loss = torch.mean(torch.abs(x))
        return loss

class sam_loss2(torch.nn.Module):
    def __init__(self):
        super(sam_loss2, self).__init__()
        
    def forward(self, x, y):
        norm_top = torch.mean(torch.mean(x.mul(y), -1), -1)
        norm_out = torch.sqrt(torch.mean(torch.mean(x**2,-1),-1))
        norm_label = torch.sqrt(torch.mean(torch.mean(y**2,-1),-1))
        angle_loss = torch.mean(torch.acos(torch.div(norm_top, norm_out.mul(norm_label)+1e-8)))
        return angle_loss

class angle_loss(torch.nn.Module):
    def __init__(self):
        super(angle_loss, self).__init__()
        
    def forward(self, x, y):
        x = torch.clamp(x,0,1)
        y = torch.clamp(y,0,1)
        norm_top = torch.sum(x.mul(y), 1)
        norm_out = torch.sqrt(torch.sum(x**2, 1))
        norm_label = torch.sqrt(torch.sum(y**2, 1))
        loss = torch.mean(torch.div(norm_top, norm_out.mul(norm_label)+1e-5))
        return loss
    
class sam_loss(torch.nn.Module):
    def __init__(self):
        super(sam_loss, self).__init__()
        
    def forward(self, x, y):
        norm_top = torch.sum(x.mul(y), 1)
        norm_out = torch.sqrt(torch.sum(x**2, 1))
        norm_label = torch.sqrt(torch.sum(y**2, 1))
        angle_loss = torch.mean(torch.acos(torch.div(norm_top, norm_out.mul(norm_label)+1e-5)))
        return angle_loss

class SumOfWeightsLoss(nn.Module):
    """ Encourages the kernel G is imitating to sum to 1 """

    def __init__(self):
        super(SumOfWeightsLoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, kernel):
        return self.loss(torch.ones(1).to(kernel.device), torch.sum(kernel))


class CentralizedLoss(nn.Module):
    """ Penalizes distance of center of mass from K's center"""

    def __init__(self, k_size, scale_factor=.5):
        super(CentralizedLoss, self).__init__()
        self.indices = Variable(torch.arange(0., float(k_size)).cuda(), requires_grad=False)/1e2
        wanted_center_of_mass = k_size // 2 + 0.5 * (int(1 / scale_factor) - k_size % 2)/1e2
        self.center = Variable(torch.FloatTensor([wanted_center_of_mass, wanted_center_of_mass]).cuda(), requires_grad=False)
        self.loss = nn.MSELoss()

    def forward(self, kernel):
        """Return the loss over the distance of center of mass from kernel center """
        r_sum, c_sum = torch.sum(kernel, dim=1).reshape(1, -1), torch.sum(kernel, dim=0).reshape(1, -1)
        return self.loss(torch.stack((torch.matmul(r_sum, self.indices) / torch.sum(kernel),
                                      torch.matmul(c_sum, self.indices) / torch.sum(kernel))), self.center)

class CentralizedLoss1(nn.Module):
    """ Penalizes distance of center of mass from K's center"""

    def __init__(self, k_size, scale_factor=.5):
        super(CentralizedLoss1, self).__init__()
        self.indices = Variable(torch.arange(0., float(k_size)).cuda(), requires_grad=False)
        wanted_center_of_mass = k_size // 2 + 0.5 * (int(1 / scale_factor) - k_size % 2)
        self.center = Variable(torch.FloatTensor([wanted_center_of_mass, wanted_center_of_mass]).cuda(), requires_grad=False)
        self.loss = nn.MSELoss()

    def forward(self, kernel):
        """Return the loss over the distance of center of mass from kernel center """
        r_sum, c_sum = torch.sum(kernel, dim=1).reshape(1, -1), torch.sum(kernel, dim=0).reshape(1, -1)
        return self.loss(torch.stack((torch.matmul(r_sum, self.indices) / torch.sum(kernel),
                                      torch.matmul(c_sum, self.indices) / torch.sum(kernel))), self.center)

'''
class BoundariesLoss(nn.Module):
    """ Encourages sparsity of the boundaries by penalizing non-zeros far from the center """

    def __init__(self, k_size):
        super(BoundariesLoss, self).__init__()
        self.mask = map2tensor(create_penalty_mask(k_size, 30))
        self.zero_label = Variable(torch.zeros(k_size).cuda(), requires_grad=False)
        self.loss = nn.L1Loss()

    def forward(self, kernel):
        return self.loss(kernel * self.mask, self.zero_label)
'''

class Non_Negetive_Loss(nn.Module):
    """ Penalizes small values to encourage sparsity """
    def __init__(self):
        super(Non_Negetive_Loss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, kernel):
        mask = torch.ge(kernel, 0)
        kernel = kernel.masked_fill(mask, 0)
        return self.loss(torch.abs(kernel), torch.zeros_like(kernel))

class SparsityLoss(nn.Module):
    """ Penalizes small values to encourage sparsity """
    def __init__(self):
        super(SparsityLoss, self).__init__()
        self.power = 0.5
        self.loss = nn.L1Loss()

    def forward(self, kernel):
        return self.loss(torch.abs(kernel) ** self.power, torch.zeros_like(kernel))

    
