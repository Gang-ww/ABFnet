# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 20:44:56 2022

@author: ww
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from maxpha_v47_token_v1 import ButterFlowNet_Anisotropic as ButterFlowNetFinal
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.optim.lr_scheduler as lrs
from data_wv3_update import Dataset_GF_Pro
import socket

from utils import cal_psnr, print_network, compute_ssim, compute_ergas, compute_sam
from Loss import l1_loss, ergas_loss

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--batchSize', type=int, default=8, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=350, help='number of epochs to train for')
parser.add_argument('--Warm_Epochs', type=int, default=1200, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=5, help='Snapshots')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning Rate. Default=0.01')
parser.add_argument('--warm-lr', type=float, default=1.25e-5, help='Learning Rate. Default=0.01')
parser.add_argument('--skip_threshold', type=float, default=1e6, help='Learning Rate. Default=0.01')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=2, help='number of threads for data loader to use')
parser.add_argument('--decay', type=int, default='100', help='learning rate decay type')
parser.add_argument('--decay1', type=int, default='200', help='learning rate decay type')
parser.add_argument('--decay2', type=int, default='300', help='learning rate decay type')
parser.add_argument('--decay3', type=int, default='320', help='learning rate decay type')
parser.add_argument('--warm_decay', type=int, default='300', help='learning rate decay type')
parser.add_argument('--warm_gamma', type=float, default=2, help='learning rate decay factor for step decay')
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--data_augmentation', type=bool, default=False)
parser.add_argument('--image_dataset', type=str, default='data/train_gf.mat')
parser.add_argument('--test_dataset', type=str, default='data/validation_gf.mat')
parser.add_argument('--model_type', type=str, default='IBP2')
parser.add_argument('--residual', type=bool, default=False)
parser.add_argument('--patch_size', type=int, default=64, help='Size of cropped HR image')
parser.add_argument('--pretrained_sr', default='smartdsp3IBP2tpami_residual_filter8_epoch_799.pth', help='sr pretrained base model')
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--save_folder', default='weights_gf/', help='Location to save checkpoint models')

Seed=555
torch.manual_seed(Seed)
torch.cuda.manual_seed(Seed)
torch.cuda.manual_seed_all(Seed)
cudnn.deterministic = True


opt = parser.parse_args()

# Handle device selection
if opt.gpu_mode and torch.cuda.is_available():
    try:
        device = torch.device(opt.device)
        # Try a simple operation to verify device
        torch.cuda.get_device_properties(device)
    except (RuntimeError, AssertionError):
        print(f"Warning: Requested device {opt.device} is invalid. Falling back to cuda:0.")
        device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

hostname = str(socket.gethostname())
cudnn.benchmark = True
print(opt)
if not os.path.exists(opt.save_folder):
    os.makedirs(opt.save_folder)
    print(f"Created save folder: {opt.save_folder}")

args = []

def train(epoch):
    epoch_loss = 0
    for _, batch in enumerate(training_data_loader, 1):
        HSI,_, MS, HS = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(batch[3])
        if cuda:
            MS = MS.to(device)
            HS = HS.to(device)
            HSI = HSI.to(device)

        f_model.train()
        optimizer_f.zero_grad()
        out_HSI = f_model(MS,HS)
        HS_l1 = l1(out_HSI, HSI)
        HS_sam = 1-torch.mean(sam(HSI, out_HSI))
        l_ergas = ergas(out_HSI, HSI)
        loss = HS_l1
        loss.backward()
        nn.utils.clip_grad_norm_(f_model.parameters(), max_norm=2e-4, norm_type=2)
        optimizer_f.step()
        epoch_loss += loss.item()
        print("===> Epoch{}: Loss: {:.2e} || s_loss: {:.2e} || e_loss: {:.2e} || Learning rate: lr={}.".format(epoch, 
              loss.item(), 0.1*HS_sam, 1e-3*l_ergas, optimizer_f.param_groups[0]['lr']))
        #writer.add_scalar('loss', epoch_loss, epoch)

def test():
    avg_psnr = 0.0
    avg_ergas = 0.0
    avg_sam = 0.0
    avg_ssim = 0.0
    torch.set_grad_enabled(False)

    epoch = scheduler_f.last_epoch
    
    f_model.eval()
    print('\nEvaluation:')
     
    for batch in test_data_loader:
        with torch.no_grad():
            HSI, _, MS, HS = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(batch[3])
        if cuda:
            MS = MS.to(device)
            HS = HS.to(device)
            HSI = HSI.to(device)
            
            with torch.no_grad():
                out_HSI = f_model.forward(MS, HS)
                #print(out_HSI.shape)
                out_HSI = out_HSI.cpu().data.squeeze().clamp(0, 1).numpy().transpose(1,2,0)
                HSI = HSI.cpu().data.squeeze().clamp(0, 1).numpy().transpose(1,2,0)
                
                #print(out_HSI.shape, HSI.shape)
                avg_psnr += cal_psnr(out_HSI, HSI)
                avg_ergas += compute_ergas(out_HSI, HSI)
                avg_ssim += compute_ssim(out_HSI, HSI)
                avg_sam += compute_sam(out_HSI, HSI)
            
    avg_psnr = avg_psnr / len(test_data_loader)
    avg_ssim = avg_ssim / len(test_data_loader)
    avg_sam = avg_sam / len(test_data_loader)
    #print(avg_sam)
    avg_ergas = avg_ergas / len(test_data_loader)
    #writer.add_scalar('psnr', avg_psnr, epoch)
    if avg_psnr >= ckt['psnr']:
        ckt['epoch'] = epoch
        ckt['psnr'] = avg_psnr
    print("===> Avg.PSNR: {:.4f} dB || ssim: {:.4f} || ergas: {:.4f} || sam: {:.4f} || Best.PSNR: {:.4f} dB || Epoch: {}"
          .format(avg_psnr, avg_ssim, avg_ergas, avg_sam, ckt['psnr'], ckt['epoch']))
    #writer.add_scalar('best_psnr', ckt['psnr'], ckt['epoch'])
    torch.set_grad_enabled(True)


def checkpoint(epoch):
    model_out_path = opt.save_folder+hostname+opt.model_type+"_epoch_{}.pth".format(epoch)
    torch.save({
        'f_model_state_dict': f_model.state_dict()},
        model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

#writer = SummaryWriter(opt.save_folder)

print('===> Loading datasets')

train_set = Dataset_GF_Pro(opt.image_dataset)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, pin_memory=True, shuffle=True)
test_set = Dataset_GF_Pro(opt.test_dataset)
test_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, pin_memory=True, shuffle=True)
print('===> Building model ', opt.model_type)

#ButterFlowNetFinal(spectral_num=4, channels=96, num_groups=7).report_complexity()
# f_model = ButterFlowNetFinal(spectral_num=4, channels=96, num_groups=7).to(device)
f_model = ButterFlowNetFinal(spectral_num=4, channels=48, num_groups=7, window_size=4).to(device)
#f_model = ButterFlowNetFinal(spectral_num=4, channels=48, num_groups=7, window_size=4, proj_groups=4).to(device)
l1 = l1_loss().to(device)
sam = torch.nn.CosineSimilarity(dim=1, eps=1e-6).to(device)
ergas = ergas_loss().to(device)
print('---------- Networks architecture -------------')
print_network(f_model)

if opt.pretrained:
    model_name = os.path.join(opt.save_folder + opt.pretrained_sr)
    if os.path.exists(model_name):
        #model= torch.load(model_name, map_location=lambda storage, loc: storage)
        f_model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
        print('Pre-trained SR model is loaded.')

optimizer_f = optim.Adam(f_model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

milestones = []
for i in range(1, opt.nEpochs+1):
    if i == opt.decay:
        milestones.append(i)
    if i == opt.decay1:
        milestones.append(i)
    if i == opt.decay2:
        milestones.append(i)
    if i == opt.decay3:
        milestones.append(i)
        
    
scheduler_f = lrs.MultiStepLR(optimizer_f, milestones, opt.gamma)

#warm_scheduler = lrs.MultiStepLR(optimizer, warm_milestones, opt.warm_gamma)

#curr_k = torch.FloatTensor(opt.D_kernel_size, opt.D_kernel_size).cuda()
ckt = {'epoch':0, 'psnr':0.0} 

for epoch in range(opt.start_iter, opt.nEpochs + 1):
            
            train(epoch)
            #if epoch < opt.Warm_Epochs:
                #warm_scheduler.step()
            scheduler_f.step()
            if (epoch+1) % (opt.snapshots) == 0:
                checkpoint(epoch)
                test()
