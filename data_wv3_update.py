import torch.utils.data as data
import torch
import h5py
import numpy as np
import scipy.io as sio
import random
class Dataset_WV2_FT(data.Dataset):
    def __init__(self, file_path):
        super(Dataset_WV2_FT, self).__init__()
        
        data = h5py.File(file_path)  # NxCxHxW = 0x1x2x3
        # for h5
        # tensor type:
        gt1 = data["gt"][...]  # convert to np tpye for CV2.filter
        gt1 = np.array(gt1, dtype=np.float32) / 2047.0
        self.gt = torch.from_numpy(gt1)  # NxCxHxW:
        print("number of wv3 data", self.gt.shape)
        ms1 = data["ms"][...]  # convert to np tpye for CV2.filter
        ms1 = np.array(ms1, dtype=np.float32) / 2047.0
        self.ms = torch.from_numpy(ms1)

        lms1 = data["lms"][...]  # convert to np tpye for CV2.filter
        lms1 = np.array(lms1, dtype=np.float32) / 2047.0
        self.lms = torch.from_numpy(lms1)


        pan1 = data['pan'][...]  # Nx1xHxW
        pan1 = np.array(pan1, dtype=np.float32) / 2047.0  # Nx1xHxW
        self.pan = torch.from_numpy(pan1)  # Nx1xHxW:
        
    #####必要函数
    def __getitem__(self, index):
        gt = self.gt[index, :, :, :].float()
        #print(gt.shape)
        pan = self.pan[index, :, :, :].float()
        ms = self.ms[index, :, :, :].float()
        lms = self.lms[index, :, :, :].float()
        pan,ms,lms,gt = self.get_patch(pan,ms,lms,gt,64)
        return gt.float(), ms.float(), lms.float(), pan.float()
            #####必要函数
    def __len__(self):
        return self.gt.shape[0]
    
    def get_patch(self, pan, ms, lms, gt, patch_size):
        h, w = ms.shape[-2:]
        #print(h, w)
        stride = patch_size
        x = random.randint(0, w - stride)
        y = random.randint(0, h - stride)
        pan = pan[:,y:y + stride*4, x:x + stride*4]
        ms = ms[:,y:y + stride, x:x + stride]
        lms = lms[:,y:y + stride*4, x:x + stride*4]
        gt = gt[:,y:y + stride*4, x:x + stride*4]

        #print(gt.shape,pan.shape,ms.shape,lms.shape)
        return pan, ms, lms, gt

class Dataset_Pro_h5(data.Dataset):
    def __init__(self, file_path):
        super(Dataset_Pro_h5, self).__init__()
        
        #self.ms, self.pan, self.lms, self.gt = load_setmat(file_path)
        
        data = h5py.File(file_path)  # NxCxHxW = 0x1x2x3
        # for h5
        # tensor type:
        gt1 = data["gt"][...]  # convert to np tpye for CV2.filter
        gt1 = np.array(gt1, dtype=np.float32) / 2047.0
        self.gt = torch.from_numpy(gt1)  # NxCxHxW:

        ms1 = data["ms"][...]  # convert to np tpye for CV2.filter
        ms1 = np.array(ms1, dtype=np.float32) / 2047.0
        self.ms = torch.from_numpy(ms1)

        lms1 = data["lms"][...]  # convert to np tpye for CV2.filter
        lms1 = np.array(lms1, dtype=np.float32) / 2047.0
        self.lms = torch.from_numpy(lms1)


        pan1 = data['pan'][...]  # Nx1xHxW
        pan1 = np.array(pan1, dtype=np.float32) / 2047.0  # Nx1xHxW
        self.pan = torch.from_numpy(pan1)  # Nx1xHxW:
        
        
    #####必要函数
    def __getitem__(self, index):
        return self.gt[index, :, :, :].float(), \
               self.ms[index, :, :, :].float(), \
               self.lms[index, :, :, :].float(), \
               self.pan[index, :, :, :].float()

            #####必要函数
    def __len__(self):
        return self.gt.shape[0]

class Dataset_GF_Pro(data.Dataset):
    def __init__(self, file_path):
        super(Dataset_GF_Pro, self).__init__()
        
        data = h5py.File(file_path)  # NxCxHxW = 0x1x2x3
        # for h5
        # tensor type:
        gt1 = data["gt"][...]  # convert to np tpye for CV2.filter
        gt1 = np.array(gt1, dtype=np.float32) / 2047.0
        self.gt = torch.from_numpy(gt1)  # NxCxHxW:
        self.gt = self.gt.permute(0,3,1,2)

        ms1 = data["ms"][...]  # convert to np tpye for CV2.filter
        ms1 = np.array(ms1, dtype=np.float32) / 2047.0
        self.ms = torch.from_numpy(ms1)
        self.ms = self.ms.permute(0,3,1,2)

        lms1 = data["lms"][...]  # convert to np tpye for CV2.filter
        lms1 = np.array(lms1, dtype=np.float32) / 2047.0
        self.lms = torch.from_numpy(lms1)
        self.lms = self.lms.permute(0,3,1,2)
        
        pan1 = data['pan'][...]  # Nx1xHxW
        pan1 = np.array(pan1, dtype=np.float32) / 2047.0  # Nx1xHxW
        self.pan = torch.from_numpy(pan1)  # Nx1xHxW:
        self.pan = self.pan.unsqueeze(1)
        #print(self.gt.shape, self.ms.shape, self.lms.shape, self.pan.shape)
        
        
        
    #####必要函数
    def __getitem__(self, index):
        return self.gt[index, :, :, :].float(), \
               self.ms[index, :, :, :].float(), \
               self.lms[index, :, :, :].float(), \
               self.pan[index, :, :, :].float()

            #####必要函数
    def __len__(self):
        return self.gt.shape[0]

class Dataset_GF(data.Dataset): # 
    def __init__(self, file_path):
        super(Dataset_GF, self).__init__()
        
        data = sio.loadmat(file_path)
        lms1 = data['lms'][...]  # NxCxHxW = 4x8x512x512
        lms1 = np.array(lms1, dtype=np.float32) #0/ 2047.0
        self.lms = torch.from_numpy(lms1).permute(0,3,1,2)

        pan1 = data['pan'][...]  # NxCxHxW = 4x8x512x512
        pan1 = np.array(pan1, dtype=np.float32) #/ 2047.0
        self.pan = torch.from_numpy(pan1)

        ms1 = data['ms'][...]  # NxCxHxW = 4x8x512x512
        ms1 = np.array(ms1, dtype=np.float32) #/ 2047.0
        self.ms = torch.from_numpy(ms1).permute(0,3,1,2)

        gt1 = data['gt'][...]  # NxCxHxW = 4x8x512x512
        gt1 = np.array(gt1, dtype=np.float32) #/ 2047.0
        self.gt = torch.from_numpy(gt1).permute(0,3,1,2)

        self.pan = self.pan.unsqueeze(1)

        print(self.ms.size(),self.pan.size(),self.lms.size(),self.gt.size())
    
    #####必要函数
    def __getitem__(self, index):
        return self.gt[index, :, :, :].float(), \
               self.ms[index, :, :, :].float(), \
               self.lms[index, :, :, :].float(), \
               self.pan[index, :, :, :].float()

            #####必要函数
    def __len__(self):
        return self.gt.shape[0]

class Dataset_QB(data.Dataset): # 
    def __init__(self, file_path):
        super(Dataset_QB, self).__init__()
        
        data = sio.loadmat(file_path)
        lms1 = data['lms'][...]  # NxCxHxW = 4x8x512x512
        lms1 = np.array(lms1, dtype=np.float32) / 2047.0
        self.lms = torch.from_numpy(lms1).permute(0,3,1,2)

        pan1 = data['pan'][...]  # NxCxHxW = 4x8x512x512
        pan1 = np.array(pan1, dtype=np.float32) / 2047.0
        self.pan = torch.from_numpy(pan1)

        ms1 = data['ms'][...]  # NxCxHxW = 4x8x512x512
        ms1 = np.array(ms1, dtype=np.float32) / 2047.0
        self.ms = torch.from_numpy(ms1).permute(0,3,1,2)

        gt1 = data['gt'][...]  # NxCxHxW = 4x8x512x512
        gt1 = np.array(gt1, dtype=np.float32) / 2047.0
        self.gt = torch.from_numpy(gt1).permute(0,3,1,2)

        self.pan = self.pan.unsqueeze(1)

        print(self.ms.size(),self.pan.size(),self.lms.size(),self.gt.size())
    
    #####必要函数
    def __getitem__(self, index):
        return self.gt[index, :, :, :].float(), \
               self.ms[index, :, :, :].float(), \
               self.lms[index, :, :, :].float(), \
               self.pan[index, :, :, :].float()

            #####必要函数
    def __len__(self):
        return self.gt.shape[0]


class Dataset_US(data.Dataset):
    def __init__(self, file_path):
        super(Dataset_US, self).__init__()
        
        data = h5py.File(file_path)  # NxCxHxW = 0x1x2x3

        # tensor type:
        gt1 = data["gt"][...]  # convert to np tpye for CV2.filter
        gt1 = np.array(gt1, dtype=np.float32) / 2047.0
        self.gt = torch.from_numpy(gt1)  # NxCxHxW:

        ms1 = data["ms"][...]  # convert to np tpye for CV2.filter
        ms1 = np.array(ms1, dtype=np.float32) / 2047.0
        self.ms = torch.from_numpy(ms1)

        lms1 = data["lms"][...]  # convert to np tpye for CV2.filter
        lms1 = np.array(lms1, dtype=np.float32) / 2047.0
        self.lms = torch.from_numpy(lms1)


        pan1 = data['pan'][...]  # Nx1xHxW
        pan1 = np.array(pan1, dtype=np.float32) / 2047.0  # Nx1xHxW
        self.pan = torch.from_numpy(pan1)  # Nx1xHxW:
        
    #####必要函数
    def __getitem__(self, index):
        return self.gt[index, :, :, :].float(), \
               self.ms[index, :, :, :].float(), \
               self.lms[index, :, :, :].float(), \
               self.pan[index, :, :, :].float()

            #####必要函数
    def __len__(self):
        return self.gt.shape[0]

class Dataset_Pro_Eval_Dual1(data.Dataset):
    def __init__(self, file_path):
        super(Dataset_Pro_Eval_Dual1, self).__init__()
        
        self.ms, self.pan, self.lms, self.l_ms, self.l_pan, self.l_lms = load_setmat_dual1(file_path)
        '''
        data = h5py.File(file_path)  # NxCxHxW = 0x1x2x3

        # tensor type:
        gt1 = data["gt"][...]  # convert to np tpye for CV2.filter
        gt1 = np.array(gt1, dtype=np.float32) / 2047.0
        self.gt = torch.from_numpy(gt1)  # NxCxHxW:

        ms1 = data["ms"][...]  # convert to np tpye for CV2.filter
        ms1 = np.array(ms1, dtype=np.float32) / 2047.0
        self.ms = torch.from_numpy(ms1)

        lms1 = data["lms"][...]  # convert to np tpye for CV2.filter
        lms1 = np.array(lms1, dtype=np.float32) / 2047.0
        self.lms = torch.from_numpy(lms1)


        pan1 = data['pan'][...]  # Nx1xHxW
        pan1 = np.array(pan1, dtype=np.float32) / 2047.0  # Nx1xHxW
        self.pan = torch.from_numpy(pan1)  # Nx1xHxW:
        '''
    #####必要函数
    def __getitem__(self, index):
        return self.ms[index, :, :, :].float(), \
               self.lms[index, :, :, :].float(), \
               self.pan[index, :, :, :].float(), \
               self.l_ms[index, :, :, :].float(),\
               self.l_pan[index, :, :, :].float(),\
               self.l_lms[index, :, :, :].float()

            #####必要函数
    def __len__(self):
        return self.lms.shape[0]

class Train_Dual_QB(data.Dataset):
    def __init__(self, file_path):
        super(Train_Dual_QB, self).__init__()
        
        self.ms, self.pan, self.lms, self.l_ms, self.l_pan, self.l_lms = load_setmat_dual_qb(file_path)
    #####必要函数
    def __getitem__(self, index):
        return self.ms[index, :, :, :].float(), \
               self.lms[index, :, :, :].float(), \
               self.pan[index, :, :, :].float(), \
               self.l_ms[index, :, :, :].float(),\
               self.l_pan[index, :, :, :].float(),\
               self.l_lms[index, :, :, :].float()

            #####必要函数
    def __len__(self):
        return self.l_ms.shape[0]

class Eval_Dual_QB(data.Dataset):
    def __init__(self, file_path):
        super(Eval_Dual_QB, self).__init__()
        
        self.ms, self.pan, self.lms, self.l_ms, self.l_pan, self.l_lms = load_eval_dual_qb(file_path)
    #####必要函数
    def __getitem__(self, index):
        return self.ms[index, :, :, :].float(), \
               self.lms[index, :, :, :].float(), \
               self.pan[index, :, :, :].float(), \
               self.l_ms[index, :, :, :].float(),\
               self.l_pan[index, :, :, :].float(),\
               self.l_lms[index, :, :, :].float()

            #####必要函数
    def __len__(self):
        return self.l_ms.shape[0]

class Dataset_Pro_Eval_Dual(data.Dataset):
    def __init__(self, file_path):
        super(Dataset_Pro_Eval_Dual, self).__init__()
        
        self.ms, self.pan, self.lms, self.l_ms, self.l_pan, self.l_lms = load_setmat_dual_qb(file_path)
        # self.ms, self.pan, self.lms, self.l_ms, self.l_pan = load_setmat_dual(file_path)
        '''
        data = h5py.File(file_path)  # NxCxHxW = 0x1x2x3

        # tensor type:
        gt1 = data["gt"][...]  # convert to np tpye for CV2.filter
        gt1 = np.array(gt1, dtype=np.float32) / 2047.0
        self.gt = torch.from_numpy(gt1)  # NxCxHxW:

        ms1 = data["ms"][...]  # convert to np tpye for CV2.filter
        ms1 = np.array(ms1, dtype=np.float32) / 2047.0
        self.ms = torch.from_numpy(ms1)

        lms1 = data["lms"][...]  # convert to np tpye for CV2.filter
        lms1 = np.array(lms1, dtype=np.float32) / 2047.0
        self.lms = torch.from_numpy(lms1)


        pan1 = data['pan'][...]  # Nx1xHxW
        pan1 = np.array(pan1, dtype=np.float32) / 2047.0  # Nx1xHxW
        self.pan = torch.from_numpy(pan1)  # Nx1xHxW:
        '''
    #####必要函数
    def __getitem__(self, index):
        return self.ms[index, :, :, :].float(), \
               self.lms[index, :, :, :].float(), \
               self.pan[index, :, :, :].float(), \
               self.l_ms[index, :, :, :].float(),\
               self.l_pan[index, :, :, :].float(),\
               self.l_lms[index, :, :, :].float()

            #####必要函数
    def __len__(self):
        return self.l_ms.shape[0]

class Dataset_Pro_Eval_Dual2(data.Dataset):
    def __init__(self, file_path):
        super(Dataset_Pro_Eval_Dual2, self).__init__()
        
        self.ms, self.pan, self.lms, self.l_ms, self.l_pan = load_setmat_dual_qb1(file_path)
        '''
        data = h5py.File(file_path)  # NxCxHxW = 0x1x2x3

        # tensor type:
        gt1 = data["gt"][...]  # convert to np tpye for CV2.filter
        gt1 = np.array(gt1, dtype=np.float32) / 2047.0
        self.gt = torch.from_numpy(gt1)  # NxCxHxW:

        ms1 = data["ms"][...]  # convert to np tpye for CV2.filter
        ms1 = np.array(ms1, dtype=np.float32) / 2047.0
        self.ms = torch.from_numpy(ms1)

        lms1 = data["lms"][...]  # convert to np tpye for CV2.filter
        lms1 = np.array(lms1, dtype=np.float32) / 2047.0
        self.lms = torch.from_numpy(lms1)


        pan1 = data['pan'][...]  # Nx1xHxW
        pan1 = np.array(pan1, dtype=np.float32) / 2047.0  # Nx1xHxW
        self.pan = torch.from_numpy(pan1)  # Nx1xHxW:
        '''
    #####必要函数
    def __getitem__(self, index):
        return self.ms[index, :, :, :].float(), \
               self.lms[index, :, :, :].float(), \
               self.pan[index, :, :, :].float(), \
               self.l_ms[index, :, :, :].float(),\
               self.l_pan[index, :, :, :].float(),\
               #self.l_lms[index, :, :, :].float()

            #####必要函数
    def __len__(self):
        return self.l_ms.shape[0]

class Dataset_Pro_Eval_Full(data.Dataset):
    def __init__(self, file_path):
        super(Dataset_Pro_Eval_Full, self).__init__()
        
        self.ms, self.pan, self.lms = load_setmat_full(file_path)
        '''
        data = h5py.File(file_path)  # NxCxHxW = 0x1x2x3

        # tensor type:
        gt1 = data["gt"][...]  # convert to np tpye for CV2.filter
        gt1 = np.array(gt1, dtype=np.float32) / 2047.0
        self.gt = torch.from_numpy(gt1)  # NxCxHxW:

        ms1 = data["ms"][...]  # convert to np tpye for CV2.filter
        ms1 = np.array(ms1, dtype=np.float32) / 2047.0
        self.ms = torch.from_numpy(ms1)

        lms1 = data["lms"][...]  # convert to np tpye for CV2.filter
        lms1 = np.array(lms1, dtype=np.float32) / 2047.0
        self.lms = torch.from_numpy(lms1)


        pan1 = data['pan'][...]  # Nx1xHxW
        pan1 = np.array(pan1, dtype=np.float32) / 2047.0  # Nx1xHxW
        self.pan = torch.from_numpy(pan1)  # Nx1xHxW:
        '''
    #####必要函数
    def __getitem__(self, index):
        return self.ms[index, :, :, :].float(), \
               self.lms[index, :, :, :].float(), \
               self.pan[index, :, :, :].float()
               #self.lms1[index, :, :, :].float()

            #####必要函数
    def __len__(self):
        return self.lms.shape[0]


class Dataset_Eval_1258(data.Dataset):
    def __init__(self, file_path):
        super(Dataset_Eval_1258, self).__init__()

        self.ms, self.pan, self.lms, self.gt = load_evalmat(file_path)

    #####必要函数
    def __getitem__(self, index):
        return self.gt[index, :, :, :].float(), \
            self.ms[index, :, :, :].float(), \
            self.lms[index, :, :, :].float(), \
            self.pan[index, :, :, :].float()

        #####必要函数

    def __len__(self):
        return self.gt.shape[0]

class Dataset_Pro_Eval(data.Dataset):
    def __init__(self, file_path):
        super(Dataset_Pro_Eval, self).__init__()

        data = h5py.File(file_path)  # NxCxHxW = 0x1x2x3

        # tensor type:
        gt1 = data["gt"][...]  # convert to np tpye for CV2.filter
        gt1 = np.array(gt1, dtype=np.float32) / 2047.0
        self.gt = torch.from_numpy(gt1)  # NxCxHxW:

        ms1 = data["ms"][...]  # convert to np tpye for CV2.filter
        ms1 = np.array(ms1, dtype=np.float32) / 2047.0
        self.ms = torch.from_numpy(ms1)

        lms1 = data["lms"][...]  # convert to np tpye for CV2.filter
        lms1 = np.array(lms1, dtype=np.float32) / 2047.0
        self.lms = torch.from_numpy(lms1)


        pan1 = data['pan'][...]  # Nx1xHxW
        pan1 = np.array(pan1, dtype=np.float32) / 2047.0  # Nx1xHxW
        self.pan = torch.from_numpy(pan1)  # Nx1xHxW:
    #####必要函数
    def __getitem__(self, index):
        return self.gt[index, :, :, :].float(), \
               self.ms[index, :, :, :].float(), \
               self.lms[index, :, :, :].float(), \
               self.pan[index, :, :, :].float()

            #####必要函数
    def __len__(self):
        return self.gt.shape[0]

class Dataset_Eval_WV3_78(data.Dataset):
    def __init__(self, file_path):
        super(Dataset_Eval_WV3_78, self).__init__()

        data = h5py.File(file_path)  # NxCxHxW = 0x1x2x3

        # tensor type:
        gt1 = data["gt"][...]  # convert to np tpye for CV2.filter
        gt1 = np.array(gt1, dtype=np.float32) / 2047.0
        self.gt = torch.from_numpy(gt1)  # NxCxHxW:

        ms1 = data["ms"][...]  # convert to np tpye for CV2.filter
        ms1 = np.array(ms1, dtype=np.float32) / 2047.0
        self.ms = torch.from_numpy(ms1)

        lms1 = data["lms"][...]  # convert to np tpye for CV2.filter
        lms1 = np.array(lms1, dtype=np.float32) / 2047.0
        self.lms = torch.from_numpy(lms1)

        pan1 = data['pan'][...]  # Nx1xHxW
        pan1 = np.array(pan1, dtype=np.float32) / 2047.0  # Nx1xHxW
        self.pan = torch.from_numpy(pan1)  # Nx1xHxW:
    def __getitem__(self, index):
        return self.gt[index, :, :, :].float(), \
               self.ms[index, :, :, :].float(), \
               self.lms[index, :, :, :].float(), \
               self.pan[index, :, :, :].float()

            #####必要函数
    def __len__(self):
        return self.gt.shape[0]


def load_setmat(file_path): #从1258.mat中读取数据
    #data = h5py.File(file_path)  #
    data = sio.loadmat(file_path)
    lms1 = data['lms'][...]  # NxCxHxW = 4x8x512x512
    lms1 = np.array(lms1, dtype=np.float32) / 2047.0
    lms = torch.from_numpy(lms1)

    pan1 = data['pan'][...]  # NxCxHxW = 4x8x512x512
    pan1 = np.array(pan1, dtype=np.float32) / 2047.0
    pan = torch.from_numpy(pan1)

    ms1 = data['ms'][...]  # NxCxHxW = 4x8x512x512
    ms1 = np.array(ms1, dtype=np.float32) / 2047.0
    ms = torch.from_numpy(ms1)

    gt1 = data['gt'][...]  # NxCxHxW = 4x8x512x512
    gt1 = np.array(gt1, dtype=np.float32) / 2047.0
    gt = torch.from_numpy(gt1)
    Nn, Wn, Hn, Cn = gt.shape
    #ms = ms.permute(0, 3, 1, 2)
    #print(pan.shape, ms.shape, gt.shape)
    # for qb
    #pan = pan.unsqueeze(1)
    #lms = lms.permute(0, 3, 1, 2)
    #gt = gt.permute(0, 3, 1, 2)
    #ms = ms.permute(0, 3, 1, 2)
    '''
    # for transfer data
    ms = ms.permute(3,2,0,1)
    lms = lms.permute(3,2,0,1)
    gt = gt.permute(3,2,0,1)
    pan = pan.unsqueeze(0)
    pan = pan.permute(3,0,1,2)
    print(pan.shape, ms.shape, gt.shape)
    
    '''
    return ms, pan, lms, gt

def load_evalmat(file_path): #从1258.mat中读取数据
    data = h5py.File(file_path)  #
    #data = sio.loadmat(file_path)
    lms1 = data['lms'][...]  # NxCxHxW = 4x8x512x512
    lms1 = np.array(lms1, dtype=np.float32) / 2047.0
    lms = torch.from_numpy(lms1)

    pan1 = data['pan'][...]  # NxCxHxW = 4x8x512x512
    pan1 = np.array(pan1, dtype=np.float32) / 2047.0
    pan = torch.from_numpy(pan1)
    
    ms1 = data['ms'][...]  # NxCxHxW = 4x8x512x512
    ms1 = np.array(ms1, dtype=np.float32) / 2047.0
    ms = torch.from_numpy(ms1)

    gt1 = data['gt'][...]  # NxCxHxW = 4x8x512x512
    gt1 = np.array(gt1, dtype=np.float32) / 2047.0
    gt = torch.from_numpy(gt1)
    # for transfer data and 1258
    # start
    
    pan = pan.unsqueeze(1)
    ms = ms.permute(0,3,1,2)
    lms = lms.permute(0,3,1,2)
    gt = gt.permute(0,3,1,2)
    
    # end
    return ms, pan, lms, gt

def load_qb(file_path):
    #data = h5py.File(file_path)  #
    data = sio.loadmat(file_path)
    lms1 = data['lms'][...]  # NxCxHxW = 4x8x512x512
    lms1 = np.array(lms1, dtype=np.float32)
    lms = torch.from_numpy(lms1)

    pan1 = data['pan'][...]  # NxCxHxW = 4x8x512x512
    pan1 = np.array(pan1, dtype=np.float32)
    pan = torch.from_numpy(pan1)

    ms1 = data['ms'][...]  # NxCxHxW = 4x8x512x512
    ms1 = np.array(ms1, dtype=np.float32)
    ms = torch.from_numpy(ms1)

    gt1 = data['gt'][...]  # NxCxHxW = 4x8x512x512
    gt1 = np.array(gt1, dtype=np.float32)
    gt = torch.from_numpy(gt1)
    Nn, Wn, Hn, Cn = gt.shape
    ms = ms.permute(0, 3, 1, 2)
    
    #print(pan.shape, ms.shape, gt.shape)
    pan = pan.reshape(Nn,Wn,Hn,1).permute(0, 3, 1, 2)
    gt = gt.permute(0, 3, 1, 2)
    lms = lms.permute(0, 3, 1, 2)


    return ms, pan, lms, gt

def load_setmat_single(file_path): #从1258.mat中读取数据
    #data = h5py.File(file_path)  #
    data = sio.loadmat(file_path)
    lms1 = data['lms'][...]  # NxCxHxW = 4x8x512x512
    lms1 = np.array(lms1, dtype=np.float32) / 2047.0
    lms = torch.from_numpy(lms1)
    lms = lms.unsqueeze(0)

    pan1 = data['pan'][...]  # NxCxHxW = 4x8x512x512
    pan1 = np.array(pan1, dtype=np.float32) / 2047.0
    pan = torch.from_numpy(pan1)
    pan = pan.unsqueeze(0)
    
    ms1 = data['ms'][...]  # NxCxHxW = 4x8x512x512
    ms1 = np.array(ms1, dtype=np.float32) / 2047.0
    ms = torch.from_numpy(ms1)
    ms = ms.unsqueeze(0)

    gt1 = data['gt'][...]  # NxCxHxW = 4x8x512x512
    gt1 = np.array(gt1, dtype=np.float32) / 2047.0
    gt = torch.from_numpy(gt1)
    gt = gt.unsqueeze(0)    
    
    #print(gt.shape)
    Nn, Wn, Hn, Cn = gt.shape
    #ms = ms.permute(0, 3, 1, 2)

    pan = pan.reshape(Nn,Wn,Hn,1).permute(0, 3, 1, 2)

    lms = lms.permute(0, 3, 1, 2)


    return ms, pan, lms, gt

def load_setmat_full(file_path): #从1258.mat中读取数据
    data = h5py.File(file_path)  #
    #data = sio.loadmat(file_path)
    lms1 = data['lms'][...]  # NxCxHxW = 4x8x512x512
    lms1 = np.array(lms1, dtype=np.float32) / 2047.0
    lms = torch.from_numpy(lms1)
    #lms = lms.unsqueeze(0)

    pan1 = data['pan'][...]  # NxCxHxW = 4x8x512x512
    pan1 = np.array(pan1, dtype=np.float32) / 2047.0
    pan = torch.from_numpy(pan1)
    pan = pan.unsqueeze(0)
    
    ms1 = data['ms'][...]  # NxCxHxW = 4x8x512x512
    ms1 = np.array(ms1, dtype=np.float32) / 2047.0
    ms = torch.from_numpy(ms1)
    #ms = ms.unsqueeze(0)
    
    print(lms.shape, ms.shape, pan.shape)
    #print(pan.shape)
    
    Nn, Wn, Hn, Cn = lms.shape
    ms = ms.permute(0, 3, 1, 2)
    
    pan = pan.permute(1, 0, 2, 3)

    lms = lms.permute(0, 3, 1, 2)
    

    return ms, pan, lms

def load_setmat_full_inv(file_path): #从1258.mat中读取数据
    #data = h5py.File(file_path)  #
    data = sio.loadmat(file_path)
    lms1 = data['lms'][...]  # NxCxHxW = 4x8x512x512
    lms1 = np.array(lms1, dtype=np.float32) / 2047.0
    lms = torch.from_numpy(lms1)
    #lms = lms.unsqueeze(0)

    pan1 = data['pan'][...]  # NxCxHxW = 4x8x512x512
    pan1 = np.array(pan1, dtype=np.float32) / 2047.0
    pan = torch.from_numpy(pan1)
    pan = pan.unsqueeze(0)
    
    ms1 = data['ms'][...]  # NxCxHxW = 4x8x512x512
    ms1 = np.array(ms1, dtype=np.float32) / 2047.0
    ms = torch.from_numpy(ms1)
    #ms = ms.unsqueeze(0)
    
    print(lms.shape, ms.shape, pan.shape)
    #print(lms.shape)
    '''
    Nn, Wn, Hn, Cn = lms.shape

    ms = ms.permute(0, 3, 1, 2)

    

    pan = pan.reshape(Nn,Wn*2,Hn*2,1).permute(0, 3, 1, 2)



    lms = lms.permute(0, 3, 1, 2)
    '''
    # for qb full
    lms2 = data['lms1'][...]  # NxCxHxW = 4x8x512x512
    lms2 = np.array(lms2, dtype=np.float32) / 2047.0
    lms1 = torch.from_numpy(lms2)
    ms = ms.permute(3,2,0, 1)
    pan = pan.permute(3, 0, 1, 2)
    lms = lms.permute(3,2, 0, 1)
    lms1 = lms1.permute(3,2, 0, 1)
    

    return ms, pan, lms, lms1
def load_setmat_dual_qb(file_path): #从1258.mat中读取数据
    #data = h5py.File(file_path)  #
    data = sio.loadmat(file_path)
    lms1 = data['lms'][...]  # NxCxHxW = 4x8x512x512
    lms1 = np.array(lms1, dtype=np.float32) / 2047.0
    lms = torch.from_numpy(lms1)
    #lms = lms.unsqueeze(0)

    pan1 = data['pan'][...]  # NxCxHxW = 4x8x512x512
    pan1 = np.array(pan1, dtype=np.float32) / 2047.0
    pan = torch.from_numpy(pan1)
    #pan = pan.unsqueeze(0)
    
    ms1 = data['ms'][...]  # NxCxHxW = 4x8x512x512
    ms1 = np.array(ms1, dtype=np.float32) / 2047.0
    ms = torch.from_numpy(ms1)
    #ms = ms.unsqueeze(0)
    
    l_ms1 = data['l_ms'][...]  # NxCxHxW = 4x8x512x512
    l_ms1 = np.array(l_ms1, dtype=np.float32) / 2047.0
    l_ms = torch.from_numpy(l_ms1)
    
    l_lms1 = data['l_lms'][...]  # NxCxHxW = 4x8x512x512
    l_lms1 = np.array(l_lms1, dtype=np.float32) / 2047.0
    l_lms = torch.from_numpy(l_lms1)
    
    l_pan1 = data['l_pan'][...]  # NxCxHxW = 4x8x512x512
    l_pan1 = np.array(l_pan1, dtype=np.float32) / 2047.0
    l_pan = torch.from_numpy(l_pan1)
    #l_pan = l_pan.unsqueeze(0)
    
    Nn, Wn, Hn, Cn = lms.shape
    #ms = ms.permute(0, 3, 1, 2)
    # for qb
    # for train
    '''
    pan = pan.squeeze(0)
    l_pan = l_pan.squeeze(0)
    
    pan = pan.permute(3, 2, 0, 1)
    ms = ms.permute(3,2, 0, 1)
    lms = lms.permute(3, 2, 0, 1)
    l_ms = l_ms.permute(3, 2, 0, 1)
    l_pan = l_pan.permute(3, 2, 0, 1)
    l_lms = l_lms.permute(3, 2, 0, 1)
    '''
    # for test
    pan = pan.permute(3, 2, 0, 1)
    ms = ms.permute(3,2, 0, 1)
    lms = lms.permute(3,2, 0, 1)
    l_ms = l_ms.permute(3, 2,0, 1)
    l_pan = l_pan.permute(3, 2, 0, 1)
    l_lms = l_lms.permute(3, 2, 0, 1)
    
    
    return ms, pan, lms, l_ms, l_pan, l_lms

def load_eval_dual_qb(file_path):
    data = sio.loadmat(file_path)
    lms1 = data['lms'][...]  # NxCxHxW = 4x8x512x512
    lms1 = np.array(lms1, dtype=np.float32) / 2047.0
    lms = torch.from_numpy(lms1)

    pan1 = data['pan'][...]  # NxCxHxW = 4x8x512x512
    pan1 = np.array(pan1, dtype=np.float32) / 2047.0
    pan = torch.from_numpy(pan1)
    pan = pan.unsqueeze(2)
    
    ms1 = data['ms'][...]  # NxCxHxW = 4x8x512x512
    ms1 = np.array(ms1, dtype=np.float32) / 2047.0
    ms = torch.from_numpy(ms1)
    
    l_ms1 = data['l_ms'][...]  # NxCxHxW = 4x8x512x512
    l_ms1 = np.array(l_ms1, dtype=np.float32) / 2047.0
    l_ms = torch.from_numpy(l_ms1)
    
    l_lms1 = data['l_lms'][...]  # NxCxHxW = 4x8x512x512
    l_lms1 = np.array(l_lms1, dtype=np.float32) / 2047.0
    l_lms = torch.from_numpy(l_lms1)
    
    l_pan1 = data['l_pan'][...]  # NxCxHxW = 4x8x512x512
    l_pan1 = np.array(l_pan1, dtype=np.float32) / 2047.0
    l_pan = torch.from_numpy(l_pan1)
    l_pan = l_pan.unsqueeze(2)
    
    Nn, Wn, Hn, Cn = lms.shape
    # for test
    #print(pan.shape,l_pan.shape)
    pan = pan.permute(3, 2, 0, 1)
    ms = ms.permute(3,2, 0, 1)
    lms = lms.permute(3,2, 0, 1)
    l_ms = l_ms.permute(3, 2,0, 1)
    l_pan = l_pan.permute(3, 2, 0, 1)
    l_lms = l_lms.permute(3, 2, 0, 1)
    
    
    return ms, pan, lms, l_ms, l_pan, l_lms

def load_setmat_dual_qb1(file_path): #从1258.mat中读取数据
    #data = h5py.File(file_path)  #
    data = sio.loadmat(file_path)
    lms1 = data['lms'][...]  # NxCxHxW = 4x8x512x512
    lms1 = np.array(lms1, dtype=np.float32) / 2047.0
    lms = torch.from_numpy(lms1)
    #lms = lms.unsqueeze(0)

    pan1 = data['pan'][...]  # NxCxHxW = 4x8x512x512
    pan1 = np.array(pan1, dtype=np.float32) / 2047.0
    pan = torch.from_numpy(pan1)
    #pan = pan.unsqueeze(0)
    
    ms1 = data['ms'][...]  # NxCxHxW = 4x8x512x512
    ms1 = np.array(ms1, dtype=np.float32) / 2047.0
    ms = torch.from_numpy(ms1)
    #ms = ms.unsqueeze(0)
    
    l_ms1 = data['l_ms'][...]  # NxCxHxW = 4x8x512x512
    l_ms1 = np.array(l_ms1, dtype=np.float32) / 2047.0
    l_ms = torch.from_numpy(l_ms1)
    
    l_pan1 = data['l_pan'][...]  # NxCxHxW = 4x8x512x512
    l_pan1 = np.array(l_pan1, dtype=np.float32) / 2047.0
    l_pan = torch.from_numpy(l_pan1)
    #l_pan = l_pan.unsqueeze(0)
    
    Nn, Wn, Hn, Cn = lms.shape
    #ms = ms.permute(0, 3, 1, 2)
    # for qb
    print(pan.shape, ms.shape, lms.shape, l_ms.shape, l_pan.shape)
    # for train
    
    #pan = pan.squeeze(0)
    #l_pan = l_pan.squeeze(0)

    pan = pan.permute(3, 0, 1, 2)
    ms = ms.permute(3,2, 0, 1)
    lms = lms.permute(3, 2, 0, 1)
    l_ms = l_ms.permute(3, 2, 0, 1)
    l_pan = l_pan.permute(3, 0, 1, 2)
   
    return ms, pan, lms, l_ms, l_pan

def load_setmat_dual(file_path): #从1258.mat中读取数据
    #data = h5py.File(file_path)  #
    data = sio.loadmat(file_path)
    lms1 = data['lms'][...]  # NxCxHxW = 4x8x512x512
    lms1 = np.array(lms1, dtype=np.float32) / 2047.0
    lms = torch.from_numpy(lms1)
    #lms = lms.unsqueeze(0)

    pan1 = data['pan'][...]  # NxCxHxW = 4x8x512x512
    pan1 = np.array(pan1, dtype=np.float32) / 2047.0
    pan = torch.from_numpy(pan1)
    pan = pan.unsqueeze(0)
    
    ms1 = data['ms'][...]  # NxCxHxW = 4x8x512x512
    ms1 = np.array(ms1, dtype=np.float32) / 2047.0
    ms = torch.from_numpy(ms1)
    #ms = ms.unsqueeze(0)
    
    l_ms1 = data['l_ms'][...]  # NxCxHxW = 4x8x512x512
    l_ms1 = np.array(l_ms1, dtype=np.float32) / 2047.0
    l_ms = torch.from_numpy(l_ms1)
    
    
    l_pan1 = data['l_pan'][...]  # NxCxHxW = 4x8x512x512
    l_pan1 = np.array(l_pan1, dtype=np.float32) / 2047.0
    l_pan = torch.from_numpy(l_pan1)
    l_pan = l_pan.unsqueeze(0)
    
    Nn, Wn, Hn, Cn = lms.shape
    #ms = ms.permute(0, 3, 1, 2)
    # for qb
    #pan = pan.squeeze(0)
    #l_pan = l_pan.squeeze(0)
    #print(pan.shape, ms.shape, lms.shape, l_ms.shape, l_pan.shape)
    pan = pan.permute(3, 0, 2, 1)
    ms = ms.permute(3, 0, 2, 1)
    lms = lms.permute(3, 0, 2, 1)
    l_ms = l_ms.permute(3, 0, 2, 1)
    l_pan = l_pan.permute(3, 0, 2, 1)
    '''
    pan = pan.permute(3,2, 0, 1)
    ms = ms.permute(3,2, 0, 1)
    lms = lms.permute(3,2, 0, 1)
    l_ms = l_ms.permute(3, 2,0, 1)
    l_pan = l_pan.permute(3,2, 0, 1)
    '''
    print(pan.shape, ms.shape, lms.shape, l_ms.shape, l_pan.shape)

    return ms, pan, lms, l_ms, l_pan
    
def load_setmat_dual1(file_path): 
    #data = h5py.File(file_path)  #
    data = sio.loadmat(file_path)
    lms1 = data['lms'][...]  # NxCxHxW = 4x8x512x512
    lms1 = np.array(lms1, dtype=np.float32) / 2047.0
    lms = torch.from_numpy(lms1)
    #lms = lms.unsqueeze(0)

    pan1 = data['pan'][...]  # NxCxHxW = 4x8x512x512
    pan1 = np.array(pan1, dtype=np.float32) / 2047.0
    pan = torch.from_numpy(pan1)
    pan = pan.unsqueeze(0)
    
    ms1 = data['ms'][...]  # NxCxHxW = 4x8x512x512
    ms1 = np.array(ms1, dtype=np.float32) / 2047.0
    ms = torch.from_numpy(ms1)
    #ms = ms.unsqueeze(0)
    
    l_ms1 = data['l_ms'][...]  # NxCxHxW = 4x8x512x512
    l_ms1 = np.array(l_ms1, dtype=np.float32) / 2047.0
    l_ms = torch.from_numpy(l_ms1)
    
    l_lms1 = data['l_lms'][...]  # NxCxHxW = 4x8x512x512
    l_lms1 = np.array(l_lms1, dtype=np.float32) / 2047.0
    l_lms = torch.from_numpy(l_lms1)
    
        
    
    l_pan1 = data['l_pan'][...]  # NxCxHxW = 4x8x512x512
    l_pan1 = np.array(l_pan1, dtype=np.float32) / 2047.0
    l_pan = torch.from_numpy(l_pan1)
    l_pan = l_pan.unsqueeze(0)
    
    Nn, Wn, Hn, Cn = lms.shape
    #ms = ms.permute(0, 3, 1, 2)
    
    #print(ms.shape, lms.shape, l_ms.shape, l_pan.shape)
    pan = pan.permute(3, 0, 1, 2)
    ms = ms.permute(3, 0, 1, 2)
    lms = lms.permute(3, 0, 1, 2)
    l_ms = l_ms.permute(3, 0, 1, 2)
    l_pan = l_pan.permute(3, 0, 1, 2)
    l_lms = l_lms.permute(3, 0, 1, 2)

    return ms, pan, lms, l_ms, l_pan, l_lms
    
def load_set(file_path): # case1 从h5中读取test数据 case2 从.mat中读取单图的数据

    ## ===== case1: NxCxHxW
    # data = h5py.File(file_path)
    # ms1 = data["ms"][...]  # NxCxHxW=0,1,2,3
    # shape_size = len(ms1.shape)

    ## ===== case2: HxWxC
    data = sio.loadmat(file_path)  #
    ms1 = data["I_MS_LR"][...]  # NxCxHxW=0,1,2,3
    shape_size = len(ms1.shape)

    if shape_size == 4:  # NxCxHxW
        # tensor type:
        lms1 = data['lms'][...]  # NxCxHxW = 4x8x512x512
        lms1 = np.array(lms1, dtype=np.float32) / 2047.0
        lms = torch.from_numpy(lms1)

        pan1 = data['pan'][...]  # NxCxHxW = 4x8x512x512
        pan1 = np.array(pan1, dtype=np.float32) / 2047.0
        pan = torch.from_numpy(pan1)

        ms1 = data['ms'][...]  # NxCxHxW = 4x8x512x512
        ms1 = np.array(ms1, dtype=np.float32) / 2047.0
        ms = torch.from_numpy(ms1)

        return ms, pan, lms

    if shape_size == 3:  # HxWxC
        # tensor type:
        lms1 = data['I_MS'][...]  # HxWxC=0,1,2
        lms1 = np.expand_dims(lms1, axis=0)  # 1xHxWxC
        lms1 = np.array(lms1, dtype=np.float32) / 2047.0 # 1xHxWxC
        lms = torch.from_numpy(lms1).permute(0, 3, 1, 2)  # NxCxHxW  or HxWxC

        pan1 = data['I_PAN'][...]  # HxW
        pan1 = np.expand_dims(pan1, axis=0)  # 1xHxW
        pan1 = np.expand_dims(pan1, axis=3)  # 1xHxWx1
        pan1 = np.array(pan1, dtype=np.float32) / 2047.  # 1xHxWx1
        pan = torch.from_numpy(pan1).permute(0, 3, 1, 2)  # Nx1xHxW:

        ms1 = data['I_MS_LR'][...]  # HxWxC=0,1,2
        ms1 = np.expand_dims(ms1, axis=0)  # 1xHxWxC
        ms1 = np.array(ms1, dtype=np.float32) / 2047.0 # 1xHxWxC
        ms = torch.from_numpy(ms1).permute(0, 3, 1, 2)  # NxCxHxW  or HxWxC

        return ms, pan, lms
