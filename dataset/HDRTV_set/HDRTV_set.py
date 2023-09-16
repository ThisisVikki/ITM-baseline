import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import time
import os
import sys
import lpips
sys.path.insert(0, os.path.dirname(__file__))
import HDRTV_utils as util
import skimage.metrics as sm

class Dataset(data.Dataset):
    def __init__(self,args=None, dataset_train=None, cfg=None):
        self.args = args
        self.cfg = cfg
        self.dataset_train = dataset_train
        if dataset_train:
            if self.args.dataset=="SRITM":
                _,_,self.gt_files = util.traverse_under_folder(self.cfg.SRITM.TRAIN_DATAROOT_GT)
                self.gt_files.sort()
                _,_,self.lq_files = util.traverse_under_folder(self.cfg.SRITM.TRAIN_DATAROOT_LQ)
                self.lq_files.sort()
            elif self.args.dataset=="HSR":
                _,_,self.gt_files = util.traverse_under_folder(self.cfg.HSR.TRAIN_DATAROOT_GT)
                self.gt_files.sort()
                _,_,self.lq_files = util.traverse_under_folder(self.cfg.HSR.TRAIN_DATAROOT_LQ)
                self.lq_files.sort()
            elif self.args.dataset=="SSR":
                _,_,self.gt_files = util.traverse_under_folder(self.cfg.SSR.TRAIN_DATAROOT_GT)
                self.gt_files.sort()
                _,_,self.lq_files = util.traverse_under_folder(self.cfg.SSR.TRAIN_DATAROOT_LQ)
                self.lq_files.sort()
            else:
                _,_,self.gt_files = util.traverse_under_folder(self.cfg.ITM.TRAIN_DATAROOT_GT)
                self.gt_files.sort()
                _,_,self.lq_files = util.traverse_under_folder(self.cfg.ITM.TRAIN_DATAROOT_LQ)
                self.lq_files.sort()
        else:
            if self.args.dataset=="SRITM":
                _,_,self.gt_files = util.traverse_under_folder(self.cfg.SRITM.VALID_DATAROOT_GT)
                self.gt_files.sort()
                _,_,self.lq_files = util.traverse_under_folder(self.cfg.SRITM.VALID_DATAROOT_LQ)
                self.lq_files.sort()
            elif self.args.dataset=="HSR":
                _,_,self.gt_files = util.traverse_under_folder(self.cfg.HSR.VALID_DATAROOT_GT)
                self.gt_files.sort()
                _,_,self.lq_files = util.traverse_under_folder(self.cfg.HSR.VALID_DATAROOT_LQ)
                self.lq_files.sort()
            elif self.args.dataset=="SSR":
                _,_,self.gt_files = util.traverse_under_folder(self.cfg.SSR.VALID_DATAROOT_GT)
                self.gt_files.sort()
                _,_,self.lq_files = util.traverse_under_folder(self.cfg.SSR.VALID_DATAROOT_LQ)
                self.lq_files.sort()
            else:
                _,_,self.gt_files = util.traverse_under_folder(self.cfg.ITM.VALID_DATAROOT_GT)
                self.gt_files.sort()
                _,_,self.lq_files = util.traverse_under_folder(self.cfg.ITM.VALID_DATAROOT_LQ)
                self.lq_files.sort()

    def __getitem__(self, index):
        GT_path = self.gt_files[index]
        LQ_path = self.lq_files[index]
        # 归一化[0,1]  RGB->YUV
        if self.args.dataset=="SRITM":
            img_GT = cv2.cvtColor((cv2.imread(GT_path,cv2.IMREAD_UNCHANGED) / 65535).astype(np.float32), cv2.COLOR_BGR2YCrCb)
            img_LQ = cv2.cvtColor((cv2.imread(LQ_path,cv2.IMREAD_UNCHANGED) / 255).astype(np.float32), cv2.COLOR_BGR2YCrCb)
        elif self.args.dataset=="HSR":
            img_GT = cv2.cvtColor((cv2.imread(GT_path,cv2.IMREAD_UNCHANGED) / 65535).astype(np.float32), cv2.COLOR_BGR2YCrCb)
            img_LQ = cv2.cvtColor((cv2.imread(LQ_path,cv2.IMREAD_UNCHANGED) / 65535).astype(np.float32), cv2.COLOR_BGR2YCrCb)
        elif self.args.dataset=="SSR":
            img_GT = cv2.cvtColor((cv2.imread(GT_path,cv2.IMREAD_UNCHANGED) / 255).astype(np.float32), cv2.COLOR_BGR2YCrCb)
            img_LQ = cv2.cvtColor((cv2.imread(LQ_path,cv2.IMREAD_UNCHANGED) / 255).astype(np.float32), cv2.COLOR_BGR2YCrCb)
        else:
            img_GT = cv2.cvtColor((cv2.imread(GT_path,cv2.IMREAD_UNCHANGED) / 65535).astype(np.float32), cv2.COLOR_BGR2YCrCb)
            img_LQ = cv2.cvtColor((cv2.imread(LQ_path,cv2.IMREAD_UNCHANGED) / 255).astype(np.float32), cv2.COLOR_BGR2YCrCb)
        
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float().clamp(min=0, max=1)
        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ , (2, 0, 1)))).float().clamp(min=0, max=1)
        
        return img_LQ, img_GT

    def __len__(self):
        return len(self.gt_files)
    
    def __measure__(self, output, gt,metrics):
        # 四维的矩阵，0维表示图片的数量，这里是归一化后的
        outputBatch_yuvNumpy_bhwc = output.detach().cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1)
        gtBatch_yuvNumpy_bhwc = gt.detach().cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1)
        
        start_tme = time.time()

        output_yuvNumpy_hwc = outputBatch_yuvNumpy_bhwc[0,:,:,:]
        gt_yuvNumpy_hwc = gtBatch_yuvNumpy_bhwc[0,:,:,:]

        # 计算绿色通道的psnr+ssim
        metrics['psnr'].append(sm.peak_signal_noise_ratio(image_true=output_yuvNumpy_hwc[:,:,0], image_test=gt_yuvNumpy_hwc[:,:,0], data_range=1))
        metrics['ssim'].append(util.calculate_ssim(img=np.expand_dims(output_yuvNumpy_hwc[:, :, 0] * 255, axis=-1), img2=np.expand_dims(gt_yuvNumpy_hwc[:, :, 0] * 255, axis=-1)))
        endup_time = time.time()
        
        print('\rValidation with: ' + str(endup_time - start_tme) + ' (s) per inference ', end="")
        return metrics
