import torch.nn as nn
from . import block as B
import torch
import logging
import numpy as np



class IRNet_2(nn.Module):
    def __init__(self, in_nc=3, nf=64, num_modules=2, out_nc=3, upscale=4):
        super(IRNet_2, self).__init__()

        self.fea_conv = B.conv_layer(in_nc, nf, kernel_size=1)

        # IRBs
        self.IRB1 = B.IRB_add(in_channels=nf)
        self.IRB2 = B.IRB_add(in_channels=nf)


        self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)


        self.upsampler = B.conv_layer(nf, 3, kernel_size=3)


    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.IRB1(out_fea)
        out_B2 = self.IRB2(out_B1)
        out_B = self.c(torch.cat([out_B1, out_B2], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea
        output = self.upsampler(out_lr)
        return output

class IRNet_1(nn.Module):
    def __init__(self, in_nc=3, nf=64, num_modules=1, out_nc=3, upscale=4):
        super(IRNet_1, self).__init__()

        self.fea_conv = B.conv_layer(in_nc, nf, kernel_size=1)

        # There just 1 IRB)
        self.IRB1 = B.IRB_add(in_channels=nf)  # use add!!!!!!!!
        
        self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)


        self.upsampler = B.conv_layer(nf, 3, kernel_size=3)


    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.IRB1(out_fea)

        # out_B = self.c(torch.cat(out_B1, dim=1))
        out_B = self.c(out_B1)
        
        out_lr = self.LR_conv(out_B) + out_fea
        output = self.upsampler(out_lr)
        return output