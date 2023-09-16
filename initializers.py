import torch.nn as nn
import torch

initializer = ''

def choose_initializer(cfg):
    global initializer
    try:
        initializer = cfg.INITIALIZER.NAME
    except AttributeError:
        initializer = 'Normal'

def init_weight(m):
    classname = m.__class__.__name__
    if classname == 'Conv2d': # if classname.find('Conv') != -1:
        try:
            if initializer == 'Normal':
                nn.init.normal_(m.weight.data, mean=0.0, std=1e-11)
            elif initializer == 'Xavier Normal':
                nn.init.xavier_normal_(m.weight.data, gain=1.0)
            elif initializer == 'Kaiming Normal':
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
        except AttributeError:
            nn.init.normal_(m.weight.data, mean=0.0, std=1e-11)
        if not m.bias == None:
            nn.init.constant_(m.bias.data, 0.0)