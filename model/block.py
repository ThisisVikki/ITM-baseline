import torch.nn as nn

import torch
import torch.nn.functional as F



class CCALayer1(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCALayer1, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )


    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y



def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)


def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)


def mean_channels(inp):
    torch._assert(inp.dim() == 4, 'dim4') # assert(inp.dim() == 4)
    # assert not torch.any(torch.isnan(inp)) 因为可视化把这行去掉了
    spatial_sum = inp.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (inp.size(2) * inp.size(3))

def stdv_channels(inp):
    #assert(inp.dim() == 4)
    #assert not torch.any(torch.isnan(inp))
    F_mean = mean_channels(inp)
    #print(F_mean)
    #print(inp)
    #assert not torch.any(torch.isnan(inp-F_mean))
    #assert not torch.any(torch.isnan((inp-F_mean).pow(2)))
    F_variance =  (inp - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (inp.size(2) * inp.size(3))
    #print("F_variance:",F_variance)
    #print(torch.count_nonzero(F_variance))
    #print(F_variance.size())
    #return F_variance.pow(0.5)
    return F_variance.pow(0.5)


class CCALayer(nn.Module):
    def __init__(self, channel):
        super(CCALayer, self).__init__()

        self.conv3_1 = conv_layer(channel, 16, 3)
        self.act = activation('lrelu', neg_slope=0.05)


    def forward(self, x):
        y = self.act(self.conv3_1(x))

        return y

class CCALayer_ksize(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size):
        super(CCALayer_ksize, self).__init__()

        self.conv3_1 = conv_layer(channel_in, channel_out, kernel_size)
        self.act = activation('lrelu', neg_slope=0.05)


    def forward(self, x):
        y = self.act(self.conv3_1(x))
        #y = self.conv3_1(x)
        return y


def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


class CCALayer_ksizeReLU(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size):
        super(CCALayer_ksizeReLU, self).__init__()

        self.conv3_1 = conv_layer(channel_in, channel_out, kernel_size)
        self.act = activation('relu')


    def forward(self, x):
        y = self.act(self.conv3_1(x))
        #y = self.conv3_1(x)
        return y

def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer



def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

class ESA(nn.Module):
    """
    attention
    """
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False) 
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3+cf)
        m = self.sigmoid(c4)
        
        return x * m

# reduce
class IRB_minus(nn.Module):
    def __init__(self, in_channels):
        super(IRB_minus, self).__init__()
        self.rc = self.remaining_channels = in_channels
        self.dc =int(in_channels // 2)
        self.lastc = int(self.rc - self.dc)
        self.distill_1 = CCALayer_ksize(self.rc,self.dc, 3)
        self.mid_1 = conv_layer(self.dc, self.rc,3)
        #self.distills = []
        #self.mids = []
        self.distill_2 = conv_layer(self.rc, self.lastc, 1)
        #setattr(self, 'distill_{}'.format(str(fdrbs)), distill)
        self.conv = conv_layer (in_channels, in_channels, 1)
        #self.c4_d = conv_layer(self.remaining_channels, 16, 1)
        #self.c5 = conv_layer(in_channels, in_channels, 1)
        self.act = activation('lrelu', neg_slope=0.05)
        self.esa = CCALayer1(in_channels)
        #self.distil_1 = CCALayer(in_channels)
        #self.distil_2 = CCALayer(in_channels)
        #self.distil_3 = CCALayer(in_channels)
        #self.mid1 = conv_layer(16, 64, 3)
        #self.mid2 = conv_layer(16, 64, 3)
        #self.mid3 = conv_layer(16, 64, 3)
    def forward(self, input):
        F_in = input
        dist1 = self.distill_1(input)  # F1
        mid1 = self.mid_1(dist1)  # conv3
        rem1 = input - mid1
        
        # dist2 = self.act(self.distill_2(rem1))
        dist2 = self.distill_2(rem1)
        out = torch.cat([dist1, dist2], dim = 1)  # concate

        out_fused = self.esa(self.conv(out))  # conv1+CCAlayer
        output = F_in + out_fused
        return output


# extract F1 after conv  
class IRB_F1(nn.Module):
    def __init__(self, in_channels):
        super(IRB_F1, self).__init__()
        self.rc = self.remaining_channels = in_channels
        self.dc =int(in_channels // 2)
        self.lastc = int(self.rc - self.dc)
        self.distill_1 = CCALayer_ksize(self.rc,self.dc, 3)
        self.mid_1 = conv_layer(self.dc, self.rc,3)
        #self.distills = []
        #self.mids = []
        self.distill_2 = conv_layer(self.rc, self.lastc, 1)
        #setattr(self, 'distill_{}'.format(str(fdrbs)), distill)
        self.conv = conv_layer (in_channels+self.dc, in_channels, 1)
        #self.c4_d = conv_layer(self.remaining_channels, 16, 1)
        #self.c5 = conv_layer(in_channels, in_channels, 1)
        self.act = activation('lrelu', neg_slope=0.05)
        self.esa = CCALayer1(in_channels)
        #self.distil_1 = CCALayer(in_channels)
        #self.distil_2 = CCALayer(in_channels)
        #self.distil_3 = CCALayer(in_channels)
        #self.mid1 = conv_layer(16, 64, 3)
        #self.mid2 = conv_layer(16, 64, 3)
        #self.mid3 = conv_layer(16, 64, 3)
    def forward(self, input):
        F_in = input
        dist1 = self.distill_1(input)  # conv3+leaky relu
        mid1 = self.mid_1(dist1)
        rem1 = input + mid1  # add
        
        # dist2 = self.act(self.distill_2(rem1))
        dist2 = self.distill_2(rem1)  # conv1
        out = torch.cat([mid1, dist2], dim = 1)

        out_fused = self.esa(self.conv(out))
        output = F_in + out_fused
        return output


# w/o F1
class IRB_noF1(nn.Module):
    def __init__(self, in_channels):
        super(IRB_noF1, self).__init__()
        self.rc = self.remaining_channels = in_channels
        self.dc =int(in_channels // 2)
        self.lastc = int(self.rc - self.dc)
        self.distill_1 = CCALayer_ksize(self.rc,self.dc, 3)
        self.mid_1 = conv_layer(self.dc, self.rc,3)
        #self.distills = []
        #self.mids = []
        self.distill_2 = conv_layer(self.rc, self.lastc, 1)
        #setattr(self, 'distill_{}'.format(str(fdrbs)), distill)
        self.conv = conv_layer (self.rc-self.dc, in_channels, 1)
        #self.c4_d = conv_layer(self.remaining_channels, 16, 1)
        #self.c5 = conv_layer(in_channels, in_channels, 1)
        self.act = activation('lrelu', neg_slope=0.05)
        self.esa = CCALayer1(in_channels)
        #self.distil_1 = CCALayer(in_channels)
        #self.distil_2 = CCALayer(in_channels)
        #self.distil_3 = CCALayer(in_channels)
        #self.mid1 = conv_layer(16, 64, 3)
        #self.mid2 = conv_layer(16, 64, 3)
        #self.mid3 = conv_layer(16, 64, 3)
    def forward(self, input):
        F_in = input
        dist1 = self.distill_1(input)  # conv3+leaky relu
        mid1 = self.mid_1(dist1)  # conv3
        rem1 = input + mid1  # add
        
        # dist2 = self.act(self.distill_2(rem1))
        dist2 = self.distill_2(rem1)  # conv1
        # out = torch.cat([mid1, dist2], dim = 1)

        out_fused = self.esa(self.conv(dist2))
        output = F_in + out_fused
        return output


# w/o conv1
class IRB_noConv1(nn.Module):
    def __init__(self, in_channels):
        super(IRB_noConv1, self).__init__()
        self.rc = self.remaining_channels = in_channels
        self.dc =int(in_channels // 2)
        self.lastc = int(self.rc - self.dc)
        self.distill_1 = CCALayer_ksize(self.rc,self.dc, 3)
        self.mid_1 = conv_layer(self.dc, self.rc,3)
        #self.distills = []
        #self.mids = []
        self.distill_2 = conv_layer(self.rc, self.lastc, 1)
        #setattr(self, 'distill_{}'.format(str(fdrbs)), distill)
        self.conv = conv_layer (in_channels + self.dc, in_channels, 1)
        #self.c4_d = conv_layer(self.remaining_channels, 16, 1)
        #self.c5 = conv_layer(in_channels, in_channels, 1)
        self.act = activation('lrelu', neg_slope=0.05)
        self.esa = CCALayer1(in_channels)
        #self.distil_1 = CCALayer(in_channels)
        #self.distil_2 = CCALayer(in_channels)
        #self.distil_3 = CCALayer(in_channels)
        #self.mid1 = conv_layer(16, 64, 3)
        #self.mid2 = conv_layer(16, 64, 3)
        #self.mid3 = conv_layer(16, 64, 3)
    def forward(self, input):
        F_in = input
        dist1 = self.distill_1(input)
        mid1 = self.mid_1(dist1)
        rem1 = input + mid1
        
        # dist2 = self.act(self.distill_2(rem1)) # 去掉激活
        # dist2 = self.distill_2(rem1) # 去掉Conv1
        out = torch.cat([dist1, rem1], dim = 1)

        out_fused = self.esa(self.conv(out))
        output = F_in + out_fused
        return output


# change F1 to F0(Fin)
class IRB_F0(nn.Module):
    def __init__(self, in_channels):
        super(IRB_F0, self).__init__()
        self.rc = self.remaining_channels = in_channels
        self.dc =int(in_channels // 2)
        self.lastc = int(self.rc - self.dc)
        self.distill_1 = CCALayer_ksize(self.rc,self.dc, 3)
        self.mid_1 = conv_layer(self.dc, self.rc,3)
        #self.distills = []
        #self.mids = []
        self.distill_2 = conv_layer(self.rc, self.lastc, 1)
        #setattr(self, 'distill_{}'.format(str(fdrbs)), distill)
        self.conv = conv_layer (in_channels+self.dc, in_channels, 1)
        #self.c4_d = conv_layer(self.remaining_channels, 16, 1)
        #self.c5 = conv_layer(in_channels, in_channels, 1)
        self.act = activation('lrelu', neg_slope=0.05)
        self.esa = CCALayer1(in_channels)
        #self.distil_1 = CCALayer(in_channels)
        #self.distil_2 = CCALayer(in_channels)
        #self.distil_3 = CCALayer(in_channels)
        #self.mid1 = conv_layer(16, 64, 3)
        #self.mid2 = conv_layer(16, 64, 3)
        #self.mid3 = conv_layer(16, 64, 3)
    def forward(self, input):
        F_in = input
        dist1 = self.distill_1(input)  # conv3+leaky relu
        mid1 = self.mid_1(dist1)
        rem1 = input + mid1  # add
        
        # dist2 = self.act(self.distill_2(rem1))
        dist2 = self.distill_2(rem1)  # conv1
        out = torch.cat([input, dist2], dim = 1)  # F1 to F0

        out_fused = self.esa(self.conv(out))
        output = F_in + out_fused
        return output

# no second Conv1
class IRB_no2Conv1(nn.Module):
    def __init__(self, in_channels):
        super(IRB_no2Conv1, self).__init__()
        self.rc = self.remaining_channels = in_channels
        self.dc =int(in_channels // 2)
        self.lastc = int(self.rc - self.dc)
        self.distill_1 = CCALayer_ksize(self.rc,self.dc, 3)
        self.mid_1 = conv_layer(self.dc, self.rc,3)
        #self.distills = []
        #self.mids = []
        self.distill_2 = conv_layer(self.rc, self.lastc, 1)
        #setattr(self, 'distill_{}'.format(str(fdrbs)), distill)
        self.conv = conv_layer (in_channels + self.dc, in_channels, 1)
        #self.c4_d = conv_layer(self.remaining_channels, 16, 1)
        #self.c5 = conv_layer(in_channels, in_channels, 1)
        # self.act = activation('lrelu', neg_slope=0.05)
        self.esa = CCALayer1(in_channels)  # without conv1, input channels no change
        #self.distil_1 = CCALayer(in_channels)
        #self.distil_2 = CCALayer(in_channels)
        #self.distil_3 = CCALayer(in_channels)
        #self.mid1 = conv_layer(16, 64, 3)
        #self.mid2 = conv_layer(16, 64, 3)
        #self.mid3 = conv_layer(16, 64, 3)
    def forward(self, input):
        F_in = input
        dist1 = self.distill_1(input)
        mid1 = self.mid_1(dist1)
        rem1 = input + mid1
        
        # dist2 = self.act(self.distill_2(rem1)) # 去掉激活
        dist2 = self.distill_2(rem1)
        out = torch.cat([dist1, dist2], dim = 1)

        # out_fused = self.esa(self.conv(out)) # 去掉第二个conv1
        out_fused = self.esa(out)
        output = F_in + out_fused
        return output


class IRB_add(nn.Module):
    def __init__(self, in_channels):
        super(IRB_add, self).__init__()
        self.rc = self.remaining_channels = in_channels
        self.dc =int(in_channels // 2)
        self.lastc = int(self.rc - self.dc)
        self.distill_1 = CCALayer_ksize(self.rc,self.dc, 3)
        self.mid_1 = conv_layer(self.dc, self.rc,3)
        #self.distills = []
        #self.mids = []
        self.distill_2 = conv_layer(self.rc, self.lastc, 1)
        #setattr(self, 'distill_{}'.format(str(fdrbs)), distill)
        self.conv = conv_layer (in_channels, in_channels, 1)
        #self.c4_d = conv_layer(self.remaining_channels, 16, 1)
        #self.c5 = conv_layer(in_channels, in_channels, 1)
        self.act = activation('lrelu', neg_slope=0.05)
        self.esa = CCALayer1(in_channels)
        #self.distil_1 = CCALayer(in_channels)
        #self.distil_2 = CCALayer(in_channels)
        #self.distil_3 = CCALayer(in_channels)
        #self.mid1 = conv_layer(16, 64, 3)
        #self.mid2 = conv_layer(16, 64, 3)
        #self.mid3 = conv_layer(16, 64, 3)
    def forward(self, input):
        F_in = input
        dist1 = self.distill_1(input)
        mid1 = self.mid_1(dist1)
        rem1 = input + mid1 # add
        
        # dist2 = self.act(self.distill_2(rem1))
        dist2 = self.distill_2(rem1)
        out = torch.cat([dist1, dist2], dim = 1)

        out_fused = self.esa(self.conv(out))
        output = F_in + out_fused
        return output
