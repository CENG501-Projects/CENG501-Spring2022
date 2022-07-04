# PyTorch libraries:
from statistics import mode
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pytorch_wavelets import DWTForward

class ResidualDsBlock(nn.Module):
    def __init__(self):
    # in_channels must be 64 by design 
        super(ResidualDsBlock, self).__init__()
        _channels = 64
        self.conv1 = nn.Conv2d(_channels, _channels, kernel_size=3, padding="same")
        self.conv2 = nn.Conv2d(_channels, _channels, kernel_size=3, padding="same")
        self.conv3 = nn.Conv2d(_channels, _channels, kernel_size=3, padding="same")
        self.conv4 = nn.Conv2d(_channels, _channels, kernel_size=3, padding="same")
        self.conv5 = nn.Conv2d(_channels, _channels, kernel_size=3, padding="same")
        self.conv_last = nn.Conv2d(_channels*6, _channels, kernel_size=1, padding="same")
        self.relu = nn.ReLU(inplace=False)
    
    def forward(self, x):
        block1 = self.relu(self.conv1(x))
        prev_sum = torch.add(x, block1)
        block2 = self.relu(self.conv2(prev_sum))
        prev_sum = torch.add(prev_sum, block2)
        block3 = self.relu(self.conv3(prev_sum))
        prev_sum = torch.add(prev_sum, block2)
        block4 = self.relu(self.conv4(prev_sum))
        prev_sum = torch.add(prev_sum, block2)
        block5 = self.relu(self.conv5(prev_sum))
        all_blocks_concat = torch.cat([x, block1, block2, block3, block4, block5], dim=1)
        out = self.conv_last(all_blocks_concat)
        x = torch.add(x, out)
        return x

# Details about the residual blocks were not given, this implementation is just an assumption.
class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        _channels = 64
        self.conv1_0 = nn.Conv2d(_channels, _channels, kernel_size=3, padding="same")
        self.conv1_1 = nn.Conv2d(_channels, _channels, kernel_size=3, padding="same")
        self.conv2_0 = nn.Conv2d(_channels, _channels, kernel_size=3, padding="same")
        self.conv2_1 = nn.Conv2d(_channels, _channels, kernel_size=3, padding="same")
        self.relu = nn.ReLU(inplace=False)
    
    def forward(self, x):
        out = self.conv1_1(self.relu(self.conv1_0(x)))
        x = x + out
        out = self.conv2_1(self.relu(self.conv2_0(x)))
        x = x + out
        return x

class KernelEstimator(nn.Module):
    def __init__(self):
        super(KernelEstimator, self).__init__()
        _in_channels = 3
        _channels = 64
        self.layer1_1 = nn.Conv2d(_in_channels, _channels, kernel_size=(1,3), padding="same")
        self.layer1_2 = nn.Conv2d(_in_channels, _channels, kernel_size=(3,1), padding="same")
        self.layer1_3 = nn.Conv2d(_in_channels, _channels, kernel_size=(3,3), padding="same")
        # Cannot use same padding with stride
        self.layer2 = Conv2dSame(_channels*3, _channels, kernel_size=3, stride=2)
        self.res_ds_block = ResidualDsBlock()
        self.res_block = ResidualBlock()
        self.maxpool = nn.MaxPool2d(2)
        self.layer3 = nn.Conv2d(_channels, _channels, kernel_size=1, padding="same")
        self.relu = nn.ReLU(inplace=False)
        self.layer4 = nn.Conv2d(_channels, 1, kernel_size=1, padding="same")  # NOTE: Output channel size was not given?, I had to guess.
    
    def forward(self, x):
        concat = torch.cat([self.layer1_1(x), self.layer1_2(x), self.layer1_3(x)], dim=1)  # 9, 128, 128?
        # Pad the input before concat
        x = self.layer2(concat)
        x = self.res_ds_block.forward(x)
        x = self.res_block.forward(x)
        x = self.maxpool(x)
        x = self.layer3(x)
        x = self.relu(x)
        x = self.layer4(x)
        return x 


class FreqDensityComparator(nn.Module):
    def __init__(self):
        super(FreqDensityComparator, self).__init__()
        _channels=64
        self.conv1x3_1 = nn.Conv2d(3, _channels, kernel_size=(1,3), padding="same")
        self.conv1x3_2 = nn.Conv2d(3, _channels, kernel_size=(1,3), padding="same")
        self.conv1x3_3 = nn.Conv2d(3, _channels, kernel_size=(1,3), padding="same")
        self.conv1_1 = Conv2dSame(_channels*3, _channels, kernel_size=3, stride=2,)
        self.conv1_2 = Conv2dSame(_channels, _channels, kernel_size=3, stride=2)  
        self.conv1_3 = Conv2dSame(_channels, _channels, kernel_size=3, stride=2)  
        self.conv2 = nn.Conv2d(_channels, 1, kernel_size=3, padding="same")

        self.bn = nn.BatchNorm2d(_channels)
        self.relu = nn.ReLU(inplace=False)
    
    def forward(self, x, y):
        concat = torch.cat([self.conv1x3_1(x), self.conv1x3_2(x), self.conv1x3_2(x)], dim=1)  # 192, 128, 128
        x = self.relu(self.bn(self.conv1_1(concat)))  # 64, 64, 64
        x = self.relu(self.bn(self.conv1_2(x)))  # 64, 32, 32
        x = self.relu(self.bn(self.conv1_3(x)))  # 64, 16, 16
        x = self.conv2(x)  # 1, 16, 16
        
        concat = torch.cat([self.conv1x3_1(y), self.conv1x3_2(y), self.conv1x3_2(y)], dim=1)  # 9, 128, 128?
        y = self.relu(self.bn(self.conv1_1(concat)))  # 64, 128, 128
        y = self.relu(self.bn(self.conv1_2(y)))  # 64, 64, 64
        y = self.relu(self.bn(self.conv1_3(y)))  # 64, 32, 32
        y = self.conv2(y)  # 1, 16,16
        
        diff = torch.subtract(x, y)
        return diff
        
class ConvBNLeakyRelu(nn.Module):
    def __init__(self, _in_channels, _out_channels, _kernel_size, _stride, _padding):
        super(ConvBNLeakyRelu, self).__init__()
        if(_stride == 1):
            self.conv = nn.Conv2d(_in_channels, _out_channels, _kernel_size, stride=_stride, padding=_padding)
        else:
            self.conv = Conv2dSame(_in_channels, _out_channels, _kernel_size, stride=_stride)
        self.bn = nn.BatchNorm2d(_out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.02, inplace=False)
    
    def forward(self, x):
        x = self.lrelu(self.bn(self.conv(x)))
        return x
class WaveletDiscriminator(nn.Module):
    def __init__(self):
        super(WaveletDiscriminator, self).__init__()
        _channels = 64  # TODO: Check input channel size
        
        # Haar wavelet transform
        self.DWT2 = DWTForward(J=1, wave="haar", mode="reflect")
        # High-Pass filter
        self.filter = self.filter_wavelet
        _in_channels = 9 # TODO: Find the reason why

        # Discriminator from the "Least Squares GAN" paper (LS_GAN)
        self.layer1 = ConvBNLeakyRelu(_in_channels, 64, 5, 2, "same")
        self.layer2 = ConvBNLeakyRelu(64, 128, 5, 2, "same")
        self.layer3 = ConvBNLeakyRelu(128, 256, 5, 2, "same")
        self.layer4 = ConvBNLeakyRelu(256, 512, 5, 2, "same")
        self.fc = nn.Linear(8192, 1)
    
    def forward(self, x, y):
        x = self.filter(x) # 9, 64, 64
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # Reshape before feeding fc layer, otherwise it gives errors
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        y = self.filter(y)
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        y = y.view(y.size(0), -1)
        y = self.fc(y)

        diff = x - y
        return diff
    
    # Filter implementation from: https://github.com/ShuhangGu/DASR/
    def filter_wavelet(self, x, norm=True):
        LL, Hc = self.DWT2(x)
        LH, HL, HH = Hc[0][:, :, 0, :, :], Hc[0][:, :, 1, :, :], Hc[0][:, :, 2, :, :]
        if norm:
            LH, HL, HH = LH * 0.5 + 0.5, HL * 0.5 + 0.5, HH * 0.5 + 0.5
        return torch.cat((LH, HL, HH), 1)


class AdaptationGenerator(nn.Module):
    def __init__(self):
        super(AdaptationGenerator, self).__init__()
        self.kernel_estimator = KernelEstimator()

    def forward(self, x):
        kernel = self.kernel_estimator(x)

        x_downsampled = F.interpolate(x, scale_factor=1/4, mode='bicubic', antialias=True)
        
        # Convolve the downsampled patch with the kernel to obtain g_x
        g_x = x_downsampled * kernel
        return g_x

# Strided convolution with same padding. Taken from: https://github.com/pytorch/pytorch/issues/67551
class Conv2dSame(torch.nn.Conv2d):

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
