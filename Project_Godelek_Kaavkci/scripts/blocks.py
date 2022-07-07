import torch.nn as nn
import torch
import torch.nn.functional as F

class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, tensor):
        return tensor.sign().add(0.5).sign()


class BinaryConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding):
        super(BinaryConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, dilation=dilation, padding=padding)

    def forward(self, X):
        # Binarization of the weights
        self.conv.weight.data = self.conv.weight.sign().add(0.5).sign()
        out = self.conv(X)
        return out


class MrbConv2d(nn.Module):
    def __init__(self, in_channels):
        super(MrbConv2d, self).__init__()

        self.conv1 = BinaryConv2d(in_channels,
                                  int(in_channels/4),
                                  kernel_size=(1, 5),
                                  dilation=1,
                                  padding='same')
        self.conv2 = BinaryConv2d(in_channels,
                                  int(in_channels/4),
                                  kernel_size=(1, 5),
                                  dilation=2,
                                  padding='same')
        self.conv3 = BinaryConv2d(in_channels,
                                  int(in_channels/4),
                                  kernel_size=(5, 1),
                                  dilation=1,
                                  padding='same')
        self.conv4 = BinaryConv2d(in_channels,
                                  int(in_channels/4),
                                  kernel_size=(5, 1),
                                  dilation=2,
                                  padding='same')

    def forward(self, X):
        out1 = self.conv1(X)
        out2 = self.conv2(X)
        out3 = self.conv3(X)
        out4 = self.conv4(X)
        return torch.cat([out1, out2, out3, out4], 1)


class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.binarize1 = BinaryActivation()
        self.mrb = MrbConv2d(in_channels)
        self.prelu1 = nn.PReLU()

        self.binarize2 = BinaryActivation()
        self.conv = BinaryConv2d(in_channels, in_channels, (3, 3), 1, 'same')
        self.prelu2 = nn.PReLU()

    def forward(self, X):
        out = self.binarize1(X)
        out = self.mrb(out)
        out = self.prelu1(out)

        out = self.binarize2(out)
        out = self.conv(out)
        out = self.prelu2(out)

        out = out + X
        return out
