import torch
import torch.nn as nn
from blocks import *

class VDSR_new(nn.Module):
    def __init__(self, mean_image):
        super(VDSR_new, self).__init__()
        self.conv_first = nn.Conv2d(in_channels=3,
                                    out_channels=16,
                                    kernel_size=(3, 3),
                                    padding='same')
        self.relu1 = nn.ReLU()
        self.res_blocks = [ResBlock(16) for _ in range(9)]
        self.conv_last = nn.Conv2d(in_channels=16,
                                   out_channels=3,
                                   kernel_size=(3, 3),
                                   padding='same')
        self.mean_image = mean_image

    def forward(self, X):
        out = self.conv_first(X)
        out = self.relu1(out)
        for block in self.res_blocks:
            out = block(out)
        out = self.conv_last(out)
        out = out + X
        return out + self.mean_image
