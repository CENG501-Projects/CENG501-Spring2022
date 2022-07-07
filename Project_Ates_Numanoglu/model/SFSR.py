import torch
import torch.nn as nn
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.module import Module
from torch.autograd import Variable
import math
from .sepformer import Sepformer
from .SuperResolution import SuperResolution

class SFSR(nn.Module):
    def __init__(self, N=64, C=2, L=4, H=4, K=250, Global_B=2, Local_B=4):
        super(SFSR,self).__init__()

        self.N = N
        self.C = C
        self.L = L
        self.H = H
        self.K = K
        self.Global_B = Global_B
        self.Local_B = Local_B

        self.sep_former = Sepformer(N=self.N, C=self.C, L=self.L, H=self.H, K=self.K, Global_B=self.Global_B, Local_B=self.Local_B)
        self.super_res = SuperResolution()

    def forward(self,mixture):
        sep_out_list = list()
        # Get k outputs from sepformer blocks
        for k, output in enumerate(self.sep_former(mixture)):
            sep_out_list.append(output)
        
        super_res_output = self.super_res(mixture, sep_out_list[-1])

        return sep_out_list, super_res_output

    @classmethod
    def load_model(cls, path):

        package = torch.load(path, map_location=lambda storage, loc: storage)

        model = cls.load_model_from_package(package)

        return model

    @classmethod
    def load_model_from_package(cls, package):

        model = cls(N=package['N'], C=package['C'], L=package['L'],
                    H=package['H'], K=package['K'], Global_B=package['Global_B'],
                    Local_B=package['Local_B'])

        model.load_state_dict(package['state_dict'])

        return model

    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss=None, cv_loss=None):

        package = {
            # hyper-parameter
            'N': model.N, 'C': model.C, 'L': model.L,
            'H': model.H, 'K': model.K, 'Global_B': model.Global_B,
            'Local_B': model.Local_B,

            # state
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch
        }

        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss

        return package