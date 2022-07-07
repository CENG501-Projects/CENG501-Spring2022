import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class DIMPLayer(MessagePassing):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__(aggr='add')

        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.non_lin = nn.ReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_dim))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()


    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin.weight.data)


    def forward(self, x, edge_index):
        msg = self.propagate(edge_index, x=x)

        out = self.lin(msg)
        if self.bias is not None:
            out += self.bias

        out = self.non_lin(out)

        return out


    def message(self, x_i, x_j):
        elem_mult = x_i * x_j
        inner_mult = elem_mult.sum(dim = 1, keepdim = True)
        inner_mult = inner_mult.expand_as(elem_mult)
        mask = (inner_mult != 0)
        elem_mult[mask] = elem_mult[mask] / inner_mult[mask]

        return elem_mult




class Discriminator(nn.Module):
    """
    Note: During the implementation of the Discriminator module, I benefited from 
    the repo given at https://github.com/PetarV-/DGI.
    """
    def __init__(self, feat_dim):
        super(Discriminator, self).__init__()
        self.bi = nn.Bilinear(feat_dim, feat_dim, 1)

        for m in self.modules():
            self.init_weights(m)


    def init_weights(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


    def forward(self, x, s):
        s_x = s.expand_as(x)

        out = self.bi(x, s_x)
        out = torch.squeeze(out)

        return out