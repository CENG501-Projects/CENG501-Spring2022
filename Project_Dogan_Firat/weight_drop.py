import pdb
import torch
import torch.nn as nn
from models import WeightDrop


class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(12, 1)
        self.drop = nn.Dropout(0.4)

    def forward(self, x):
        x = self.fc1(x)
        x = self.drop(x)
        return x


linear = Linear()
pdb.set_trace()
bayesian = WeightDrop(torch.nn.Linear(12, 1), ['weight'], dropout=0.4)


x = torch.randn(10, 12)
