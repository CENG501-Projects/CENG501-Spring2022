import torch
import torch.nn as nn


class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        eps = smoothing / num_classes
        self.negative = eps
        self.positive = (1 - smoothing) + eps

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=1)
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.negative)
        true_dist.scatter_(1, target.data.unsqueeze(1), self.positive)
        return torch.sum(-true_dist * pred, dim=1).mean()


def Softmax(scores):
    scores = nn.functional.softmax(scores, dim=1)
    return scores - scores.min() + 1e-8


def Uncertainty(gcn_output):
    p = Softmax(gcn_output)
    return - torch.sum(p * p.log(), 1)


def KL_Divergence(input, target):
    return torch.sum(target * (target.exp() - input), 1)


def Wasserstein(gcn_output, fc_output):
    minus_alpha = torch.log(1 + Uncertainty(gcn_output))
    # print(minus_alpha.min(), minus_alpha.max())
    beta = torch.softmax(minus_alpha - minus_alpha.min() + 1e-8, dim=0)
    return torch.sum(beta * KL_Divergence(gcn_output, fc_output)).abs()


gcn_output = torch.randn(120, 6)
fc_output = torch.randn(120, 6)
