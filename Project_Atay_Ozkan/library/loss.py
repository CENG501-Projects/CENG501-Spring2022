import torch

"""
This script contains the Loss Functions that were used in training
"""

def calc_nll_loss(target, output):
    """
    Determine Negative Log Likelihood Loss
    :param target: target class
    :param output: output class probability
    :return: calculated nll loss
    """
    criterion = torch.nn.NLLLoss()
    loss = criterion(output, target)
    return loss


def calc_mse_loss(output, target):
    """
    Determine Mean Squared Error Loss
    :param output: output data of network
    :param target: target data
    :return: calculated mse loss
    """
    criterion = torch.nn.MSELoss(reduction='sum')
    loss = criterion(output, target)
    return loss
