from enum import Enum

"""
This script contains the Enum classes that were used in other scripts and main.ipynb
Enums are used to represent the data that represent a finite set of state
"""


class RegularizerType(Enum):
    """
        Regularizers that were used for comparison
    """
    L1 = 1
    L2 = 2
    CorrReg = 3
    MaxNorm = 4
    Dropout = 5


class NetworkType(Enum):
    """
        Networks that were used for comparison
    """
    MLP = 1
    LSTM = 2
    AE = 3
    LeNet = 4
    VGG = 5


class MLPType(Enum):
    """
        The datasets that were used in MultiLayer Perceptron Network
    """
    Polarity = 1
    MNIST = 2


class MetricType(Enum):
    """
        Metrics that were used for comparison
    """
    AUC = 1
    Error_Rate = 2
    Recon_Err = 3


class LossType(Enum):
    """
        Loss Functions that were used in training
    """
    NLLoss = 1
    MSELoss = 2
