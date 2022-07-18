import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import RRL
import random
import numpy as np

RANDOM_SEED = 42

class LeNet5(nn.Module):
    """
    LeNet5 Architecture
    
    Attributes
    ---------
    n_classes : int
        Number of classes in the dataset
    rrl : bool
        Activation for RRL layer, 
        If set True, RRL layer will be added before each convolution layer and before fully connected layers
    
    Methods
    ------
    forward(x):
        Calculates feed forward pass for the LeNet5 model
    """
    
    def __init__(self,
                 n_classes: int=10,
                 rrl: bool=False) -> None:
        """
        Constructs all the necessary attributes for the LeNet5 class 
        
        Parameters
        ---------
            n_classes : int
                Number of classes in the dataset
            rrl : bool
                Activation for RRL layer, 
                If set True, RRL layer will be added before each convolution layer and before fully connected layers
        """
        
        torch.manual_seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        
        super().__init__()
        self.n_classes = n_classes
        self.rrl = rrl
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=(1, 1))
        self.pooling = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.tanh = nn.Tanh()
        if self.rrl:
            self.RRL = RRL()

        self.fc1 = nn.Linear(in_features=16*6*6, out_features=84)
        self.fc2 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates feed forward pass for the LeNet5 model
        
        Parameters
        ---------
        x : torch.Tensor
            Input matrix with dimensions: batch_size x channel_size x height x width
            
        Returns
        ------
        out : torch.Tensor
            Probability distribution of the results
        """
        try:
            x = self.RRL(x)
            x = self.tanh(self.conv1(x))
            x = self.pooling(x)

            x = self.RRL(x)
            x = self.tanh(self.conv2(x))
            x = self.pooling(x)

            x = self.RRL(x)
            x = torch.flatten(x, 1)
            x = self.tanh(self.fc1(x))

            x = self.fc2(x)
            
            out =  F.softmax(x, dim=1)
            return out
        
        except:
            x = self.tanh(self.conv1(x))
            x = self.pooling(x)

            x = self.tanh(self.conv2(x))
            x = self.pooling(x)

            x = torch.flatten(x, 1)
            x = self.tanh(self.fc1(x))

            x = self.fc2(x)
            
            out =  F.softmax(x, dim=1)
            return out
        

# 3x3 convolution
def conv3x3(in_channels: int,
            out_channels: int,
            stride: int=1,
            padding: int=1) -> torch.Tensor:
    """
    2D convolution layer with kernel size 3x3
    
    Parameters
    ---------
    in_channels : int
        Number of channels in the input matrix
    out_chanels : int
        Number of channels that will be in the output matrix, i.e number of kernels
    stride : int
        Stride value - number of skips during the sliding window
    padding : int:
        padding size, to add zeros to both height and width
        
    Returns
    ------
    conv2d : torch.Tensor
        Result of the convolution layer
    """
    
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
          
    conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=padding, bias=False)
    return conv2d
    
# Residual block
class ResidualBlock(nn.Module):
    """
    Residual Block for ResNet model
    
    Attributes
    ---------
    in_channels : int
        Number of channels in the input matrix
    out_chanels : int
        Number of channels that will be in the output matrix, i.e number of kernels
    stride : int
        Stride value - number of skips during the sliding window
    downsample : bool
        Downsampling, if set True, additional 2D convolution layer with kernel size 3x3, followed by batch normalization will be added
    
    Methods
    ------
    forward(x):
        Returns the result of feed forward path
    """
    
    def __init__(self, in_channels: int,
                 out_channels: int,
                 stride: int=1,
                 downsample: torch.Tensor=None,
                 padding: int=1,
                 RRL: bool=False) -> None:
        """
        Constructs all the necessary attributes for the ResidualBlock class
        
        Parameters
        ---------
        in_channels : int
            Number of channels in the input matrix
        out_chanels : int
            Number of channels that will be in the output matrix, i.e number of kernels
        stride : int
            Stride value - number of skips during the sliding window
        downsample : torch.Tensor
            Downsampling, if not None, additional 2D convolution layer with kernel size 3x3, followed by batch normalization will be added
        """
        torch.manual_seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        if RRL:
            self.RRL = RRL()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the result of feed forward path
        
        Parameters
        ---------
        x : torch.Tensor
            The input matrix with dimensions: batch_size x channel_size x height x width
            
        Returns 
        ------
        out : torch.Tensor
            The output of the residual block
        """
        
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    
# ResNet
class ResNet18(nn.Module):
    """
    ResNet18 Architecture
    
    Attributes
    ---------
    block : ResidualBlock
        ResidualBlock class to add blocks in ResNet
    layers: list
        List of integers indicating the number of blocks
    num_classes : int
        Number of classes in the dataset
    downsample : torch.Tensor
        Downsampling, if not None, additional 2D convolution layer with kernel size 3x3, followed by batch normalization will be added
    rrl : bool
        Activation for RRL layer, 
        If set True, RRL layer will be added before each convolution layer and before fully connected layers  
    
    Methods
    ------
    make_layer(block: ResidualBlock, 
                   out_channels: int, 
                   blocks: int, 
                   stride: int=1,
                   padding: int=1):
                   
                   Creates n number of Residual Blocks, where n=blocks
    forward(x):
        Returns the result of the feed forward pass
    """
    
    def __init__(self, block: ResidualBlock, 
                 layers: list[int]=[2, 2, 2], 
                 num_classes: int=10, 
                 downsample: torch.Tensor=None, 
                 rrl: bool=False) -> None:
        """
        Constructs all the necessary attributes for the ResNet class
        
        Parameters
        ---------
        block : ResidualBlock
            ResidualBlock class to add blocks in ResNet
        layers: list
            List of integers indicating the number of blocks
        num_classes : int
            Number of classes in the dataset
        downsample : torch.Tensor
            Downsampling, if not None, additional 2D convolution layer with kernel size 3x3, followed by batch normalization will be added
        rrl : bool
            Activation for RRL layer, 
            If set True, RRL layer will be added before each convolution layer and before fully connected layers  
        """
        
        torch.manual_seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        
        super(ResNet18, self).__init__()
        self.downsample = downsample
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2, padding=2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)
        if rrl:
            self.RRL = RRL()

    def make_layer(self, block: ResidualBlock, 
                   out_channels: int, 
                   blocks: int, 
                   stride: int=1,
                   padding: int=1) -> nn.Sequential:
        """
        Creates n number of Residual Blocks, where n=blocks
        
        Parameters
        ---------
        block : ResidualBlock
            ResidualBlock class to add blocks in ResNet
        out_chanels : int
            Number of channels that will be in the output matrix, i.e number of kernels
        blocks : int
            Number of Residual Blocks
        stride : int
            Stride value - number of skips during the sliding window
        padding : int:
            padding size, to add zeros to both height and width
            
        Returns
        ------
        out : nn.Sequential
            Sequential layer containing Residual Blocks
        """
        
        if (stride != 1) or (self.in_channels != out_channels):
            self.downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride, padding=padding),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, self.downsample, padding=padding))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        out = nn.Sequential(*layers)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the result of the feed forward pass
        
        Parameters
        ---------
        x : torch.Tensor
            Input matrix with dimensions: batch_size x channel_size x height x width
            
        Returns
        ------
        out : torch.Tensor
            Probability distribution of the results
        """
        try:
            x = self.RRL(x)
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)

            x = self.RRL(x)
            x = self.layer1(x)
            
            x = self.RRL(x)
            x = self.layer2(x)
            
            x = self.RRL(x)
            x = self.layer3(x)

            x = self.RRL(x)
            x = self.avg_pool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            
            out = F.softmax(x, dim=1)
            return out
            
        except:
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

            x = self.avg_pool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            
            out = F.softmax(x, dim=1)
            return out