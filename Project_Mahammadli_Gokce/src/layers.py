import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed
import multiprocessing
from lpb import min_lpb


class RRL(nn.Module):
    """
    A class to create Regional Rotation Layer for CNN architectures to make them rotation invariant, add this layer before each convolution layer, and after the feature extraction if task is classification
    
    Attributes
    ---------
    kernel_size : int
        Height and Width of the kernel. This is also stride, as sliding windos do not overlap
    padding : tuple[int]
        Number of row/vector of zeros to add to height/width
        
    Methods
    ------
    forward(x):
        Returns a tensor with the same dimension as input x,
        A result of concatenated rotated sliding windows
    """
    def __init__(self, kernel_size: int=3, padding: tuple[int]=(0, 0)) -> None:
        """
        Contructs all the necessary attributes for the RRL class 
        
        Parameters
        ---------
            kernel_size : int
                Height and Width of the kernel. This is also stride, as sliding windos do not overlap
            padding : tuple[int]
                Number of row/vector of zeros to add to height/width     
        """
        super(RRL, self).__init__()
        self.FH = self.FW = self.stride = kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns a tensor with the same dimension as input x,
        A result of concatenated rotated sliding windows
        
        Parameters
        ---------
        x : torch.Tensor
            input batch with dimensions: batch_size x channel_size x height x width
        
        Returns
        ------
        x : torch.Tensor
            a tensor with the same dimension as input x,
            A result of concatenated rotated sliding windows
        """
        # x -> (batch_size, channels, height, width)
        # OH, OW - feature map height and width
        # FH, FW - kernel height and width

        batch_size, ch, H, W = x.shape
        OH = H // self.FH
        OW = W // self.FW

        # pad the input
        x = F.pad(x, pad=self.padding)
        
        # create sliding windows
        x = x.unfold(2, self.FH, self.stride).unfold(3, self.FW, self.stride) # x -> (batch_size, channels, OH, OW, FH, FW)
        x = x.contiguous().view(batch_size, ch, -1, self.FH, self.FW) # x -> (batch_size, channels, OH*OW, FH, FW)
        x = x.permute(0, 2, 1, 3, 4) # x -> (batch_size, OH*OW, channels, FH, FW)

        
        # calculate minimum LPB state for all the windows 
        #x = vmap(min_lpb, in_dims=(-2, -1), out_dims=(-2, -1))(x) # x -> (batch_size, OH*OW, channels, FH, FW)
        '''
        vmap did not work, and actually it is meant to be used for operations on 2 or more matrices like matrix multiplication,
        and there is no support for custom broadcasting or vectorization just for single matrix to map over
        some dimensions in PyTorch, like in numpy. Therefore We had to use cpu, but at least to get more speed, we use
        multiprocessing. We have created discussion for this issue, currently in PyTorch, these operations cannot
        be done on GPU, other sliding window layers can take advantage of GEMM, like Convolution layer, or can use's PyTorch's default
        functions over sliding windows, like max function for MaxPooling2d. However, our functions do not solely depend 
        on elementary matrix operations, and its main part is rotation that needs to be applied all sliding windows 
        paralelly. Due to these reasons, we had no choice but run on CPU which is very very slow :/. 
        '''
        num_cores = multiprocessing.cpu_count()
        x = Parallel(n_jobs=num_cores, backend="threading")(delayed(min_lpb)(x[b, o, c, :, :]) for b in range(batch_size) for o in range(OH*OW) for c in range(ch))
        x = torch.stack(x).view(batch_size, OH*OW, ch, self.FH, self.FW) # x -> (batch_size, OH*OW, channels, FH, FW)

        # reshape matrix into original format
        x = x.permute(0, 2, 1, 3, 4) # x -> (batch_size, channels, OH*OW, FH, FW)
        x = x.view(batch_size, ch, OH, OW, self.FH, self.FW) # x -> (batch_size, channels, OH, OW, FH, FW)
        x = x.permute(0, 1, 2, 4, 3, 5) # x -> (batch_size, channels, OH, FH, OW, FW)
        x = x.contiguous().view(batch_size, ch, H, W) # x -> (batch_size, channels, OH*FH, OW*FW)
    
        return x
