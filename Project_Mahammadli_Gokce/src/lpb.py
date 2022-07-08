from numpy import dtype
import torch
from torchvision.transforms.functional import rotate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def rotate_mx(mx: torch.Tensor, degree: float) -> torch.Tensor:
  """
  Rotates input matrix with given degree, in counterclockwise
  
  Parameters
  ---------
  mx : torch.Tensor
    Input matrix that will be rotated
  degree: float
    Degree to rotate matrix in counterclockwise
  
  Returns
  ------
  rot_mx : torch.Tensor
    Rotated matrix
  """
  
  rot_mx = rotate(mx.unsqueeze(0), degree).squeeze(0)
  return rot_mx

# calculating local binary pattern value
def LPB(mx: torch.Tensor) -> torch.Tensor:
  """
  Calculates Local Binary Pattern value for given matrix
  
  Parameters
  ---------
  mx : torch.Tensor
    Input matrix
  
  Returns
  ------
  lpb_value : torch.Tensor
    A single value tensor containing LPB
  """
  
  mid = mx.shape[0] // 2
  threshold = mx[mid, mid]

  bin_mx = (mx >= threshold)
  power_mx = torch.Tensor([
                       [1, 2, 4],
                       [8, 0, 16],
                       [32, 64, 128]
  ]).to(device)
  
  lpb_value = (bin_mx * power_mx).sum()

  return lpb_value

# finding rotation matrix leading to minimum LPB value
def min_lpb(mx: torch.Tensor) -> torch.Tensor:
  """
  Finds optimal rotated form of given matrix that produces minimum Local Binary Pattern value
  
  Parameters
  ---------
  mx : torch.Tensor
    Input matrix
  
  Returns
  ------
  min_mx : torch.Tensor
    Rotated matrix resulting in minimum LPB value
  """
  
  num_of_surr_elements = mx.shape[0] ** 2 - 1
  rot_deg = 360 // num_of_surr_elements

  min_mx = mx.clone()
  lpb = LPB(mx)
  for degree in range(rot_deg, 360, rot_deg):
    mx = rotate_mx(mx, degree)
    cur_lpb = LPB(mx)

    if cur_lpb < lpb:
      lpb = cur_lpb
      min_mx = mx.clone()
     
  return min_mx
