import torch
import torchvision.transforms as T
from torchvision.io import read_image
import os
from PIL import Image
import random
class FreqConsistentSRDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, scale, ext=".png"):
        self.dataset_path = dataset_path
        self.scale = scale
        self.ext = ext
        self.transform = T.Compose([T.ToTensor()])
        
    def __len__(self):
        length = len(os.listdir(self.dataset_path))
        return len(os.listdir(self.dataset_path))

    def __getitem__(self, index):
        idx_path =  self.dataset_path + "/" + os.listdir(self.dataset_path)[index]
        patch_list512 = os.listdir(idx_path + f"/patches512")
        patch_list128 = os.listdir(idx_path + f"/patches128")
        down_list = os.listdir(idx_path + f"/down_{self.scale}")
        up_list = os.listdir(idx_path + f"/up_{self.scale}")
        
        length_p512 = len(patch_list512)
        length_p128 = len(patch_list128)
        length_d = len(down_list)
        length_u = len(up_list)
        
        # Get random samples from upsampled/downsampled/same-sized patches
        if length_p512 > 1:
            idx_p512 = random.randint(0, length_p512-1)
        else:
            idx_p512 = 0
        if length_p128 > 1:
            idx_p128 = random.randint(0, length_p128-1)
        else:
            idx_p128 = 0
        if length_d > 1:
            idx_d = random.randint(0, length_d-1)
        else:
            idx_d = 0
        if length_u > 1:    
            idx_u = random.randint(0, length_u-1)
        else:
            idx_u = 0
        x = Image.open(idx_path + "/patches512/" + str(idx_p512) + self.ext)
        x_random = Image.open(idx_path + "/patches128/" + str(idx_p128) + self.ext)
        x_random_down = Image.open(idx_path + f"/down_{self.scale}/" + str(idx_d) + self.ext)
        x_random_up = Image.open(idx_path + f"/up_{self.scale}/" + str(idx_u) + self.ext)

        return self.transform(x), self.transform(x_random), self.transform(x_random_down), self.transform(x_random_up)

    def set_scale(self, scale):
        self.scale = scale    