import os
from PIL import Image
from torch.utils.data import Dataset

class dataset(Dataset):
    def __init__(self,image_dir,mask_dir,groud_truth_dir, transformation):
        super(dataset,self).__init__()
        files = os.listdir(image_dir)
        self.I_s_dir = [os.path.join(image_dir, k) for k in files]
        files = os.listdir(image_dir)
        self.M_dir = [os.path.join(mask_dir, k) for k in files]
        files = os.listdir(image_dir)
        self.GT_dir = [os.path.join(groud_truth_dir, k) for k in files]
        self.transformation = transformation
    
    def __getitem__(self, index):
        I_s_path = self.I_s_dir[index]
        M_path = self.M_dir[index]
        gt_path = self.GT_dir[index]

        I_s = Image.open(I_s_path)
        M = Image.open(M_path)
        gt = Image.open(gt_path)
        I_s = self.transformation(I_s)
        M = self.transformation(M)
        gt = self.transformation(gt)

        return I_s,M,gt

    def __len__(self):
        return len(self.I_s_dir)