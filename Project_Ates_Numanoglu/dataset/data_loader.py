"""
    The code is taken from public GitHub repository of Zhongyang-debug. GitHup repository of Zhongyang-debug including reproduction of SepFormer model, can be found at:
    https://github.com/Zhongyang-debug/Attention-Is-All-You-Need-In-Speech-Separation
"""

from torch.utils.data import Dataset, DataLoader
import os
import librosa
import numpy as np


class MyDataset(Dataset):

    def __init__(self, data_dir, sr):

        super(MyDataset, self).__init__()

        self.data_dir = data_dir
        self.sr = sr

        file = ["mix", "s1", "s2"]

        self.mix_dir = os.path.join(data_dir, file[0])
        self.mix_list = os.listdir(os.path.abspath(self.mix_dir))

        self.s1_dir = os.path.join(data_dir, file[1])
        self.s1_list = os.listdir(os.path.abspath(self.s1_dir))

        self.s2_dir = os.path.join(data_dir, file[2])
        self.s2_list = os.listdir(os.path.abspath(self.s2_dir))

    def __getitem__(self, item):

        mix_path = os.path.join(self.mix_dir, self.mix_list[item])
        mix_data = librosa.load(path=mix_path,
                                sr=self.sr,
                                mono=True,
                                offset=0, 
                                duration=None,
                                dtype=np.float32,
                                res_type="kaiser_best",
                                )[0]
        length = len(mix_data)

        s1_path = os.path.join(self.s1_dir, self.s1_list[item])
        s1_data = librosa.load(path=s1_path,
                               sr=self.sr,
                               mono=True,
                               offset=0, 
                               duration=None,
                               )[0]

        s2_path = os.path.join(self.s2_dir, self.s2_list[item])
        s2_data = librosa.load(path=s2_path,
                               sr=self.sr,
                               mono=True,
                               offset=0, 
                               duration=None,
                               )[0]

        s_data = np.stack((s1_data, s2_data), axis=0)

        return mix_data, length, s_data

    def __len__(self):

        return len(self.mix_list)
