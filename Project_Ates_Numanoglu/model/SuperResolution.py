"""

    This code snippet is the construction of the Super Resolution model based on 
    the methods mentioned in SFSRNet: Super-resolution for single-channel Audio Source Separation

    * Authors:   

       Süleyman Ateş - ates.suleyman@metu.edu.tr    
       Arda Numanoğlu - arda.numanoglu@metu.edu.tr  
    
    * Created:   04.07.2022

    * This code is written to reproduce the paper of SFSRNet, paper can be found at this link:
        - https://www.aaai.org/AAAI22Papers/AAAI-1535.RixenJ.pdf   

"""

import torch
import torch.nn as nn

class Heuristic(nn.Module):
    def __init__(self):
        super(Heuristic,self).__init__()


    def forward(self,mix,est):
        """
            Implementation of Heuristic to correct the 
            magnitude of the higher frequencies of each estimation.

        """    
        if(mix.shape[1]%2 != 0):
            mix_high = mix[:,int(mix.shape[1]/2):] 
            mix_low = mix[:,:int(mix.shape[1]/2)+1]
            est_high = est[:,int(est.shape[1]/2):]
            est_low = est[:,:int(est.shape[1]/2)+1]
        else:
            mix_high = mix[:,int(mix.shape[1]/2):] 
            mix_low = mix[:,:int(mix.shape[1]/2)]
            est_high = est[:,int(est.shape[1]/2):]
            est_low = est[:,:int(est.shape[1]/2)]

        mix_low_seq = torch.sum(mix_low,dim=1)
        est_low_seq = torch.sum(est_low,dim=1)

        divison = mix_low_seq/est_low_seq

        mult = mix_high*divison

        #padding
        out = torch.zeros(mix.shape)
        out[:,out.shape[1]-mult.shape[1]:,:] = mult

        return out.cuda()


class SRNetwork(nn.Module):
    def __init__(self, C=2):
        super(SRNetwork,self).__init__()

        self.C = C

        self.conv1 =  nn.Sequential(
            nn.Conv2d(5,128,kernel_size=5,padding=2),
            nn.ReLU()
        )
        self.conv2 =  nn.Sequential(
            nn.Conv2d(128,256,kernel_size=9,padding=4),
            nn.ReLU()
        )
        self.conv3 =  nn.Sequential(
            nn.Conv2d(256,128,kernel_size=11,padding=5),
            nn.ReLU()
        )
        self.conv4 =  nn.Sequential(
            nn.Conv2d(128,self.C,kernel_size=11,padding=5),
            nn.ReLU()
        )

        self.layernorm1 = nn.LayerNorm(128)
        self.layernorm2 = nn.LayerNorm(256)
        self.layernorm3 = nn.LayerNorm(128)

    def forward(self,X):
        X = self.conv1(X)
        X = X.permute(0, 2, 3, 1)
        X = self.layernorm1(X)
        X = X.permute(0, 3, 1, 2)

        X = self.conv2(X)
        X = X.permute(0, 2, 3, 1)
        X = self.layernorm2(X)
        X = X.permute(0, 3, 1, 2)

        X = self.conv3(X)
        X = X.permute(0, 2, 3, 1)
        X = self.layernorm3(X)
        X = X.permute(0, 3, 1, 2)

        X = self.conv4(X)
        return X


class SuperResolution(nn.Module):
    def __init__(self, C=2):
        super(SuperResolution,self).__init__()

        self.C = C  #number of speakers

        self.heuristic = Heuristic()
        self.network = SRNetwork(C)

    def forward(self,mixture,estimations):

        #Splitting estimations of Sepformer
        estimation1 = estimations[:,0,:].clone()
        estimation2 = estimations[:,1,:].clone()
        
        length = estimation1.shape[1]

        #Getting the STFT's
        stft_mixture = torch.stft(mixture,n_fft=256,hop_length=64,return_complex=True)
        stft_estimation1 = torch.stft(estimation1,n_fft=256,hop_length=64, return_complex=True) #frame length = 256, frame_step = 64
        stft_estimation2 = torch.stft(estimation2,n_fft=256,hop_length=64, return_complex=True)

        #Converting to magnitudes
        magnitude_mixture = torch.abs(stft_mixture)
        magnitude_estimation1 = torch.abs(stft_estimation1)
        magnitude_estimation2 = torch.abs(stft_estimation2)

        #Getting the Phase of conversion
        phase1 = torch.angle(stft_estimation1)
        phase2 = torch.angle(stft_estimation2)

        #Heuristic
        out_heruistic1 = self.heuristic(magnitude_mixture,magnitude_estimation1)
        out_heruistic2 = self.heuristic(magnitude_mixture,magnitude_estimation2)

        #Concatenating the neccessary inputs for Network
        concat = torch.stack((magnitude_mixture,magnitude_estimation1,magnitude_estimation2,out_heruistic1,out_heruistic2))
        concat = concat.permute(1,0,2,3)
        out_network = self.network(concat)

        #Splitting the output of net and adding the magnitude of old_estimation
        estimation_new_1 = magnitude_estimation1+out_network[:,0,:,:]
        estimation_new_2 = magnitude_estimation2+out_network[:,1,:,:]


        #Getting complex from magnitude and phase
        complex1 = estimation_new_1*(torch.cos(phase1)+1j*torch.sin(phase1))
        complex2 = estimation_new_2*(torch.cos(phase2)+1j*torch.sin(phase2))

        #Inverse STFT
        out1 = torch.istft(complex1,n_fft=256,hop_length=64, length=length)
        out2 = torch.istft(complex2,n_fft=256,hop_length=64, length=length)

        #Stacking the estimations
        out = torch.stack((out1,out2),dim=1)

        return out

# dummmy_mix = torch.randn((1,28000,))
# dummmy_est = torch.randn((1,2,28000))

# model = SuperResolution(5,2)


# out = model(dummmy_mix,dummmy_est)