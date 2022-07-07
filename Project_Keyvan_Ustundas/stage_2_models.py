import torch
import numpy as np
import torch.nn.functional as F


class BootstrapL2Loss(torch.nn.Module):
    def __init__(self, bootstrap_ratio = 8):
        super(BootstrapL2Loss, self).__init__()
        self.b_ratio = bootstrap_ratio
        self.mse = torch.nn.MSELoss(reduction="none")
 
    def forward(self, output:torch.Tensor, target:torch.Tensor):
        batch_size = output.shape[0]   
        out_flat = output.reshape((batch_size, -1))
        target_flat = target.reshape((batch_size, -1))

        loss = self.mse(out_flat, target_flat)
        loss_vals,_ = torch.topk(loss, k= loss.shape[1] // self.b_ratio)

        return loss_vals.mean()


class AugmentedEncoder(torch.nn.Module):
    def __init__(self, latent_size, channel_size=3, input_size=128):
        super(AugmentedEncoder, self).__init__()

        self.latent_size = latent_size

        self.conv1 = torch.nn.Conv2d(channel_size, 128, kernel_size=5, stride=2, padding=2)
        self.conv2 = torch.nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.conv3 = torch.nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2)
        self.conv4 = torch.nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2)

        self.fc = torch.nn.Linear(2 * input_size * input_size, self.latent_size)

    def forward(self, x):
        batch_size = x.shape[0]
        t1 = F.relu(self.conv1(x))
        t2 = F.relu(self.conv2(t1))
        t3 = F.relu(self.conv3(t2))
        
        return self.fc(F.relu(self.conv4(t3)).reshape(batch_size, -1))


class AugmentedDecoder(torch.nn.Module):
    def __init__(self, latent_size, channel_size=3, input_size=128):
        super(AugmentedDecoder, self).__init__()

        self.latent_size = latent_size

        self.conv1 = torch.nn.Conv2d(512, 256, kernel_size=5, stride=1, padding="same")
        self.conv2 = torch.nn.Conv2d(256, 256, kernel_size=5, stride=1, padding="same")
        self.conv3 = torch.nn.Conv2d(256, 128, kernel_size=5, stride=1, padding="same")
        self.conv4 = torch.nn.Conv2d(128, channel_size, kernel_size=5, stride=1, padding="same")

        self.fc = torch.nn.Linear(self.latent_size, 2 * input_size * input_size)
        self.input_size = input_size

    def forward(self, x):
        batch_size = x.shape[0]
        t1 = self.fc(x).reshape((-1, 512, self.input_size // 16, self.input_size // 16))
        
        t2 = F.relu(self.conv1(F.upsample_bilinear(t1, scale_factor=2)))
        t3 = F.relu(self.conv2(F.upsample_bilinear(t2, scale_factor=2)))
        t4 = F.relu(self.conv3(F.upsample_bilinear(t3, scale_factor=2)))


        t5 = self.conv4(F.upsample_bilinear(t4, scale_factor=2))
        
        
        return F.sigmoid(t5)


class AugmentedAutoEncoder(torch.nn.Module):
    def __init__(self, latent_size, channel_size=3, input_size=128):
        super(AugmentedAutoEncoder, self).__init__()

        self.encoder = AugmentedEncoder(latent_size, channel_size, input_size)
        self.decoder = AugmentedDecoder(latent_size, channel_size, input_size)

        

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
