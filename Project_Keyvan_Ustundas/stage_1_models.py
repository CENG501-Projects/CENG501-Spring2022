import torch
import numpy as np
import torch.nn.functional as F


def correlation(t1, t2):
    return torch.sum((t1 - t1.mean()) * (t2 - t2.mean())) / t1.shape[0]

class CorrelationBlock(torch.nn.Module):
    def __init__(self, upsample_shape):
        super(CorrelationBlock, self).__init__()
        self.up_shape = upsample_shape
 
    # Zero mean the data
    def forward(self, ok, fk):
        B, H,W, _ = fk.shape
        corr = torch.einsum( "...xyz,...hwd->hwxyz", ok, fk).reshape((B, H, W, -1))

        ak = torch.sum(corr, axis=2)

        avg_ak = torch.mean(ak)

        mask = torch.gt(ak, avg_ak)
        avg_ak_hat = mask * ak

        avg_ak_hat -= avg_ak_hat.min(1, keepdim=True)[0]
        avg_ak_hat /= avg_ak_hat.max(1, keepdim=True)[0]

        return corr, avg_ak_hat


class AttentionBlock(torch.nn.Module):
    def __init__(self):
        super(AttentionBlock, self).__init__()
        self.cr = CorrelationBlock(torch.Size([5,6,6]))
        self.conv1 = torch.nn.Conv2d(2880, 1440, (1,1))
        self.conv2 = torch.nn.Conv2d(256, 1440, (1,1))

    def forward(self, ok, fk):
        corr, ak = self.cr(ok, fk)
        corr_conv = self.conv1(corr.permute((2,0,1)))

        fk = fk.permute((2,0,1))

        fk_conv = self.conv2((fk * ak.expand_as(fk)))

        return torch.cat((corr_conv, fk_conv))        


class Stage1(torch.nn.Module):
    def __init__(self, features, extractor, resnet50):
        super(Stage1, self).__init__()
        self.o1 = torch.Tensor(features["o1"])
        self.o2 = torch.Tensor(features["o2"])
        self.o3 = torch.Tensor(features["o3"])
        self.extractor = extractor
        
        self.attention_256 = AttentionBlock()
        self.attention_512 = AttentionBlock()
        self.attention_1024 = AttentionBlock()
        self.resBlock = resnet50
        self.conv1 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding="same")
        self.conv2 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding="same")
        self.conv3 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding="same")
        self.upsample_1 = torch.nn.Upsample(scale_factor=2)
        self.upsample_2 = torch.nn.Upsample(scale_factor=2)
        self.upsample_3 = torch.nn.Upsample(scale_factor=4)

    def forward(self, image_features_256, image_features_512, image_features_1024):
        data_256 = torch.Tensor(self.zero_mean(image_features_256))
        data_512 = torch.Tensor(self.zero_mean(image_features_512))
        data_1024 = torch.Tensor(self.zero_mean(image_features_1024))
        out_256 = self.attention_256(self.o1, data_256.permute(0,3,2,1))
        out_512 = self.attention_512(self.o2, data_512.permute(0,3,2,1))
        out_1024 = self.attention_1024(self.o3, data_1024.permute(0,3,2,1))
        
        out_1024 = self.resnet(out_1024)
        out_1024 = self.conv1(out_1024)
        out_1024 = self.upsample_1(out_1024)
        
        out_merged_1 = torch.cat(out_1024, out_512)

        out_merged_1 = self.resnet(out_merged_1)
        out_merged_1 = self.conv2(out_merged_1)
        out_merged_1 = self.upsample_2(out_merged_2)
        
        out_merged_2 = torch.cat(out_merged_2, out_1024)

        out_merged_2 = self.resBlock(out_merged_2)
        out_merged_2 = self.upsample_3(out_merged_2)
        out_merged_2 = self.conv3(out_merged_2)
        
        return out_merged_2
    
    def zero_mean(self, image):
        mean = torch.mean(image)
        std = torch.std(image)
        return (image-mean)/std


class DiceLoss(torch.nn.Module):
    def __init__(self):
        self.eps = 1e-6
        
    def forward(self, y, y_bar):
        y_soft = F.softmax(y, dim=1)
        one_hots = self.one_hot(y_bar, num_classes=y.shape[1])
        intersect = torch.sum(y_soft * one_hots, (1,2,3))
        cards = torch.sum(y_soft + one_hots, (1,2,3))
        score = 2 * intersect / (cards + self.eps)
        dice_score = torch.mean(1 - score)
        
        return dice_score
    
    
    def one_hot(self, y, num_classes):
        shape = np.array(y.shape)
        shape[1] = num_classes
        shape = tuple(shape)
        res = torch.zeros(shape)
        res = res.scatter_(1, input.cpu(), 1)
        
        return result