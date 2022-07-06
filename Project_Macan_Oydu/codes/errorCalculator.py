import math
import numpy as np
from skimage.metrics import structural_similarity

def peak_signal_noise_ratio(img, gt):
    mse = mean_squared_error(img, gt)
    if mse == 0:
        return 100
    max = 255.0
    return 20*math.log10(max / math.sqrt(mse))

def structural_sim(img, gt):
    img = img.cpu().detach().numpy()
    gt = gt.cpu().detach().numpy()
    img = np.squeeze(img,axis=0)
    gt = np.squeeze(gt,axis=0)
    img = np.transpose(img, (1, 2, 0))
    gt = np.transpose(gt, (1, 2, 0))
    ssim_value = structural_similarity(img, gt, multichannel=True)
    return ssim_value

def mean_squared_error(img, gt):
    img = img.cpu().detach().numpy()
    gt = gt.cpu().detach().numpy()
    img = np.squeeze(img,axis=0)
    gt = np.squeeze(gt,axis=0)
    img = np.transpose(img, (1, 2, 0))
    gt = np.transpose(gt, (1, 2, 0))
    mse = np.mean((img - gt) ** 2)
    return mse