import torch
import torch.nn as nn
from torchmetrics import PeakSignalNoiseRatio
import numpy as np
import matplotlib.pyplot as plt

import dataset
import networks
import utils
from torchinfo import summary


def main_for_test(datasets, scale_factor=4):
    print("Scale factor for tests:", scale_factor)
    PATH = ("model_weights_x" + str(scale_factor) + ".pt").replace(" ", "")
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    device = 'cpu'

    hr_images, lr_images, mean_image = dataset.get_test_images(scale_factor, datasets)
    model = networks.VDSR_new(mean_image.to(device))
    model.load_state_dict(torch.load(PATH))
    model.eval()
    model.to(device)
    psnr = PeakSignalNoiseRatio()
    hr_images = torch.autograd.Variable(hr_images.float())
    lr_images = torch.autograd.Variable(lr_images.float())
    output_hr = model(lr_images)
    print('PSNR value for the dataset ', datasets, " :", psnr(output_hr, hr_images))


if __name__ == "__main__":
    main_for_test('Set5', 3)
    main_for_test('Set14', 3)
    main_for_test('B100', 3)
    main_for_test("Urban100", 3)
