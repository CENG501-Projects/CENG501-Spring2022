import os
import torchvision.transforms as T
import torch
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def get_images(scale_factor, crop_window=48, crop_number=5):
    ''' Prepare and return image list
    Args:
        scale_factor(int): scale factor of interpolation
        crop_window(int): window size of random crops
        crop_number(int): how many cropped samples will be generated from
                          each image
    Returns:
        image_array_hr: HR images Tensor with shape (N, 3, crop_window, crop_window)
        image_array_lr: LR images Tensor with shape (N, 3, crop_window, crop_window)
    '''
    # Read HR and LR training images from file
    hr_path_train = "../data/DIV2K/DIV2K_train_HR"
    lr_path_train = "../data/DIV2K/DIV2K_train_LR_bicubic/X"+str(scale_factor)

    dirlist_train_hr = os.listdir(hr_path_train)
    dirlist_train_hr.sort()
    # Get only first 100 images due to memory contraint
    dirlist_train_hr = dirlist_train_hr[:300]
    image_array_hr = [cv.imread(hr_path_train+"/"+image_path)
                      for image_path in dirlist_train_hr]
    image_array_hr = [torch.from_numpy(img).permute(2, 0, 1)
                      for img in image_array_hr]
    dirlist_train_lr = os.listdir(lr_path_train)
    dirlist_train_lr.sort()
    dirlist_train_hr = dirlist_train_lr[:300]
    image_array_lr = [cv.imread(lr_path_train+"/"+image_path)
                      for image_path in dirlist_train_lr]
    for idx, image in enumerate(image_array_lr):
        image_array_lr[idx] = cv.resize(src=image,
                                        dsize=(image.shape[1]*scale_factor,
                                               image.shape[0]*scale_factor),
                                        interpolation=cv.INTER_CUBIC)
    image_array_lr = [torch.from_numpy(img).permute(2, 0, 1)
                      for img in image_array_lr]

    image_array_hr, image_array_lr, mean_image = random_crops(image_array_hr,
                                                              image_array_lr,
                                                              crop_window,
                                                              crop_number)
    return image_array_hr, image_array_lr, mean_image


def get_test_images(scale_factor, dataset='Set14', crop_window=48, crop_number=5):
    ''' Prepare and return test image list
    Args:
        scale_factor(int): scale factor of interpolation
        crop_window(int): window size of random crops
        crop_number(int): how many cropped samples will be generated from
                          each image
    Returns:
        image_array_hr: HR images Tensor with shape (N, 3, crop_window, crop_window)
        image_array_lr: LR images Tensor with shape (N, 3, crop_window, crop_window)
    '''
    # Read HR and LR training images from file
    hr_path_train = "data/test/benchmark/" + dataset + "/HR".replace(' ', '')
    lr_path_train = "data/test/benchmark/" + dataset + " /LR_bicubic/X"+str(scale_factor)
    lr_path_train = lr_path_train.replace(" ", '')
    dirlist_train_hr = os.listdir(hr_path_train)
    dirlist_train_hr.sort()
    # Get only first 100 images due to memory contraint

    image_array_hr = [cv.imread(hr_path_train+"/"+image_path)
                      for image_path in dirlist_train_hr]
    image_array_hr = [torch.from_numpy(img).permute(2, 0, 1)
                      for img in image_array_hr]
    dirlist_train_lr = os.listdir(lr_path_train)
    dirlist_train_lr.sort()

    image_array_lr = [cv.imread(lr_path_train+"/"+image_path)
                      for image_path in dirlist_train_lr]
    for idx, image in enumerate(image_array_lr):
        image_array_lr[idx] = cv.resize(src=image,
                                        dsize=(image.shape[1]*scale_factor,
                                               image.shape[0]*scale_factor),
                                        interpolation=cv.INTER_CUBIC)
    image_array_lr = [torch.from_numpy(img).permute(2, 0, 1)
                      for img in image_array_lr]

    image_array_hr, image_array_lr, mean_image = random_crops(image_array_hr,
                                                              image_array_lr,
                                                              crop_window,
                                                              crop_number)
    return image_array_hr, image_array_lr, mean_image


def random_crops(hr_imgs, lr_imgs, crop_size, crop_number):
    ''' Randomly select the specified number of square crops from each image
        with the crop size
    Args:
        hr_imgs(list[Tensor]): List of high resolution images
        lr_imgs(list[Tensor]): List of low resolution and interpolated images
        crop_size(int): the function generates (crop_size x crop_size) random
                        crops
        crop_number(int): how many cropped samples will be generated from
                          each image
    Returns:
        image_array_hr: cropped HR samples
        image_array_lr: cropped LR samples
    '''
    hr_samples = []
    lr_samples = []
    while len(hr_imgs) != 0:
        hr = hr_imgs[0]
        lr = lr_imgs[0]
        for _ in range(crop_number):
            random_h = np.random.randint(0, hr.size(dim=1)-crop_size-1)
            random_w = np.random.randint(0, hr.size(dim=2)-crop_size-1)
            hr_samples.append(hr[:, random_h:random_h+crop_size,
                                 random_w:random_w+crop_size].unsqueeze(0))
            lr_samples.append(lr[:, random_h:random_h+crop_size,
                                 random_w:random_w+crop_size].unsqueeze(0))
        hr_imgs.pop(0)
        lr_imgs.pop(0)
    hr_samples = torch.cat(hr_samples, 0)
    lr_samples = torch.cat(lr_samples, 0)
    #mean_image = torch.mean(lr_samples, )
    lr_samples, total_mean = normalize_images(lr_samples)
    return hr_samples, lr_samples, total_mean


def normalize_images(imgs_lr):
    #https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/03/08/image-mean-std.html
    # placeholders
    psum    = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])

    #loop through images
    for img in imgs_lr:
        psum    += img.long().sum(axis=[1, 2])
        psum_sq += (img.long() ** 2).sum(axis=[1, 2])

    # pixel count
    count = len(imgs_lr) * imgs_lr[0].size()[1] * imgs_lr[0].size()[2]

    # mean and std
    total_mean = psum / count
    #total_var  = (psum_sq / count) + (total_mean ** 2) - ((2*total_mean*psum)/count)
    total_var  = (psum_sq / count) - (total_mean ** 2)
    total_std  = torch.sqrt(total_var)

    # output

    total_mean = torch.reshape(total_mean, [3, 1, 1])
    total_std = torch.reshape(total_std, [3, 1, 1])
    return (imgs_lr-total_mean)/total_std, total_mean

    #transform = T.Compose([
    #    T.ToTensor(),
    #    T.Normalize(
    #        mean=total_mean,
    #        std=total_std,
    #    ),
    #])
    #imgs_hr = transform(imgs_hr)
    #imgs_lr = transform(imgs_lr)

    #return imgs_hr, imgs_lr, total_mean

def plot_crops(imgs):
    '''  This function is used for plotting 5 crops to test the crops
    Args:
        imgs(list[Tensor]): Image list to be plotted
    Returns:
        None
    '''
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img.permute(1, 2, 0)))
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()
    plt.show()
