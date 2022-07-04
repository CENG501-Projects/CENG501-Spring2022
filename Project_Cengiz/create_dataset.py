import argparse
import os
from PIL import Image
import re
import numpy as np
from math import floor


def main():
    # TODO: Add args
    parser = argparse.ArgumentParser(description='Train Downscaling kernel')
    parser.add_argument('--dataset_base_path', default="/home/baran/Documents/datasets/DIV2K", type=str, help='path to the downloaded dataset')
    parser.add_argument('--train_save_path', default="/train_GT", type=str, help='path to train validation images')
    parser.add_argument('--val_save_path', default="/val_GT", type=str, help='path to save validation images')
    _args = parser.parse_args()
    # Path of the downloaded dataset
    base_path = _args.dataset_base_path
    train_path = base_path + "/dataset_train_128"
    val_path = base_path + "/dataset_val_128"
    # Create related folders
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(val_path):
        os.makedirs(val_path)
    
    train_gt_path = base_path + _args.train_save_path
    val_gt_path = base_path + _args.val_save_path

    # Generate train dataset
    if os.path.exists(base_path + "/DIV2K_train_HR"):
        if not os.path.exists(train_gt_path):
            os.makedirs(train_gt_path)
            # Downsample HR images to generate cleaner images.
            # Frequency Separation for Real-World Super-Resolution, https://arxiv.org/abs/1911.07850
            input_folder = base_path + "/DIV2K_train_HR"
            # input_folder = base_path + "/DIV2K_train_HR"
            for image_name in sorted(os.listdir(input_folder)):
                img = Image.open(input_folder + "/" + image_name)
                if(img.size[0] >= 1024 and img.size[1] >= 1024):
                    img_interpolated = interpolate(img, scale=0.5)
                    img_interpolated.save(train_gt_path + "/" + image_name)

        input_folder = train_gt_path
        # Different scales are used for the curriculum learning.
        # Generate different datasets for different scales for speeding up the training process.
        scales = [3.5, 1.2]
        for scale in scales:
            print(f"Processing... scale: {scale}")
            for image_name in sorted(os.listdir(input_folder)):
                name, ext = os.path.splitext(image_name)
                img = Image.open(input_folder + "/" + image_name)

                # Create upsampled and downsampled images
                img_down = interpolate(img, 1/scale)
                img_up = interpolate(img, scale)
                
                # Increase stride, otherwise too many images will be generated.
                stride=4
                idx_path = train_path + "/" + name
                
                if not os.path.exists(idx_path + "/" + "patches512"):
                    os.makedirs(idx_path + "/" + "patches512")
                if not os.path.exists(idx_path + "/" + "patches128"):
                    os.makedirs(idx_path + "/" + "patches128")
                    # Extract 128x128 patches from the GT images to create LR inputs.
                if not os.path.exists(idx_path + "/" + f"down_{scale}"):
                    os.makedirs(idx_path + "/" + f"down_{scale}")
                    os.makedirs(idx_path + "/" + f"up_{scale}")

                # Calculate once, does not depend on scale
                if scale == scales[0]:
                    extract_patches_and_save(img, stride, output_base=idx_path + "/patches512", patch_size=512)
                    extract_patches_and_save(img, stride, output_base=idx_path + "/patches128", patch_size=128)
                
                extract_patches_and_save(img_down, stride, output_base=idx_path + f"/down_{scale}", patch_size=128)
                # Increase stride, otherwise too many images will be generated.
                stride = floor(scale) * 4
                extract_patches_and_save(img_up, stride, output_base=idx_path + f"/up_{scale}", patch_size=128)
                print(f"Saved image: {image_name}")

    # Generate validation dataset
    if os.path.exists(base_path + "/DIV2K_valid_HR"):
        if not os.path.exists(val_gt_path):
            os.makedirs(val_gt_path)
            # Downsample HR images to generate cleaner images.
            # Frequency Separation for Real-World Super-Resolution, https://arxiv.org/abs/1911.07850
            input_folder = base_path + "/DIV2K_valid_HR"
            # input_folder = base_path + "/DIV2K_train_HR"
            for image_name in sorted(os.listdir(input_folder)):
                img = Image.open(input_folder + "/" + image_name)
                if(img.size[0] >= 1024 and img.size[1] >= 1024):
                    img_interpolated = interpolate(img, scale=0.5)
                    img_interpolated.save(val_gt_path + "/" + image_name)

        input_folder = val_gt_path
        # Different scales are used for the curriculum learning.
        # Generate different datasets for different scales for speeding up the training process.
        scales = [3.5, 1.2]
        for scale in scales:
            print(f"Processing... scale: {scale}")
            for image_name in sorted(os.listdir(input_folder)):
                name, ext = os.path.splitext(image_name)
                img = Image.open(input_folder + "/" + image_name)

                # Create upsampled and downsampled images
                img_down = interpolate(img, 1/scale)
                img_up = interpolate(img, scale)
                
                # Increase stride, otherwise too many images will be generated.
                stride=4
                idx_path = val_path + "/" + name
                
                if not os.path.exists(idx_path + "/" + "patches512"):
                    os.makedirs(idx_path + "/" + "patches512")
                if not os.path.exists(idx_path + "/" + "patches128"):
                    os.makedirs(idx_path + "/" + "patches128")
                    # Extract 128x128 patches from the GT images to create LR inputs.
                if not os.path.exists(idx_path + "/" + f"down_{scale}"):
                    os.makedirs(idx_path + "/" + f"down_{scale}")
                    os.makedirs(idx_path + "/" + f"up_{scale}")

                # Calculate once, does not depend on scale
                if scale == scales[0]:
                    extract_patches_and_save(img, stride, output_base=idx_path + "/patches512", patch_size=512)
                    extract_patches_and_save(img, stride, output_base=idx_path + "/patches128", patch_size=128)
                
                extract_patches_and_save(img_down, stride, output_base=idx_path + f"/down_{scale}", patch_size=128)
                # Increase stride, otherwise too many images will be generated.
                stride = floor(scale) * 4
                extract_patches_and_save(img_up, stride, output_base=idx_path + f"/up_{scale}", patch_size=128)
                print(f"Saved image: {image_name}")


def create_dir(base_path, dir):
    if not os.path.exists(base_path + "/" + dir):
        os.makedirs(base_path + "/" + dir)
        print("Created Directory : ", base_path + "/" + dir)


def extract_patches_and_save(img_interpolated, stride, output_base, patch_size=512, ext=".png"):
    idx = 0
    np_img = np.asarray(img_interpolated)
    for idx_x in range(0, np_img.shape[0]//patch_size, stride):
        for idx_y in range(0, np_img.shape[1]//patch_size, stride):
            x_offset = patch_size*idx_x
            y_offset = patch_size*idx_y
            patch = np_img[x_offset:x_offset+patch_size, y_offset:y_offset+patch_size]
            im_crop = Image.fromarray(patch)
            im_crop.save(output_base + "/" + str(idx) + ext)
            idx = idx + 1


def interpolate(img, scale):
    return img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), resample=Image.BICUBIC)
    # Warning: Above line might give an error depending on the pillow version.
    # If that is the case try the code below instead
    # return img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), resample=Image.Resampling.BICUBIC)


if __name__ == "__main__":
    main()
