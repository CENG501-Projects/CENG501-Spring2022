import argparse

from model import AdaptationGenerator
import time
from sys import exit
import torchvision.transforms as T

import torch
from PIL import Image
import os
from tensorboardX import SummaryWriter


def main():
    parser = argparse.ArgumentParser(description='Train Downscaling kernel')
    parser.add_argument('--checkpoint', default="./checkpoints/last_epoch.tar", type=str, help='input checkpoint')
    parser.add_argument('--input_data', default="./test_inputs", type=str, help='location of the train dataset')
    parser.add_argument('--output_data', default="./test_outputs", type=str, help='location of the val dataset')
    _args = parser.parse_args()

    if torch.cuda.is_available():
        print("Cuda (GPU support) is available and enabled!")
        device = torch.device("cuda")
    else:
        print("Cuda (GPU support) is not available :(")
        device = torch.device("cpu")

    if not os.path.exists(_args.output_data):
        os.makedirs(_args.output_data)

    print("# Initializing model")
    adapt_gen = AdaptationGenerator()
    adapt_gen = adapt_gen.to(device)
    adapt_gen.eval()

    # Load from checkpoint if it exists
    if _args.checkpoint is not None:
        print(f"Trying to load checkpoint: {_args.checkpoint}")
        checkpoint = torch.load(_args.checkpoint)
        adapt_gen.load_state_dict(checkpoint["model_adapt_gen_state_dict"])
        print("Loaded checkpoint")
    else:
        print("Please provide a checkpoint")
        exit(0)

    pillow2tensor = T.ToTensor()
    tensor2pillow = T.ToPILImage()
    # For all inputs, generate, and save outputs
    for image_name in sorted(os.listdir(_args.input_data)):
        img = Image.open(_args.input_data + "/" + image_name)
        img = pillow2tensor(img)
        out_img = test(adapt_gen, img, device)
        out_img = tensor2pillow(out_img)
        out_img.show()
        out_img.save(_args.output_data + "/" + image_name)

def test(adapt_gen, input_image, device):
    with torch.no_grad():
        input_image = input_image.to(device)
        input_image = torch.unsqueeze(input_image, dim=0)
        # Generate LR image
        out = adapt_gen(input_image)
        out = torch.squeeze(out)
        out = out.to('cpu')
        return out

if __name__ == "__main__":
    main()
