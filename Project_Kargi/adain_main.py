import argparse
from email.policy import default
import os
import torch
from tqdm import tqdm
import copy
import torch.nn as nn
from PIL import Image
from libs.models_sanet import calc_ast_style_loss, calc_ast_style_loss_normalized, calc_ast_style_loss_unnormalized, decoder as decoder,Transform,vgg,test_transform,Net
from torch.utils import data
import numpy as np
import libs.models_sanet as sanet
import libs.models_adain as adain
import libs.functions as utils

from torchvision.utils import save_image
from libs.models_adain import style_transfer
import gc
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("ok")


#_____________________________________LOAD MODELS_______________________________________#
adain_default_path = r"experiments/adain/decoder_default_iter_160000.pth.tar"

adain_balanced_path = r"experiments/adain/decoder_normalized_iter_160000.pth.tar"

adain_pretrained_path = r"experiments/adain/decoder_pretrained.pth"


adain_default = copy.deepcopy(adain.decoder)
adain_default.load_state_dict(torch.load(adain_default_path))
adain_default.eval()


adain_balanced = copy.deepcopy(sanet.decoder)
adain_balanced.load_state_dict(torch.load(adain_balanced_path))
adain_balanced.eval()

adain_pretrained = copy.deepcopy(sanet.decoder)
adain_pretrained.load_state_dict(torch.load(adain_pretrained_path))
adain_pretrained.eval()

model_list = [adain_default, adain_balanced, adain_pretrained]

vgg = adain.vgg
vgg.load_state_dict(torch.load("vgg_normalised.pth"))

print(f"Models loaded succesfully. Total model size : {len(model_list)}")

## Switch the device of the models for faster performance
vgg = nn.Sequential(*list(vgg.children())[:31]) # get the VGG layers

norm = nn.Sequential(*list(vgg.children())[:1])
enc_1 = nn.Sequential(*list(vgg.children())[:4])  # input -> relu1_1
enc_2 = nn.Sequential(*list(vgg.children())[4:11])  # relu1_1 -> relu2_1
enc_3 = nn.Sequential(*list(vgg.children())[11:18])  # relu2_1 -> relu3_1
enc_4 = nn.Sequential(*list(vgg.children())[18:31])  # relu3_1 -> relu4_1

enc_1.to(device)
enc_2.to(device)
enc_3.to(device)
enc_4.to(device)
vgg.to(device)
for model in model_list:
    model.to(device)

print(f"Models switched to defice successfully. Device : {device}")

#_____________________________________SET PARAMETERS____________________________________#

parser = argparse.ArgumentParser()

# Basic options to evaluate models
parser.add_argument('--steps', type=int, default = 1,
                    help='Steps of style transfer.')
parser.add_argument('--many_image', type=int, default = 2000,
                    help='Total amount of generated pair.')
parser.add_argument('--content_dir', type=str, default = "contents/",
                    help='Path where the content images located.')
parser.add_argument('--style_dir', type=str, default = "styles/",
                    help='Path where the style images located.')
parser.add_argument('--alpha', type=float, default = 1.0,
                    help='Alpha value.')
parser.add_argument('--content_size', type=int, default = 256,
                    help='Size of the content image.')
parser.add_argument('--style_size', type=int, default = 256,
                    help='Size of the style image.')
parser.add_argument('--crop', type=int, default = True,
                    help='Crop the image or not.')
args = parser.parse_args()

steps = args.steps
many_image = args.many_image
content_dir = args.content_dir
style_dir = args.style_dir

alpha = args.alpha
content_size = args.content_size
style_size = args.style_size
crop = args.crop
#_____________________________________DATA PROCESS______________________________________#
content_tf = adain.test_transform(content_size, crop)
style_tf = adain.test_transform(style_size, crop)

content_dataset = utils.FlatFolderDataset(content_dir, content_tf)
style_dataset = utils.FlatFolderDataset(style_dir, style_tf)

content_iter = iter(data.DataLoader(
    content_dataset, batch_size= 1,
    sampler=utils.InfiniteSamplerWrapper(content_dataset)))

style_iter = iter(data.DataLoader(
    style_dataset, batch_size= 1,
    sampler=utils.InfiniteSamplerWrapper(style_dataset)))

#_____________________________________START TESTING______________________________________#
torch.cuda.empty_cache()    

contents = []
multi_content = []
balanced_losses = []
default_losses =[]
default_unnormalized_losses = []

model_image_contents = []
unnormalized_losses = [[],[],[],[]]
normalizer_weights = [[],[],[],[]]

content_set = [next(content_iter).to(device) for i in range(many_image)]
style_set = [next(style_iter).to(device) for i in range(many_image)]

with torch.no_grad():
    for model in model_list:
        image_contents = []
        for im in tqdm(range(many_image)):
            content = content_set[im]
            style = style_set[im]
            
            contents = []
            contents.append(content)
            contents.append(style)
            #print(content.shape)
            for step in range(steps):

                Style1_1 = enc_1(style)
                Style2_1 = enc_2(Style1_1)
                Style3_1 = enc_3(Style2_1)
                Style4_1 = enc_4(Style3_1)

                Content4_1 = vgg(content)

                style_feats = [Style1_1,Style2_1,Style3_1,Style4_1]
                t = adain.adaptive_instance_normalization(Content4_1, Style4_1)
                t = alpha * t + (1 - alpha) * Content4_1

                g_t = model(t)
                
                gt1_1 = enc_1(g_t)
                gt2_1 = enc_2(gt1_1)
                gt3_1 = enc_3(gt2_1)
                gt4_1 = enc_4(gt3_1) # stylized + decoder

                content = style_transfer(vgg, model, content, style, alpha)

                g_t_feats = [gt1_1,gt2_1,gt3_1,gt4_1]
                

                loss_s = calc_ast_style_loss(g_t_feats[0], style_feats[0])
                loss_unnormalized = calc_ast_style_loss_unnormalized(g_t_feats[0], style_feats[0])
                loss_normalizer = sanet.return_normalize_weight(g_t_feats[0], style_feats[0])

                normalizer_weights[0].append(loss_normalizer) # First relu
                unnormalized_losses[0].append(loss_unnormalized) # First relu
                loss_unnorm_s = loss_unnormalized
                for i in range(1, 4):
                    loss_normalizer = sanet.return_normalize_weight(g_t_feats[i], style_feats[i])
                    loss_s += calc_ast_style_loss(g_t_feats[i], style_feats[i])
                    loss_unnormalized = calc_ast_style_loss_unnormalized(g_t_feats[i], style_feats[i])
                    loss_unnorm_s += loss_unnormalized
                    normalizer_weights[i].append(loss_normalizer)  # ith relu
                    unnormalized_losses[i].append(loss_unnormalized) # ith relu
                #print("loss_unnorm_s : ",loss_unnorm_s.item())
                default_loss = loss_s
            
                
                loss_s = calc_ast_style_loss_normalized(g_t_feats[0], style_feats[0])
                for i in range(1, 4):
                    loss_s += calc_ast_style_loss_normalized(g_t_feats[i], style_feats[i])
                #print("Balanced : ", loss_s.item())
                balanced_loss = loss_s

                
                
                # free some memory since when we load the second model, we need this
                del Content4_1
                del g_t_feats
                del gt1_1
                del gt2_1
                del gt3_1
                del gt4_1
                del Style1_1
                del Style2_1
                del Style3_1
                del style_feats
                del Style4_1
                gc.collect()
            content.clamp(0, 255)
            contents.append(content.detach().cpu())
            balanced_losses.append(balanced_loss.cpu())
            default_losses.append(default_loss.cpu())
            default_unnormalized_losses.append(loss_unnorm_s.cpu())
            image_contents.append(contents) # (content,style,stylized), (content,style,stylized)..
            del content
            del style
        model_image_contents.append(image_contents) # ((model contents), (model contents))
    del model

import matplotlib.pyplot as plt
# losses are in shape model x steps x imsize -> model, steps x imsize
default_losses = np.array([los.cpu() for los in default_losses])
default_losses = default_losses.reshape(len(model_list),-1)

balanced_losses = np.array([los.cpu() for los in balanced_losses])
balanced_losses = balanced_losses.reshape(len(model_list),-1)

default_unnormalized_losses = np.array([los.cpu() for los in default_unnormalized_losses])
default_unnormalized_losses = default_unnormalized_losses.reshape(len(model_list),-1)

for i in range(len(model_list)):
    print("___________________________________")
    print(f"Default loss : {np.mean(default_losses[i])}")
    print(f"Balanced loss : {np.mean(balanced_losses[i])}")
    print(f"Unnormalized loss : {np.mean(default_unnormalized_losses[i])}")
    print("___________________________________")


low_to_high = np.argsort(balanced_losses[1])
low_index = low_to_high[5]
high_index = low_to_high[-5]
utils.show_image_result(model_image_contents,low_index, title = "Balanced AdaIN, image with small style loss")
utils.show_image_result(model_image_contents,high_index, title = "Balanced AdaIN, image with high style loss")



low_to_high_default = np.argsort(default_losses[0])
low_index_bal = low_to_high_default[5]
high_index_bal = low_to_high_default[-5]

utils.show_image_result(model_image_contents,low_index_bal, title = "Default AdaIN, image with small style loss")
utils.show_image_result(model_image_contents,high_index_bal, title = "Default AdaIN, image with high style loss")

utils.show_loss_distribution(balanced_losses[1], save_name = "balanced_adain", title = "Loss distribution of balanced losses for balance style trained model.")
utils.show_loss_distribution(default_unnormalized_losses[2], save_name = "default_adain_unnorm", title = "Loss distribution of unnormalized classic losses for pretrained models.")

"""
for i in range(len(unnormalized_losses)):
    unnormalized_losses[i] = np.array([los.cpu() for los in unnormalized_losses[i]])
    normalizer_weights[i] = np.array([los.cpu() for los in normalizer_weights[i]])
utils.show_relation_graph(unnormalized_losses,normalizer_weights, "r5")
"""
