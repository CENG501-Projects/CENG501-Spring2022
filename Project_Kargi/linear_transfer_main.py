import argparse
from email.policy import default
import torch
import gc
from tqdm import tqdm
import torch.nn as nn

from libs.models_sanet import calc_ast_style_loss, calc_ast_style_loss_normalized, calc_ast_style_loss_unnormalized,Transform,vgg,test_transform,Net

import libs.functions as utils
import libs.models_sanet as sanet
import libs.models_adain as adain
import libs.models_linear_transfer as lt
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("ok")


#_____________________________________LOAD MODELS_______________________________________#

vgg_r41_path = "experiments/linear-transfer/vgg_r41.pth" # encoder path - pretrained
dec_r41_path = "experiments/linear-transfer/dec_r41.pth" # decoder path - pretrained
loss_network_path = "xperiments/linear-transfer/vgg_r41.pth"

r41_default_path =  "experiments/linear-transfer/r41_default_style5_160000.pth"  #matrix path - trained
r41_normalized_path =  "experiments/linear-transfer/r41_normalized_style5_160000.pth"
r41_pretrained_path = "experiments/linear-transfer/r41.pth"



vgg = sanet.vgg
dec = lt.decoder4()

default_matrix = lt.MulLayer("r41") # we used r41 for training
balanced_matrix = lt.MulLayer("r41") # we used r41 for training
pretrained_matrix = lt.MulLayer("r41") # we used r41 for training


dec.load_state_dict(torch.load(dec_r41_path))

default_matrix.load_state_dict(torch.load(r41_default_path))
balanced_matrix.load_state_dict(torch.load(r41_normalized_path))
pretrained_matrix.load_state_dict(torch.load(r41_pretrained_path))


model_list = [default_matrix, balanced_matrix, pretrained_matrix]

#vgg = adain.vgg
vgg.load_state_dict(torch.load("vgg_normalised.pth"))

print(f"Models loaded succesfully. Total model size : {len(model_list)}")

## Switch the device of the models for faster performance
vgg = nn.Sequential(*list(vgg.children())[:44]) # get the VGG layers

norm = nn.Sequential(*list(vgg.children())[:1])
enc_1 = nn.Sequential(*list(vgg.children())[:4])  # input -> relu1_1
enc_2 = nn.Sequential(*list(vgg.children())[4:11])  # relu1_1 -> relu2_1
enc_3 = nn.Sequential(*list(vgg.children())[11:18])  # relu2_1 -> relu3_1
enc_4 = nn.Sequential(*list(vgg.children())[18:31])  # relu3_1 -> relu4_1
enc_5 = nn.Sequential(*list(vgg.children())[31:44])  # relu4_1 -> relu5_1

enc_1.to(device)
enc_2.to(device)
enc_3.to(device)
enc_4.to(device)
vgg.to(device)
dec.to(device)

for model in model_list:
    model.to(device)

print(f"Models switched to defice successfully. Device : {device}")

# These classes and functions are key for loading the dataset and sampling from it.
from torch.utils import data
from pathlib import Path
import numpy as np

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

content_size = args.content_size
style_size = args.style_size
crop = args.crop


#_____________________________________DATA PROCESS______________________________________#

content_tf = adain.test_transform(content_size, crop) # Center and crop if image size is bigger than 256. Use the transformer
style_tf = adain.test_transform(style_size, crop)     # in the adain.

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

                # Linear tranfer also uses R11 R21 R31 and R41
                Style1_1 = enc_1(style)
                Style2_1 = enc_2(Style1_1)
                Style3_1 = enc_3(Style2_1)
                Style4_1 = enc_4(Style3_1)

                style_feats = [Style1_1,Style2_1,Style3_1,Style4_1]
                # Content uses R41

                Content4_1 = enc_4(enc_3(enc_2(enc_1(content))))

                feature,transmatrix = model(Content4_1,Style4_1)
                transfer = dec(feature) # use this as content!

                # sF_loss = vgg(style)
                # cF_loss = vgg(content)
        
                gt1_1 = enc_1(transfer)
                gt2_1 = enc_2(gt1_1)
                gt3_1 = enc_3(gt2_1)
                gt4_1 = enc_4(gt3_1) # stylized + decoder

                #content = style_transfer(vgg, model, content, style, alpha)

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
                #print("Default : ",loss_s.item())
                default_loss = loss_s
            
                
                loss_s = calc_ast_style_loss_normalized(g_t_feats[0], style_feats[0])
                for i in range(1, 4):
                    loss_s += calc_ast_style_loss_normalized(g_t_feats[i], style_feats[i])
                #print("Balanced : ", loss_s.item())
                balanced_loss = loss_s

                
                content = transfer
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
        model_image_contents.append(image_contents) # ((model contents), (model contents))
        del model

#_____________________________________EXPERIMENTS______________________________________#
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
utils.show_image_result(model_image_contents,low_index, title = "Balanced LinearTransfer, image with small style loss")
utils.show_image_result(model_image_contents,high_index, title = "Balanced LinearTransfer, image with high style loss")



low_to_high_default = np.argsort(default_losses[0])
low_index_bal = low_to_high_default[5]
high_index_bal = low_to_high_default[-5]

utils.show_image_result(model_image_contents,low_index_bal, title = "Default LinearTransfer, image with small style loss")
utils.show_image_result(model_image_contents,high_index_bal, title = "Default LinearTransfer, image with high style loss")

utils.show_loss_distribution(balanced_losses[1], save_name = "balanced_lt", title = "Loss distribution of balanced losses.")
utils.show_loss_distribution(default_unnormalized_losses[2], save_name = "default_lt", title = "Loss distribution of unnormalized classic losses.")
# show_relation_graph(unnormalized_losses[0],normalizer_weights[0], "r1")
# show_relation_graph(unnormalized_losses[1],normalizer_weights[1], "r2")
# show_relation_graph(unnormalized_losses[2],normalizer_weights[2], "r3")
# show_relation_graph(unnormalized_losses[3],normalizer_weights[3], "r4" )
# utils.show_relation_graph(unnormalized_losses,normalizer_weights, "r5")