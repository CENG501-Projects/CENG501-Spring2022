import argparse
from email.policy import default
import torch
from torch.utils import data
import numpy as np
import copy
import torch.nn as nn
from PIL import Image
from torchvision.utils import save_image
from libs.models_sanet import calc_ast_style_loss, calc_ast_style_loss_normalized, calc_ast_style_loss_unnormalized, decoder as decoder,Transform,vgg,test_transform,Net
from tqdm import tqdm
import libs.models_sanet as sanet
import libs.functions as utils
from tqdm import tqdm
from torchvision.utils import save_image
import gc
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#_____________________________________LOAD MODELS_______________________________________#
sanet_default_path = r"experiments\sanet\decoder_default_iter_120000.pth"
transformer_default_path = r"experiments\sanet\transformer_default_iter_120000.pth"

sanet_balanced_path = r"experiments/sanet/decoder_normalized_iter_160000.pth"
transformer_balanced_path = r"experiments/sanet/transformer_normalized_iter_160000.pth"

sanet_pretrained_path = r"experiments\sanet\decoder_pretrained_iter_500000.pth"
transformer_pretrained_path = r"experiments\sanet\transformer_pretrained_iter_500000.pth"


sanet_default = copy.deepcopy(sanet.decoder)
sanet_default.load_state_dict(torch.load(sanet_default_path))
sanet_default.eval()
transformer_default = sanet.Transform(in_planes = 512)
transformer_default.load_state_dict(torch.load(transformer_default_path))
transformer_default.eval()

sanet_balanced = copy.deepcopy(sanet.decoder)
sanet_balanced.load_state_dict(torch.load(sanet_balanced_path))
sanet_balanced.eval()
transformer_balanced = sanet.Transform(in_planes = 512)
transformer_balanced.load_state_dict(torch.load(transformer_balanced_path))
transformer_balanced.eval()

sanet_pretrained = copy.deepcopy(sanet.decoder)
sanet_pretrained.load_state_dict(torch.load(sanet_pretrained_path))
sanet_pretrained.eval()
transformer_pretrained = sanet.Transform(in_planes = 512)
transformer_pretrained.load_state_dict(torch.load(transformer_pretrained_path))
transformer_pretrained.eval()

model_list = [(sanet_default,transformer_default),(sanet_balanced,transformer_balanced),(sanet_pretrained,transformer_pretrained)]

vgg = sanet.vgg
vgg.eval()

vgg.load_state_dict(torch.load("vgg_normalised.pth"))

print(f"Models loaded succesfully. Total model size : {len(model_list)}")

## Switch the device of the models for faster performance
norm = nn.Sequential(*list(vgg.children())[:1])
enc_1 = nn.Sequential(*list(vgg.children())[:4])  # input -> relu1_1
enc_2 = nn.Sequential(*list(vgg.children())[4:11])  # relu1_1 -> relu2_1
enc_3 = nn.Sequential(*list(vgg.children())[11:18])  # relu2_1 -> relu3_1
enc_4 = nn.Sequential(*list(vgg.children())[18:31])  # relu3_1 -> relu4_1
enc_5 = nn.Sequential(*list(vgg.children())[31:44])  # relu4_1 -> relu5_1

norm.to(device)
enc_1.to(device)
enc_2.to(device)
enc_3.to(device)
enc_4.to(device)
enc_5.to(device)

for model in model_list:
    model[0].to(device)
    model[1].to(device)

print(f"Models switched to defice successfully. Device : {device}")

# These classes and functions are key for loading the dataset and sampling from it.




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

args = parser.parse_args()

steps = args.steps
many_image = args.many_image
content_dir = args.content_dir
style_dir = args.style_dir

#_____________________________________DATA PROCESS______________________________________#
content_tf = sanet.test_transform()
style_tf = sanet.test_transform()

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
model_image_contents = []
unnormalized_losses = [[],[],[],[],[]]
normalizer_weights = [[],[],[],[],[]]
default_unnormalized_losses = []

content_set = [next(content_iter).to(device) for i in range(many_image)]
style_set = [next(style_iter).to(device) for i in range(many_image)]

with torch.no_grad():
    for model in model_list:
        image_contents = []
        for im in tqdm(range(many_image)):
            content = content_set[im]
            style = style_set[im]
            
            contents = []
            contents.append(content.cpu())
            contents.append(style.cpu())
            #print(content.shape)
            for step in range(steps):

                Content4_1 = enc_4(enc_3(enc_2(enc_1(content))))
                Content5_1 = enc_5(Content4_1)
            
                Style1_1 = enc_1(style)
                Style2_1 = enc_2(Style1_1)
                Style3_1 = enc_3(Style2_1)
                Style4_1 = enc_4(Style3_1)
                Style5_1 = enc_5(Style4_1)

                content = model[0](model[1](Content4_1, Style4_1, Content5_1, Style5_1))

                gt1_1 = enc_1(content)
                gt2_1 = enc_2(gt1_1)
                gt3_1 = enc_3(gt2_1)
                gt4_1 = enc_4(gt3_1) # stylized + decoder
                gt5_1 = enc_5(gt4_1)

                g_t_feats = [gt1_1,gt2_1,gt3_1,gt4_1,gt5_1]
                style_feats = [Style1_1,Style2_1,Style3_1,Style4_1,Style5_1]

                loss_s = calc_ast_style_loss(g_t_feats[0], style_feats[0])
                loss_unnormalized = calc_ast_style_loss_unnormalized(g_t_feats[0], style_feats[0])
                loss_normalizer = sanet.return_normalize_weight(g_t_feats[0], style_feats[0])
                unnormalized_losses[0].append(loss_unnormalized) # First relu
                normalizer_weights[0].append(loss_normalizer) # First relu
                loss_unnorm_s = loss_unnormalized

                for i in range(1, 5):
                    loss_normalizer = sanet.return_normalize_weight(g_t_feats[i], style_feats[i])
                    loss_s += calc_ast_style_loss(g_t_feats[i], style_feats[i])
                    loss_unnormalized = calc_ast_style_loss_unnormalized(g_t_feats[i], style_feats[i])
                    loss_unnorm_s += loss_unnormalized
                    normalizer_weights[i].append(loss_normalizer)  # ith relu
                    unnormalized_losses[i].append(loss_unnormalized) # ith relu
                #print("Default : ",loss_s.item())
                default_loss = loss_s
                

                
                loss_s = calc_ast_style_loss_normalized(g_t_feats[0], style_feats[0])
                for i in range(1, 5):
                    loss_s += calc_ast_style_loss_normalized(g_t_feats[i], style_feats[i])
                #print("Balanced : ", loss_s.item())
                balanced_loss = loss_s
            
                # free some memory since when we load the second model, we need this
                del Content4_1
                del Content5_1
                del g_t_feats
                del gt1_1
                del gt2_1
                del gt3_1
                del gt4_1
                del gt5_1
                del Style1_1
                del Style2_1
                del Style3_1
                del style_feats
                del Style4_1
                del Style5_1
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

#utils.show_image_result(model_image_contents,0, title = "SANET")
utils.show_loss_distribution(balanced_losses[1], save_name = "balanced_sanet", title = "Loss distribution of balanced losses for balance style trained model.")
utils.show_loss_distribution(default_unnormalized_losses[2], save_name = "default_sanet_unnorm", title = "Loss distribution of unnormalized classic losses for pretrained models.")
utils.show_relation_graph(unnormalized_losses,normalizer_weights)