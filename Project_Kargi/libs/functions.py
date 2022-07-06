
import torch
import numpy as np
from torch.utils import data
from pathlib import Path
from PIL import Image

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


def _mat_sqrt(x):
    U, D, V = torch.svd(x)
    return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())


def coral(source, target):
    # assume both source and target are 3D array (C, H, W)
    # Note: flatten -> f

    source_f, source_f_mean, source_f_std = _calc_feat_flatten_mean_std(source)
    source_f_norm = (source_f - source_f_mean.expand_as(
        source_f)) / source_f_std.expand_as(source_f)
    source_f_cov_eye = \
        torch.mm(source_f_norm, source_f_norm.t()) + torch.eye(3)

    target_f, target_f_mean, target_f_std = _calc_feat_flatten_mean_std(target)
    target_f_norm = (target_f - target_f_mean.expand_as(
        target_f)) / target_f_std.expand_as(target_f)
    target_f_cov_eye = \
        torch.mm(target_f_norm, target_f_norm.t()) + torch.eye(3)

    source_f_norm_transfer = torch.mm(
        _mat_sqrt(target_f_cov_eye),
        torch.mm(torch.inverse(_mat_sqrt(source_f_cov_eye)),
                 source_f_norm)
    )

    source_f_transfer = source_f_norm_transfer * \
                        target_f_std.expand_as(source_f_norm) + \
                        target_f_mean.expand_as(source_f_norm)

    return source_f_transfer.view(source.size())


import matplotlib.pyplot as plt



def show_image_result(multiple_contents, index, save_name = "default", title = "Plot"):
    fig = plt.figure(figsize=(8, 8))
    # (Model,Image Size, 3, Batch, Channel,Height,Width) 
    default = multiple_contents[0][index]
    balanced = multiple_contents[1][index]
    pretrained = multiple_contents[2][index]

    fig.add_subplot(2,3,1)
    plt.xlabel("Content")
    plt.imshow(default[0][0].cpu().permute(1, 2, 0))

    fig.add_subplot(2,3,2)
    plt.xlabel("Style")
    plt.imshow(default[1][0].cpu().permute(1, 2, 0) )

    fig.add_subplot(2,3,4)
    plt.xlabel("Default")
    plt.imshow(default[2][0].cpu().permute(1, 2, 0) )

    fig.add_subplot(2,3,5)
    plt.xlabel("Balanced")
    plt.imshow(balanced[2][0].cpu().permute(1, 2, 0) )

    fig.add_subplot(2,3,6)
    plt.xlabel("Pretrained")
    plt.imshow(pretrained[2][0].cpu().permute(1, 2, 0) )
    
    plt.suptitle(title)
    plt.show()
    plt.savefig(f"{save_name}.png")
    

def show_loss_distribution(loss_list, save_name = "dist", title = "Plot"):

    from matplotlib import colors

    n_bins = 50
    fig, axs = plt.subplots(1, 1, tight_layout=True)

    # N is the count in each bin, bins is the lower-limit of the bin
    N, bins, patches = axs.hist(loss_list, bins=n_bins)

    # We'll color code by height, but you could use any scalar
    fracs = N / N.max()

    # we need to normalize the data to 0..1 for the full range of the colormap
    norm = colors.Normalize(fracs.min(), fracs.max())

    # Now, we'll loop through our objects and set the color of each accordingly
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)

    plt.title(title)
    plt.show()
    plt.savefig(f"{save_name}.png")

def show_relation_graph(x_,y_, save_name = "relation", sub_titles = ["r11","r21","r31","r41"]):
    fig = plt.figure(figsize=(10, 10)) 
    x = [el for el in x_[0]]
    y = [el for el in y_[0]]
    fig.add_subplot(2,2,1)
    plt.xlabel("Classic Style Loss")
    plt.scatter(x,y, alpha = 0.5)
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
    plt.ylabel("Balance term")
    plt.title(sub_titles[0])

    x = [el for el in x_[1]]
    y = [el for el in y_[1]]
    fig.add_subplot(2,2,2)
    plt.xlabel("Classic Style Loss")
    plt.scatter(x,y, alpha = 0.5)
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
    plt.ylabel("Balance term")
    plt.title(sub_titles[1])

    x = [el for el in x_[2]]
    y = [el for el in y_[2]]
    fig.add_subplot(2,2,3)
    plt.xlabel("Classic Style Loss")
    plt.scatter(x,y, alpha = 0.5)
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
    plt.ylabel("Balance term")
    plt.title(sub_titles[2])

    x = [el for el in x_[3]]
    y = [el for el in y_[3]]
    fig.add_subplot(2,2,4)
    plt.xlabel("Classic Style Loss")
    plt.scatter(x,y, alpha = 0.5)
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
    plt.ylabel("Balance term")
    plt.title(sub_titles[3])
    plt.show()
    plt.savefig(f"{save_name}.png")
    

def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'