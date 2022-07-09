# Copyright 2019 Karsten Roth and Biagio Brattoli
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


############################ LIBRARIES ######################################
import torch, os, numpy as np

import torch.nn as nn
import pretrainedmodels as ptm

import pretrainedmodels.utils as utils
import torchvision.models as models

"""============================================================="""
def initialize_weights(model):
    """
    Function to initialize network weights.
    NOTE: NOT USED IN MAIN SCRIPT.

    Args:
        model: PyTorch Network
    Returns:
        Nothing!
    """
    for idx,module in enumerate(model.modules()):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(0,0.01)
            module.bias.data.zero_()

"""============================================================="""
def get_learnable_parameters(model):
    """
    Function to get learnable network parameters.

    Args:
        model: PyTorch Network
    Returns:
        Parameters to be learned
    """
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
    return params_to_update

"""=================================================================================================================================="""
### ATTRIBUTE CHANGE HELPER
def rename_attr(model, attr, name):
    """
    Rename attribute in a class. Simply helper function.

    Args:
        model:  General Class for which attributes should be renamed.
        attr:   str, Name of target attribute.
        name:   str, New attribute name.
    """
    setattr(model, name, getattr(model, attr))
    delattr(model, attr)


"""=================================================================================================================================="""
### NETWORK SELECTION FUNCTION
def networkselect(opt):
    """
    Selection function for available networks.

    Args:
        opt: argparse.Namespace, contains all training-specific training parameters.
    Returns:
        Network of choice
    """
    if opt.arch == 'resnet50':
        network =  ModifiedResNet50(opt)
    else:
        raise Exception('Network {} not available!'.format(opt.arch))
    return network

"""=================================================================================================================================="""
class ModifiedResNet50(nn.Module):
    """
    Container for ResNet50 s.t. it can be used for metric learning.
    The Network has been broken down to allow for higher modularity, if one wishes
    to target specific layers/blocks directly. Modifications:
      - Global Avarage pooling layer is replaced by global max pooling layer. 
      - Last fully connected layer is deactivated (Replaced with identity). 
    """
    def __init__(self, opt, list_style=False, no_norm=False):
        super(ModifiedResNet50, self).__init__()

        self.pars = opt

        if not opt.not_pretrained:
            print('Getting pretrained weights on ImageNet...')
            self.model = models.resnet50(pretrained=True)
            print('Done.')
        else:
            print('Not utilizing pretrained weights!')
            self.model = models.resnet50(pretrained=False)

        for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
            module.eval()
            module.train = lambda _: None

        # Deactivate the last linear fully connected layer, save the output dimension of the new network before deactivation
        self.out_dim = self.model.fc.in_features
        self.model.fc = nn.Identity()

        self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])

    def forward(self, x, is_init_cluster_generation=False):

        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))

        for layerblock in self.layer_blocks:
            x = layerblock(x)

        # Replace the avarage pooling layer with max pooling
        x = torch.nn.functional.max_pool2d(x, kernel_size=x.size()[2:])

        # Flatten
        x = x.view(x.size(0), -1)

        x = self.model.fc(x)

        return x

"""============================================================="""
class IntraNet(nn.Module):
    """

    """
    def __init__(self, opt, backbone_net):
        super().__init__()
        self.out_embded_dim = opt.embed_dim
        self.hidden_layer_dim = opt.g_net_hidden_layer_dim
        self.backbone_net = backbone_net
        self.pars = opt

        # Create MLP with 1 hidden layer
        # g_net is comprised of one hidden layer and one output layer
        self.g_net_hidden = nn.Linear(self.backbone_net.out_dim, self.hidden_layer_dim)
        self.g_net_out = nn.Linear(self.hidden_layer_dim, self.out_embded_dim)

    def forward(self, x):
        x = self.backbone_net(x)
        x = self.g_net_hidden(x)
        x = torch.nn.functional.relu(x)
        x = self.g_net_out(x)

        # Apply L2 normalization on output embeddings (No Normalization is used if N-Pair Loss is the target criterion)
        return x if self.pars.loss=='npair' else nn.functional.normalize(x, dim=-1)

"""============================================================="""
class InterNet(nn.Module):
    """

    """
    def __init__(self, opt, backbone_net):
        super().__init__()
        self.out_embed_dim = opt.embed_dim
        self.backbone_net = backbone_net
        self.pars = opt

        # q_net is comprised of a single linear layer
        self.q_net_linear = nn.Linear(self.backbone_net.out_dim, self.out_embed_dim)

    def forward(self, x):
        # Get h_i's
        x = self.backbone_net(x)
        # Get z_i's
        x = self.q_net_linear(x)

        # Apply L2 normalization on output embeddings (No Normalization is used if N-Pair Loss is the target criterion)
        return x if self.pars.loss=='npair' else nn.functional.normalize(x, dim=-1)
