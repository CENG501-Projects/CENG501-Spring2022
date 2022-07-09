# Copyright 2019 Karsten Roth and Biagio Brattoli
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License 
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


#################### LIBRARIES ########################
import warnings

from sklearn import metrics
warnings.filterwarnings("ignore")

import os, sys, numpy as np, argparse, imp, datetime, time, pickle as pkl, random, json
os.chdir(os.path.dirname(os.path.realpath(__file__)))

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from tqdm import tqdm
import pandas as pd

import torch, torch.nn as nn
# import auxiliaries_nofaiss as aux
import auxiliaries as aux
import datasets as data

import netlib as netlib
import losses as losses
import evaluate as eval

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from torch.utils.tensorboard import SummaryWriter


################### INPUT ARGUMENTS ###################
parser = argparse.ArgumentParser()

####### Main Parameter: Dataset to use for Training
parser.add_argument('--dataset',           default='cub200',   type=str, help='Dataset to use.')

### General Training Parameters
parser.add_argument('--lr',                default=0.0001,    type=float, help='Learning Rate for network parameters.')
parser.add_argument('--fc_lr_mul',         default=10,         type=float, help='OPTIONAL: Multiply the embedding layer learning rate by this value. If set to 0, the embedding layer shares the same learning rate.')
parser.add_argument('--n_epochs',          default=100,         type=int,   help='Number of training epochs.')
parser.add_argument('--kernels',           default=8,          type=int,   help='Number of workers for pytorch dataloader.')
parser.add_argument('--bs',                default=120,        type=int,   help='Mini-Batchsize to use.')
parser.add_argument('--samples_per_class', default=4,         type=int,   help='Number of samples in one class drawn before choosing the next class. Set to >1 for losses other than ProxyNCA.')
parser.add_argument('--seed',              default=1,          type=int,   help='Random seed for reproducibility.')
parser.add_argument('--scheduler',         default='step',     type=str,   help='Type of learning rate scheduling. Currently: step & exp.')
parser.add_argument('--gamma',             default=0.5,        type=float, help='Learning rate reduction after tau epochs.')
parser.add_argument('--decay',             default=0.0004,     type=float, help='Weight decay for optimizer.')
parser.add_argument('--tau',               default=[30,50], nargs='+',type=int,help='Stepsize(s) before reducing learning rate.')
parser.add_argument('--p_task',            default=0.8,        type=float, help='Probablity of auxiliary self-supervised framework execution within each epoch.')

##### Loss-specific Settings
parser.add_argument('--loss',                 default='triplet', type=str,   help='loss options: marginloss, triplet, npair')
parser.add_argument('--sampling',             default='semihard',   type=str,   help='For triplet-based losses: Modes of Sampling: random, semihard, distance.')
parser.add_argument('--multitask_loss_gamma', default=0.8,          type=float, help='Weights the importance of self-supervised ranking loss.')

### MarginLoss
parser.add_argument('--margin',        default=0.2,          type=float, help='TRIPLET/MARGIN: Margin for Triplet-based Losses')
parser.add_argument('--beta_lr',       default=0.0005,       type=float, help='MARGIN: Learning Rate for class margin parameters in MarginLoss')
parser.add_argument('--beta',          default=1.2,          type=float, help='MARGIN: Initial Class Margin Parameter in Margin Loss')
parser.add_argument('--nu',            default=0,            type=float, help='MARGIN: Regularisation value on betas in Margin Loss.')
parser.add_argument('--beta_constant', action='store_true',              help='MARGIN: Use constant, un-trained beta.')

### NPair L2 Penalty
parser.add_argument('--l2npair',       default=0.02,         type=float, help='NPAIR: Penalty-value for non-normalized N-PAIR embeddings.')

### Ranking Settings
parser.add_argument('--ranking_bs',                  default=20,   type=int,   help='Batch size for augmented views')
parser.add_argument('--ranking_distortion_strength', default=0.2,  type=float, help='Controls the distortion strength for augmented images used in self-supervised framework.')
parser.add_argument('--ranking_N',                   default=4,    type=int,   help='Differerent transformation number applied to single sample')
parser.add_argument('--ranking_s',                   default=12,   type=float, help='Ranking loss scale factor')
parser.add_argument('--ranking_alfa',                default=0.05, type=float, help='Ranking loss alfa margin.')
parser.add_argument('--ranking_beta',                default=0.5,  type=float, help='Ranking loss boundary constraint.')
parser.add_argument('--ranking_lambda',              default=1.0,  type=float, help='Controls the balance between L-sort and L-pos in ranking loss calculation.')

##### Evaluation Settings
parser.add_argument('--k_vals',     nargs='+',  default=[1,2,4,8], type=int, help='Recall @ Values.')

##### Network parameters
parser.add_argument('--embed_dim',              default=128,         type=int, help='Embedding dimensionality of the network. Note: in literature, dim=128 is used for ResNet50 and dim=512 for GoogLeNet.')
parser.add_argument('--g_net_hidden_layer_dim', default = 512,       type=int, help='Hidden layer dimensionlaity of the g Net used by IntraNet ' )
parser.add_argument('--arch',                   default='resnet50',  type=str, help='Network backend choice: resnet50')
parser.add_argument('--not_pretrained',         action='store_true',           help='If added, the network will be trained WITHOUT ImageNet-pretrained weights.')
parser.add_argument('--num_frozen_submodules',  default=7,           type=int, help='Number of child in the ResNet50 to be frozen.')

##### Setup Parameters
parser.add_argument('--gpu',      default=0,       type=int,   help='GPU-id for GPU to use.')
parser.add_argument('--savename', default='Pro_3', type=str,   help='Save folder name if any special information is to be included.')

### Paths to datasets and storage folder
parser.add_argument('--source_path',           default=os.getcwd()+'/Datasets',                type=str, help='Path to training data.')
parser.add_argument('--save_path',             default=os.getcwd()+'/Training_Results',        type=str, help='Where to save everything.')
parser.add_argument('--tensorboard_save_path', default=os.getcwd()+'/TensorBoard/Run_17', type=str, help='Path to perform saving.')

##### Read in parameters
opt = parser.parse_args()

"""============================================================================"""
opt.source_path += '/'+opt.dataset
opt.save_path   += '/'+opt.dataset

if opt.dataset=='online_products':
    opt.k_vals = [1,10,100,1000]
if opt.dataset=='in-shop':
    opt.k_vals = [1,10,20,30,50]
if opt.dataset=='vehicle_id':
    opt.k_vals = [1,5]

assert not opt.bs%opt.samples_per_class, 'Batchsize needs to fit number of samples per class for distance sampling and margin/triplet loss!'

if opt.loss == 'npair': opt.sampling = 'None'

opt.pretrained = not opt.not_pretrained

"""============================================================================"""
################### TENSORBOARD ###########################
writer = SummaryWriter(opt.tensorboard_save_path)

"""============================================================================"""
################### GPU SETTINGS ###########################
os.environ["CUDA_DEVICE_ORDER"]   ="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= str(opt.gpu)
opt.device = torch.device('cuda')
torch.cuda.empty_cache()

"""============================================================================"""
#################### SEEDS FOR REPROD. #####################
torch.backends.cudnn.deterministic=True
np.random.seed(opt.seed); random.seed(opt.seed)
torch.manual_seed(opt.seed); torch.cuda.manual_seed(opt.seed); torch.cuda.manual_seed_all(opt.seed)


"""============================================================================"""
##################### NETWORK SETUP ##################
# Create backbone network
backbone = netlib.ModifiedResNet50(opt)

# Freeze early layers of backbone
for ct, child in enumerate(backbone.model.children(), 0):
  if ct < opt.num_frozen_submodules:  
    for param in child.parameters():
      param.requires_grad = False

# Create the network which is responsible for inter-class variance
inter_net = netlib.InterNet(opt, backbone)

# Create the network which is responsible for intra-class variance
intra_net = netlib.IntraNet(opt, backbone)

print("Inter and Intra Networks created!")
# print("============ Backbone Network Summary ============")
# print(backbone.model)
# print("============ Inter Network Summary ============")
# print(inter_net)
# print("============ Intra Network Summary ============")
# print(intra_net)

#Push to Device
_          = intra_net.to(opt.device)
_          = inter_net.to(opt.device)
print("Cuda setup is okay!")

# Get parameter lists of networks
backbone_params = netlib.get_learnable_parameters(backbone.model)
q_net_params = list(inter_net.q_net_linear.parameters())
g_net_out_params = list(intra_net.g_net_out.parameters())
g_net_hidden_params = list(intra_net.g_net_hidden.parameters())
g_net_params = g_net_out_params + g_net_hidden_params
all_params = backbone_params + g_net_params + q_net_params

if 'fc_lr_mul' in vars(opt).keys() and opt.fc_lr_mul!=0:
    # Increase the learning rate for the output layers by a factor of "fc_lr_mul"
    to_optim = [{'params':backbone_params + g_net_hidden_params,'lr':opt.lr,'weight_decay':opt.decay},
                {'params':g_net_out_params + q_net_params,'lr':opt.lr*opt.fc_lr_mul,'weight_decay':opt.decay}] 
else:
    to_optim = [{'params':all_params,'lr':opt.lr,'weight_decay':opt.decay}]


"""============================================================================"""
#################### DATALOADER SETUPS ##################
#Returns a dictionary containing 'training', 'testing', and 'evaluation' dataloaders.
#The 'testing'-dataloader corresponds to the validation set, and the 'evaluation'-dataloader
#Is simply using the training set, however running under the same rules as 'testing' dataloader,
#i.e. no shuffling and no random cropping.
dataloaders      = data.give_dataloaders(opt.dataset, opt)
#Because the number of supervised classes is dataset dependent, we store them after
#initializing the dataloader
opt.num_classes  = len(dataloaders['training'].dataset.avail_classes)


"""============================================================================"""
#################### CREATE LOGGING FILES ###############
#Each dataset usually has a set of standard metrics to log. aux.metrics_to_examine()
#returns a dict which lists metrics to log for training ('train') and validation/testing ('val')

metrics_to_log = aux.metrics_to_examine(opt.dataset, opt.k_vals)
# example output: {'train': ['Epochs', 'Time', 'Train Loss', 'Time'],
#                  'val': ['Epochs','Time','NMI','F1', 'Recall @ 1','Recall @ 2','Recall @ 4','Recall @ 8']}

#Using the provided metrics of interest, we generate a LOGGER instance.
#Note that 'start_new' denotes that a new folder should be made in which everything will be stored.
#This includes network weights as well.
LOG = aux.LOGGER(opt, metrics_to_log, name='Multi-Task', start_new=True)

"""============================================================================"""
#################### LOSS SETUP ####################
#Depending on opt.loss and opt.sampling, the respective criterion is returned,
#and if the loss has trainable parameters, to_optim is appended.

# Metric Loss
metric_criterion, to_optim = losses.loss_select(opt.loss, opt, to_optim)
_ = metric_criterion.to(opt.device)

# Ranking loss
ranking_criterion, to_optim = losses.loss_select('ranking', opt, to_optim)
_ = ranking_criterion.to(opt.device)

"""============================================================================"""
#################### OPTIM SETUP ####################
#As optimizer, AdamW with standard parameters is used.
optimizer = torch.optim.AdamW(to_optim)

if opt.scheduler=='exp':
    scheduler    = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=opt.gamma)
elif opt.scheduler=='step':
    scheduler    = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.tau, gamma=opt.gamma)
elif opt.scheduler=='none':
    print('No scheduling used!')
else:
    raise Exception('No scheduling option for input: {}'.format(opt.scheduler))


"""============================================================================"""
#################### TRAINER FUNCTION ############################
def train_one_epoch(train_dataloader, model, optimizer, criterion, opt, epoch, learn_ranking=False):
    """
    This function is called every epoch to perform training of the network over one full
    (randomized) iteration of the dataset.

    Args:
        train_dataloader: torch.utils.data.DataLoader, returns (augmented) training data.
        model:            Tuple of Networks to train (Main Model, Auxiliary Model).
        optimizer:        Optimizer to use for training.
        criterion:        Tuple of Criterions (Metric Criterion, Ranking Criterion).
        opt:              argparse.Namespace, Contains all relevant parameters.
        epoch:            int, Current epoch.
        learn_ranking:    Flag specifiying whether ranking learning will be performed or not

    Returns:
        Nothing!
    """
    print('Learn Ranking: ', learn_ranking) 

    main_model = model[0]
    metric_criterion = criterion[0]
    if learn_ranking:
        aux_model = model[1]
        ranking_criterion = criterion[1]

    metric_loss_accum = 0
    ranking_loss_accum = 0

    start = time.time()

    data_iterator = tqdm(train_dataloader, desc='Epoch {} Training...'.format(epoch))
    for i,(class_labels, input) in enumerate(data_iterator):
        # Ensure gradients are set to zero at beginning
        optimizer.zero_grad()

        # Compute embeddings for input batch.
        z_embeddings = main_model(input.to(opt.device))

        # Compute metric loss.
        metric_loss = metric_criterion(z_embeddings, class_labels)

        # Accumulate loss per iteration.
        metric_loss_accum += metric_loss.detach().item()
 
        # Compute gradients for metric loss
        metric_loss.backward()

        # Update backbone and q_net weights using comp. gradients.
        optimizer.step()

        if learn_ranking:
            # Ensure gradients are set to zero at beginning
            optimizer.zero_grad()

            # Create extended batch with augmentation
            batch_mask = list(np.random.permutation(opt.bs)[:opt.ranking_bs])
            masked_batch = input[batch_mask]
            ranking_input = data.ranking_batch_loader(opt, masked_batch)

            # Compute embeddings for ranking input batch.
            z_prime_embeddings = aux_model(ranking_input.to(opt.device))

            # Compute ranking loss, weighted with gamma 
            ranking_loss = opt.multitask_loss_gamma * ranking_criterion(z_prime_embeddings)   

            # Accumulate loss per iteration.
            ranking_loss_accum += ranking_loss.detach().item()  

            # Compute gradients for ranking loss
            ranking_loss.backward()

            # Update backbone and g_net weights using comp. gradients.
            optimizer.step()
          
        if i==len(train_dataloader)-1:
            data_iterator.set_postfix({'Mean Metric Loss': metric_loss_accum/len(train_dataloader), 'Mean Ranking Loss': ranking_loss_accum/len(train_dataloader)})

    # #Save metrics
    # LOG.log('train', LOG.metrics_to_log['train'], [epoch, np.round(time.time()-start,4), np.mean(multitask_loss_collect)])

    return  metric_loss_accum/len(train_dataloader), ranking_loss_accum/len(train_dataloader) 


"""============================================================================"""
"""========================== MAIN TRAINING PART =============================="""
"""============================================================================"""
################### SCRIPT MAIN ##########################
# Create network tuple
models = (inter_net, intra_net)
# Create criterion tuple
criterions = (metric_criterion, ranking_criterion)

print('\n-----\n')

metric_loss_list = []
ranking_loss_list = []
for epoch in range(opt.n_epochs):
    ## Print current learning rates for all parameters
    if opt.scheduler!='none': print('Running with learning rates {}...'.format(' | '.join('{}'.format(x) for x in scheduler.get_lr())))

    ### Train one epoch
    _ = models[0].train()
    _ = models[1].train()

    # Generate random number
    random_number = np.random.rand()

    # Train the auxiliary framework with probablity "p_task"
    metric_loss_epoch, ranking_loss_epoch = train_one_epoch(dataloaders['training'], models, optimizer, criterions, opt, epoch, random_number<opt.p_task)
    # metric_loss_epoch, ranking_loss_epoch = train_one_epoch(dataloaders['training'], models, optimizer, criterions, opt, epoch, False)
    metric_loss_list.append(metric_loss_epoch)

    # Log losses to Tensorboard
    writer.add_scalar('Metric Loss/Train', metric_loss_epoch, epoch)
    if random_number<opt.p_task: 
        ranking_loss_list.append(ranking_loss_epoch)
        writer.add_scalar('Ranking Loss/Train', ranking_loss_epoch, epoch)

    ### Evaluate
    if (epoch+1) % 5 == 0:  
        _ = inter_net.eval()

        #Each dataset requires slightly different dataloaders.
        if opt.dataset in ['cars196', 'cub200', 'online_products']:
            eval_params = {'dataloader':dataloaders['testing'], 'model':inter_net, 'opt':opt, 'epoch':epoch}
        elif opt.dataset=='in-shop':
            eval_params = {'query_dataloader':dataloaders['testing_query'], 'gallery_dataloader':dataloaders['testing_gallery'], 'model':inter_net, 'opt':opt, 'epoch':epoch}
        elif opt.dataset=='vehicle_id':
            eval_params = {'dataloaders':[dataloaders['testing_set1'], dataloaders['testing_set2'], dataloaders['testing_set3']], 'model':inter_net, 'opt':opt, 'epoch':epoch}
        
        # #Compute Evaluation metrics, print them and store in LOG.
        recall_at_k, nmi, f1  = eval.evaluate(opt.dataset, LOG, save=True, give_return=True, **eval_params)

        writer.add_scalar('Recall@1', recall_at_k[0], epoch)
        writer.add_scalar('NMI', nmi, epoch)
        writer.add_scalar('f1', f1, epoch)
        #Update the Metric Plot and save it.
        #LOG.update_info_plot()
    else:
        torch.cuda.empty_cache()

    ## Learning Rate Scheduling Step
    if opt.scheduler != 'none':
        scheduler.step()

    print('\n-----\n')

# Save the trained model
torch.save(inter_net.state_dict(), "/content/drive/MyDrive/CENG501/Confusezius/Trained_Models/InterNet2/InterNet_scripted.pt")
torch.save(intra_net.state_dict(), "/content/drive/MyDrive/CENG501/Confusezius/Trained_Models/IntraNet2/IntraNet_scripted.pt")

# Print the losses
print("\nMetric Loss List:", metric_loss_list)
#print( "%0.4f" % i for i in metric_loss_list)
print("\nRanking Loss List:", ranking_loss_list)
