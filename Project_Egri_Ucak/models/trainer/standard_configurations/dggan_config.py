# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from ...utils.config import BaseConfig

# Default configuration for ProgressiveGANTrainer
_C = BaseConfig()

############################################################

_C.startResolution = 16

# Depth of a convolutional layer for each scale
_C.depth = 3

# Mini batch size
_C.miniBatchSize = 16

# Dimension of the latent vector
_C.dimLatentVector = 128

# Dimension of the output image
_C.dimOutput = 3

# Dimension of the generator
_C.dimG = 8

# Dimension of the discrimator
_C.dimD = 8

# Loss mode
_C.lossMode = 'Logistic'
# add FID score
# https://github.com/mseitzer/pytorch-fid
# https://www.kaggle.com/code/ibtesama/gan-in-pytorch-with-fid/notebook

# Gradient penalty coefficient (WGANGP)
_C.lambdaGP = 0.

# Noise standard deviation in case of instance noise (0 <=> no Instance noise)
_C.sigmaNoise = 0.

# Weight penalty on |D(x)|^2
_C.epsilonD = 0.

# Base learning rate
_C.baseLearningRate = 0.0002

# In case of AC GAN, weight on the classification loss (per scale)
_C.weightConditionG = 0.0
_C.weightConditionD = 0.0

# Activate GDPP loss ?
_C.GDPP = False

# Number of epochs
_C.nEpoch = 4

# Do not modify. Field used to save the attribute dictionnary for labelled
# datasets
_C.attribKeysOrder = None
