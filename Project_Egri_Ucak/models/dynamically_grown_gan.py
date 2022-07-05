# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
print("dynamically_grown_gan.py")
import torch.optim as optim

from .base_GAN import BaseGAN
from .utils.config import BaseConfig
from .networks.dynamically_grown_net import GNet, DNet


class DynamicallyGrownGAN(BaseGAN):
    def __init__(self,
                 startResolution=32,
                 dimLatentVector=64,
                 dimG=64,
                 dimD=64,
                 depth=3,
                 **kwargs):
        if not 'config' in vars(self):
            self.config = BaseConfig()

        self.config.dimG = dimG
        self.config.dimD = dimD
        self.config.depth = depth
        self.config.startResolution = startResolution

        self.current_size = startResolution

        BaseGAN.__init__(self, dimLatentVector, **kwargs)

    def getNetG(self):

        gnet = GNet(self.config.latentVectorDim,
                    self.config.dimOutput,
                    self.config.dimG,
                    startResolution=self.config.startResolution,
                    depthModel=self.config.depth,
                    generationActivation=self.lossCriterion.generationActivation)

        return gnet

    def getNetD(self):
        print("getNetD: ", self.lossCriterion.sizeDecisionLayer, self.config.categoryVectorDim)

        dnet = DNet(self.config.dimOutput,
                    self.config.dimD,
                    self.lossCriterion.sizeDecisionLayer
                    + self.config.categoryVectorDim,
                    startResolution=self.config.startResolution,
                    depthModel=self.config.depth)
        return dnet

    def getOptimizerD(self):
        return optim.Adam(filter(lambda p: p.requires_grad, self.netD.parameters()),
                          betas=[0, 0.99], lr=self.config.learningRate)

    def getOptimizerG(self):
        return optim.Adam(filter(lambda p: p.requires_grad, self.netG.parameters()),
                          betas=[0, 0.99], lr=self.config.learningRate)
        
    def add_conv_generator_layer(self, filter_count, filter_size):
        self.netG.add_conv_generator_layer(filter_count, filter_size)

    def add_conv_discriminator_layer(self, filter_count, filter_size):
        self.netD.add_conv_discriminator_layer(filter_count, filter_size)

    def increase_resolution(self):
        self.netG.generator_increase_resolution()
        self.netD.discriminator_increase_resolution()
        self.current_size *= 2

    def getSize(self):
        size = self.current_size
        print("getSize: ", size)
        return (size, size)
