# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
from matplotlib import pyplot as plt
import numpy as np

from ..dynamically_grown_gan import DynamicallyGrownGAN
from .gan_trainer import GANTrainer
from .standard_configurations.dggan_config import _C
import torch

from itertools import product
import random

from ..metrics.inception_score import InceptionScore
from ..networks.constant_net import FeatureTransform



class DGGANTrainer(GANTrainer):
    r"""
    A trainer structure for the DCGAN and DCGAN product models
    """

    _defaultConfig = _C

    def getDefaultConfig(self):
        return DGGANTrainer._defaultConfig

    def __init__(self,
                 pathdb,
                 **kwargs):
        r"""
        Args:

            pathdb (string): path to the input dataset
            **kwargs:        other arguments specific to the GANTrainer class
        """

        GANTrainer.__init__(self, pathdb, **kwargs)

        self.lossProfile.append({"iter": [], "scale": 0})

        self.num_of_actions = 2
        self.action_list = []
        self.chosen_actions = []

    def initModel(self):
        self.model = DynamicallyGrownGAN(useGPU=self.useGPU,
                           **vars(self.modelConfig))


    def generate_action_list(self):
        filter_sizes = [3, 7]
        num_of_filters = [32, 64, 128, 256, 512, 1024]
        fade_in_block = 'fadein'
        gen_disc = ['g','d']

        action_list = list(product(num_of_filters, filter_sizes))
        action_list = list(product(gen_disc, action_list))
        action_list.append(fade_in_block)
        self.action_list = action_list

    def pick_k_from_actions(self, action_list, k):
        return random.choices(action_list, k=k)

    def choose_actions(self):
        chosen_actions = self.pick_k_from_actions(self.action_list, self.num_of_actions)
        return chosen_actions

    def decide_growth(self, action):
        if action[0] == 'g':
            self.model.netG.module.add_conv_generator_layer(action[1][0], action[1][1])
        elif action[0] == 'd':
            #self.model.netD.module.add_conv_discriminator_layer(action[1][0], action[1][1])
            pass
        else:
            pass
            #self.model.netG.module.generator_increase_resolution()
            #self.model.netD.module.discriminator_increase_resolution()


    def train(self):
        CURRENT_MODEL_PATH = "./models/trainer/model_storage/current_model.pt"
        NEW_MODEL_BASE_PATH = "./models/trainer/model_storage/next_model_"

        max_actions = 5
        n_actions = 0

        self.generate_action_list()

        while n_actions < max_actions:
            print("start training")
            score_list = []

            torch.save(self.model, CURRENT_MODEL_PATH)

            chosen_actions = self.choose_actions()
            for i in range(len(chosen_actions)):
                self.model = torch.load(CURRENT_MODEL_PATH)

                self.decide_growth(chosen_actions[i])

                #Activate train mode
                self.model.netD.train(True)
                self.model.netG.train(True)

                #start training
                shift = 0
                if self.startIter >0:
                    shift+= self.startIter

                if self.checkPointDir is not None:
                    pathBaseConfig = os.path.join(self.checkPointDir, self.modelLabel
                                                  + "_train_config.json")
                    self.saveBaseConfig(pathBaseConfig)

                maxShift = int(self.modelConfig.nEpoch * len(self.getDBLoader(0)))

                for epoch in range(10):#self.modelConfig.nEpoch
                    self.generateImages('pre')
                    print("epoch: ", epoch)
                    dbLoader = self.getDBLoader(0)
                    self.trainOnEpoch(dbLoader, 0, shiftIter=shift)
                    self.generateImages(epoch)
                    shift += len(dbLoader)

                    if shift > maxShift:
                        break
                #end training

                #start evaluation
                score = self.CalculateInceptionScore()
                score_list.append(score)
                torch.save(self.model,NEW_MODEL_BASE_PATH + str(i) + ".pt")
                #end evaluation

            max_score_index = score_list.index(max(score_list))
            self.model = torch.load(NEW_MODEL_BASE_PATH + str(max_score_index) + ".pt")
            n_actions += 1

    def CalculateInceptionScore(self):
        self.model.netD.train(False)
        self.model.netG.train(False)

        scoreMaker = InceptionScore(self.model)

        batchSize = 1
        nBatch = 100
        refMean = [2*p - 1 for p in[0.485, 0.456, 0.406]]
        refSTD = [2*p for p in [0.229, 0.224, 0.225]]
        imgTransform = FeatureTransform(mean=refMean,
                                        std=refSTD,
                                        size=16).cuda()

        print("Computing the inception score...")
        for index in range(nBatch):
            inputFake = self.model.test(self.model.buildNoiseData(batchSize)[0],
                                   toCPU=False, getAvG=True)
            scoreMaker.updateWithMiniBatch(imgTransform(inputFake))
        return scoreMaker.getScore()


    def initializeWithPretrainNetworks(self,
                                       pathD,
                                       pathGShape,
                                       pathGTexture,
                                       finetune=True):
        r"""
        Initialize a product gan by loading 3 pretrained networks

        Args:

            pathD (string): Path to the .pt file where the DCGAN discrimator is saved
            pathGShape (string): Path to .pt file where the DCGAN shape generator
                                 is saved
            pathGTexture (string): Path to .pt file where the DCGAN texture generator
                                   is saved

            finetune (bool): set to True to reinitialize the first layer of the
                             generator and the last layer of the discriminator
        """

        if not self.modelConfig.productGan:
            raise ValueError("Only product gan can be cross-initialized")

        self.model.loadG(pathGShape, pathGTexture, resetFormatLayer=finetune)
        self.model.load(pathD, loadG=False, loadD=True,
                        loadConfig=False, finetuning=True)


    def generateImages(self, epoch):
        r"""
        Generate images with the current model

        Args:

            n (int): number of images to generate

        Returns:

            list of generated images
        """

        inputRandom, randomLabels = self.model.buildNoiseData(16)

        ### Feed a random vector to the model

        images = self.model.test(inputRandom,
                getAvG=True,
                toCPU=True)

        # save images
        fig = plt.figure(figsize=(32, 32))
        out = self.model.test(inputRandom, getAvG=True, toCPU=True)
        for i in range(4):
            for j in range(4):
                fig.add_subplot(4, 4, i * 4 + j + 1)
                plt.imshow(np.transpose(out[i * 3 + j], (1, 2, 0)), interpolation='nearest')
                # Show results on a 2x2 grid
                # plt.imshow(np.transpose(out[0], (1, 2, 0)), interpolation='nearest')
        plt.savefig(f"images/epoch{epoch}.png")
        plt.close(fig)
