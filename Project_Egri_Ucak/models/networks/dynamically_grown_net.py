# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from collections import OrderedDict
from pprint import pprint
import torch.nn as nn

from models.networks.custom_layers import EqualizedLinear


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class GNet(nn.Module):
    def __init__(self,
                 dimLatentVector,
                 dimOutput,
                 dimModelG,
                 startResolution,
                 depthModel=3,
                 generationActivation=nn.Tanh()):
        super(GNet, self).__init__()

        self.depthModel = depthModel
        self.current_resolution = startResolution
        self.refDim = dimModelG
        # self.refDim = startResolution
        self.dimOutput = dimOutput

        self.initFormatLayer(dimLatentVector)

        currDepth = int(dimModelG * (2**depthModel))
        print("depthModel: ", depthModel)
        print("dimModelG: ", dimModelG)
        print("currDepth: ", currDepth)

        self.layers = nn.ModuleList().cuda()
        self.child_layers = nn.ModuleList().cuda()

        sequence = OrderedDict([])
        nextDepth = int(currDepth / 2)

        sequence["convTranspose1"] = nn.ConvTranspose2d(
            currDepth, nextDepth, 4, 2, 1, bias=False).cuda()
        sequence["relu1"] = nn.ReLU(True).cuda()

        self.currDepth = nextDepth
        print("self.currDepth: ", self.currDepth)
        print("self.dimModelG: ", dimModelG)
        self.out_layer = nn.Sequential(
            nn.ConvTranspose2d(nextDepth, dimOutput, 3, 1, 1, bias=False).cuda(),
            nn.Sigmoid().cuda()
        ).cuda()

        # sequence["outlayer"] = nn.ConvTranspose2d(
        #     dimModelG, dimOutput, 1, 1, 0, bias=False)

        self.outputAcctivation = generationActivation

        main = nn.Sequential(sequence)
        # self.sequence = sequence
        main.apply(weights_init)
        self.layers.append(main)

    def initFormatLayer(self, dimLatentVector):
        currDepth = int(self.refDim * (2**self.depthModel))
        print("currDepth: ", currDepth)
        print("self.refDim: ", self.refDim)
        print("self.depthModel: ", self.depthModel)
        self.formatLayer = nn.ConvTranspose2d(
            dimLatentVector, currDepth, self.current_resolution // 2, 1, 0, bias=False).cuda()

    def forward(self, input):
        # print("Generator input: ", input.size())

        x = input.view(-1, input.size(1), 1, 1)
        x = self.formatLayer(x)
        # x = self.main(x)
        # print(x.size())
        for layer in self.layers:
            x = layer(x)
            # print(x.size())
        # print(x.size())
        for layer in self.child_layers:
            x = layer(x)
        # print("Size before out layer: ", x.size())
        # print(self.out_layer)

        x = self.out_layer(x)

        # print("Size after out layer: ", x.size())
        if self.outputAcctivation is None:
            return x
        return self.outputAcctivation(x)

    def add_conv_generator_layer(self, filter_count, filter_size):
        print("Adding new layer ", filter_count)
        padding = self.calculate_padding_to_match_resolution(filter_size)
        print("Padding: ", padding)
        self.child_layers.append(
            nn.ConvTranspose2d(self.currDepth, filter_count,
                                filter_size, 1, padding, bias=False).cuda()
        )
        self.child_layers.append(nn.ReLU(True).cuda())
        self.child_layers.cuda()
        self.depthModel += 1
        self.currDepth = filter_count
        self.out_layer = nn.ConvTranspose2d(
            filter_count, self.dimOutput, 1, 1, 0, bias=False).cuda()

    def calculate_padding_to_match_resolution(self, filter_size):
        padding = (filter_size - 1) // 2
        return padding

    def generator_increase_resolution(self):
        self.current_resolution *= 2
        self.child_layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(self.currDepth, self.currDepth,
                                    1, 2, 0, output_padding=1, bias=False).cuda(),
                nn.Sigmoid().cuda()
            )
        )
        print("Generator resolution: ", self.current_resolution)

    def merge_layers(self):
        self.layers.append(nn.Sequential(self.child_layers).cuda())
        self.child_layers = nn.ModuleList().cuda()


class DNet(nn.Module):
    def __init__(self,
                 dimInput,
                 dimModelD,
                 sizeDecisionLayer,
                 startResolution,
                 depthModel=3):
        super(DNet, self).__init__()
        self.dimInput = dimInput

        currDepth = dimModelD
        sequence = OrderedDict([])
        self.current_resolution = startResolution

        self.layers = nn.ModuleList().cuda()
        self.child_layers = nn.ModuleList().cuda()

        # input is (nc) x 2**(depthModel + 3) x 2**(depthModel + 3)
        # self.conv1 = nn.Conv2d(
        #     dimInput, currDepth, 3, 1, 1, bias=False)
        # self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.first_conv_layer = nn.Sequential(
            nn.Conv2d(dimInput, currDepth, 3, 1, 1, bias=True).cuda(),
            nn.LeakyReLU(0.2, inplace=True).cuda()
        )
        self.first_conv_layer.apply(weights_init)
        sequence["convTranspose" +
                 str(depthModel)] = nn.Conv2d(dimInput, currDepth,
                                              3, 1, 1, bias=False).cuda()
        sequence["relu" + str(depthModel)] = nn.LeakyReLU(0.2, inplace=True).cuda()

        # for i in range(depthModel):

        #     index = depthModel - i - 1
        #     nextDepth = currDepth * 2

        #     # state size.
        #     # (currDepth) x 2**(depthModel + 2 -i) x 2**(depthModel + 2 -i)
        #     sequence["convTranspose" +
        #              str(index)] = nn.Conv2d(currDepth, nextDepth,
        #                                      3, 1, 1, bias=False)
        #     sequence["batchNorm" + str(index)] = nn.BatchNorm2d(nextDepth)
        #     sequence["relu" + str(index)] = nn.LeakyReLU(0.2, inplace=True)

        #     currDepth = nextDepth

        self.dimFeatureMap = currDepth

        main = nn.Sequential(sequence).cuda()
        main.apply(weights_init)
        self.layers.append(main)

        # self.initDecisionLayer(sizeDecisionLayer)
        # self.outLayer = nn.Sequential(
        #     nn.Conv2d(currDepth, 1, 3, 1, 1, bias=False),
        #     nn.Flatten(),
        #     nn.Linear(self.current_resolution ** 2, sizeDecisionLayer),
        #     nn.Softmax(),
        # )
        # self.outLayer.apply(weights_init)
        self.sizeDecisionLayer = sizeDecisionLayer
        self.outLayer = self.get_out_layer()

    def get_out_layer(self):
        outLayer = nn.Sequential(
            nn.Conv2d(self.dimFeatureMap, 1, 3, 1, 1, bias=False).cuda(),
            nn.Flatten().cuda(),
            nn.Linear(self.current_resolution ** 2, self.sizeDecisionLayer).cuda(),
            nn.Softmax().cuda(),
        )
        outLayer.apply(weights_init)
        return outLayer

    def initDecisionLayer(self, sizeDecisionLayer):
        print("Initializing decision layer: ", sizeDecisionLayer, self.dimFeatureMap)
        self.decisionLayer = nn.Sequential(
            nn.Conv2d(
                self.dimFeatureMap, sizeDecisionLayer, 3, 1, 0, bias=False).cuda(),
            nn.Flatten().cuda(),
            nn.Linear(11 * 3 * 3, sizeDecisionLayer).cuda(),
            nn.Softmax().cuda(),
        )
        self.decisionLayer.apply(weights_init)
        self.sizeDecisionLayer = sizeDecisionLayer
        # self.decisionLayer = nn.Linear(16, sizeDecisionLayer)

    def forward(self, input, getFeature = False):
        # x = self.main(input)
        # x = self.first_conv_layer(input)
        x = input

        for layer in self.child_layers:
            x = layer(x)

        for layer in self.layers:
            x = layer(x)
        # print("DNet output: ", x.size())
        # x = self.conv1(input)
        # x = self.relu1(x)
        # print(input.size(), x.size())

        if getFeature:
            # return self.decisionLayer(x).view(-1, self.sizeDecisionLayer), \
            #        x.view(-1, self.dimFeatureMap * 16)
            return self.outLayer(x).view(-1, self.sizeDecisionLayer), \
                   x.view(-1, self.dimFeatureMap * 16)

        # print("discriminator: ", x.size())
        # x = self.decisionLayer(x)
        x = self.outLayer(x)
        # print("decision layer: ", x.size())
        # x = x.view(16, -1)
        # x = x.view(16, self.sizeDecisionLayer)
        # print("discriminator view: ", x.size())
        # print(x.view(-1, self.sizeDecisionLayer)[0])
        return x.view(-1, self.sizeDecisionLayer)
        return x

    def init_format_layer(self):
        layer = nn.Conv2d(self.dimInput, self.dimInput,
                            2, 2, 0, bias=False).cuda()
        return layer

    def add_conv_discriminator_layer(self, filter_count, filter_size):
        pass
        padding = self.calculate_padding_to_match_resolution(filter_size)
        self.child_layers.insert(0,
            nn.Conv2d(self.dimInput, filter_count, filter_size, 1, padding, bias=True).cuda()
        )

    def discriminator_increase_resolution(self):
        pass
        # self.current_resolution *= 2
        # self.child_layers.insert(0, nn.Conv2d(self.dimInput, self.dimInput,
        #                     2, 2, 0, bias=False).cuda())

        # self.outLayer = self.get_out_layer()
        '''
        padding = self.calculate_padding_to_match_resolution(filter_size)
        print("Padding: ", padding)
        self.child_layers.append(
            nn.ConvTranspose2d(self.currDepth, filter_count,
                                filter_size, 1, padding, bias=False)
        )
        self.child_layers.append(nn.ReLU(True))
        '''

    def calculate_padding_to_match_resolution(self, filter_size):
        padding = (filter_size - 1) // 2
        return padding
