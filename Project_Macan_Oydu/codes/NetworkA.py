import torch
import torch.nn as nn

class FirstConvolution(nn.Module):

    def __init__(self, input_channel, output_channel, expansion_ratio=4):
        super(FirstConvolution, self).__init__()
        self.middle_layer_size =  input_channel*expansion_ratio
        self.conv1 = nn.Sequential(nn.Conv2d(input_channel, self.middle_layer_size, (1,1)), nn.LeakyReLU())
        self.conv2 = nn.Sequential(nn.LeakyReLU(), nn.Conv2d(self.middle_layer_size, output_channel, (1,1)))
        self.conv_outer = nn.Conv2d(input_channel, output_channel, (1,1))
        self.depthConv = nn.Conv2d(self.middle_layer_size,self.middle_layer_size,(3,3),stride=1,padding=1)

    def forward(self, x):
        input = x.clone().detach()
        x = self.conv1(x)
        x = self.depthConv(x)
        x = self.conv2(x)
        x = x + self.conv_outer(input)
        return x

class DownConvolution(nn.Module):

    def __init__(self, input_channel, output_channel, expansion_ratio=4):
        super(DownConvolution, self).__init__()
        self.middle_layer_size =  input_channel*expansion_ratio
        self.conv1 = nn.Sequential(nn.Conv2d(input_channel, self.middle_layer_size, (1,1)), nn.LeakyReLU())
        self.conv2 = nn.Sequential(nn.LeakyReLU(), nn.Conv2d(self.middle_layer_size, output_channel, (1,1)))
        self.conv_outer = nn.Conv2d(input_channel, output_channel, (1,1))
        self.depthConv = nn.Conv2d(self.middle_layer_size,self.middle_layer_size,(3,3),stride=1,padding=1)
        self.maxpool = nn.MaxPool2d(2,2)

    def forward(self, x):
        x = self.maxpool(x)
        input = x.clone().detach() 
        x = self.conv1(x)
        x = self.depthConv(x)
        x = self.conv2(x)
        x = x + self.conv_outer(input)
        return x

class UpConvolution(nn.Module):

    def __init__(self, input_channel, output_channel, expansion_ratio=4):
        super(UpConvolution, self).__init__()
        self.middle_layer_size =  input_channel*expansion_ratio
        self.conv1 = nn.Sequential(nn.Conv2d(input_channel, self.middle_layer_size, (1,1)), nn.LeakyReLU())
        self.conv2 = nn.Sequential(nn.LeakyReLU(), nn.Conv2d(self.middle_layer_size, output_channel, (1,1)))
        self.conv_outer = nn.Conv2d(input_channel, output_channel, (1,1))
        self.depthConv = nn.Conv2d(self.middle_layer_size,self.middle_layer_size,(3,3),stride=1,padding=1)
        self.convtranspose = nn.ConvTranspose2d(output_channel, output_channel//2, (2,2), (2,2))

    def forward(self, x):
        input = x.clone().detach() 
        x = self.conv1(x)
        x = self.depthConv(x)
        x = self.conv2(x)
        x = x + self.conv_outer(input)
        x = self.convtranspose(x)
        return x

class LastConvolution(nn.Module):

    def __init__(self, input_channel, output_channel, out_size, expansion_ratio=4):
        super(LastConvolution, self).__init__()
        self.middle_layer_size =  input_channel*expansion_ratio
        self.conv1 = nn.Sequential(nn.Conv2d(input_channel, self.middle_layer_size, (1,1)), nn.LeakyReLU())
        self.conv2 = nn.Sequential(nn.LeakyReLU(), nn.Conv2d(self.middle_layer_size, output_channel, (1,1)))
        self.conv_outer = nn.Conv2d(input_channel, output_channel, (1,1))
        self.depthConv = nn.Conv2d(self.middle_layer_size,self.middle_layer_size,(3,3),stride=1,padding=1)
        self.conv3 = nn.Conv2d(output_channel, out_size, (1,1))

    def forward(self, x):
        input = x.clone().detach()
        x = self.conv1(x)
        x = self.depthConv(x)
        x = self.conv2(x)
        x = x + self.conv_outer(input)
        x = self.conv3(x)
        return x

class NetworkA(nn.Module):

    def __init__(self, in_size = 4, out_size = 3, expansion_ratio = 4):
        super(NetworkA, self).__init__()
        self.simpleConv = FirstConvolution(in_size, 32, expansion_ratio=expansion_ratio)
        self.downConv1 = DownConvolution(32, 64, expansion_ratio=expansion_ratio)
        self.downConv2 = DownConvolution(64, 128, expansion_ratio=expansion_ratio)
        self.downConv3 = DownConvolution(128, 256, expansion_ratio=expansion_ratio)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.bridge = UpConvolution(256, 512, expansion_ratio=expansion_ratio)
        self.upConv1 = UpConvolution(512, 256, expansion_ratio=expansion_ratio)
        self.upConv2 = UpConvolution(256, 128, expansion_ratio=expansion_ratio)
        self.upConv3 = UpConvolution(128, 64, expansion_ratio=expansion_ratio)
        self.lastConv = LastConvolution(64, 32, out_size, expansion_ratio=expansion_ratio)
    
    def forward(self, x):
        x1 = self.simpleConv(x)
        x2 = self.downConv1(x1)
        x3 = self.downConv2(x2)
        x4 = self.downConv3(x3)
        x5 = self.maxpool(x4)
        x6 = self.bridge(x5)
        x7 = self.upConv1(torch.cat((x4, x6), 1))
        x8 = self.upConv2(torch.cat((x3, x7), 1))
        x9 = self.upConv3(torch.cat((x2, x8), 1))
        A = self.lastConv(torch.cat((x1, x9), 1))
        return A

class NetworkA_iter(nn.Module):

    def __init__(self, expansion_ratio = 4, Eta = 0.01, Beta = 0.01, Lambda = 0.01):
        super(NetworkA_iter, self).__init__()
        self.Eta = nn.Parameter(torch.Tensor([Eta]), requires_grad=True)
        self.Beta = nn.Parameter(torch.Tensor([Beta]), requires_grad=True)
        self.Lambda = nn.Parameter(torch.Tensor([Lambda]), requires_grad=True)
        self.NetworkA = NetworkA(expansion_ratio=expansion_ratio)
    
    def forward(self,M,A,I_ns,I_s):
        input = torch.cat([A,M],dim=1)
        out = self.NetworkA(input)
        gradient_of_D = torch.mean((-I_ns/torch.pow((1+A),2))*(-I_s+I_ns/(1+A)),1,keepdim=True)
        gradient_of_g =torch.mean( A*torch.pow((1-M),2),1,keepdim=True)
        A = A-self.Eta*(gradient_of_D+self.Beta*gradient_of_g+self.Lambda*out)
        I_ns = I_s+A*I_s
        return I_ns, A