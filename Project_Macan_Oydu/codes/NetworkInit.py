import torch
import torch.nn as nn

class FirstConvolution_init(nn.Module):

    def __init__(self, input_channel, output_channel, expansion_ratio=6):
        super(FirstConvolution_init, self).__init__()
        self.middle_layer_size =  input_channel*expansion_ratio
        self.conv1 = nn.Sequential(nn.Conv2d(input_channel, self.middle_layer_size, (1,1)), nn.LeakyReLU())
        self.conv2 = nn.Sequential(nn.LeakyReLU(), nn.Conv2d(self.middle_layer_size, output_channel, (1,1)))
        self.conv_outer = nn.Conv2d(input_channel, output_channel, (1,1))
        self.depthConv = nn.Conv2d(self.middle_layer_size,self.middle_layer_size,(3,3),stride=1,padding=1)

    def forward(self, x):
        input = x.clone().detach()
        input = self.conv_outer(input)
        x = self.conv1(x)
        x = self.depthConv(x)
        x = self.conv2(x)
        x = x*input
        x = x + input
        return x

class DownConvolution_init(nn.Module):

    def __init__(self, input_channel, output_channel, expansion_ratio=6):
        super(DownConvolution_init, self).__init__()
        self.middle_layer_size =  input_channel*expansion_ratio
        self.conv1 = nn.Sequential(nn.Conv2d(input_channel, self.middle_layer_size, (1,1)), nn.LeakyReLU())
        self.conv2 = nn.Sequential(nn.LeakyReLU(), nn.Conv2d(self.middle_layer_size, output_channel, (1,1)))
        self.conv_outer = nn.Conv2d(input_channel, output_channel, (1,1))
        self.depthConv = nn.Conv2d(self.middle_layer_size,self.middle_layer_size,(3,3),stride=1,padding=1)
        self.maxpool = nn.MaxPool2d(2,2)

    def forward(self, x):
        x = self.maxpool(x)
        input = x.clone().detach()
        input = self.conv_outer(input)
        x = self.conv1(x)
        x = self.depthConv(x)
        x = self.conv2(x)
        x = x*input
        x = x + input
        return x

class UpConvolution_init(nn.Module):

    def __init__(self, input_channel, output_channel, expansion_ratio=6):
        super(UpConvolution_init, self).__init__()
        self.middle_layer_size =  input_channel*expansion_ratio
        self.conv1 = nn.Sequential(nn.Conv2d(input_channel, self.middle_layer_size, (1,1)), nn.LeakyReLU())
        self.conv2 = nn.Sequential(nn.LeakyReLU(), nn.Conv2d(self.middle_layer_size, output_channel, (1,1)))
        self.conv_outer = nn.Conv2d(input_channel, output_channel, (1,1))
        self.depthConv = nn.Conv2d(self.middle_layer_size,self.middle_layer_size,(3,3),stride=1,padding=1)
        self.convtranspose = nn.ConvTranspose2d(output_channel, output_channel//2, (2,2), (2,2))

    def forward(self, x):
        input = x.clone().detach()
        input = self.conv_outer(input)
        x = self.conv1(x)
        x = self.depthConv(x)
        x = self.conv2(x)
        x = x*input
        x = x + input
        x = self.convtranspose(x)
        return x

class LastConvolution_init(nn.Module):

    def __init__(self, input_channel, output_channel, out_size, expansion_ratio=6):
        super(LastConvolution_init, self).__init__()
        self.middle_layer_size =  input_channel*expansion_ratio
        self.conv1 = nn.Sequential(nn.Conv2d(input_channel, self.middle_layer_size, (1,1)), nn.LeakyReLU())
        self.conv2 = nn.Sequential(nn.LeakyReLU(), nn.Conv2d(self.middle_layer_size, output_channel, (1,1)))
        self.conv_outer = nn.Conv2d(input_channel, output_channel, (1,1))
        self.depthConv = nn.Conv2d(self.middle_layer_size,self.middle_layer_size,(3,3),stride=1,padding=1)
        self.conv1d = nn.Conv2d(output_channel, out_size, (1,1))

    def forward(self, x):
        input = x.clone().detach()
        input = self.conv_outer(input)
        x = self.conv1(x)
        x = self.depthConv(x)
        x = self.conv2(x)
        x = x*input
        x = x + input
        x = self.conv1d(x)
        return x

class NetworkInit(nn.Module):

    def __init__(self, in_size = 4, out_size = 3, expansion_ratio = 6):
        super(NetworkInit, self).__init__()
        self.simpleConv = FirstConvolution_init(in_size, 32, expansion_ratio=expansion_ratio)
        self.downConv1 = DownConvolution_init(32, 64, expansion_ratio=expansion_ratio)
        self.downConv2 = DownConvolution_init(64, 128, expansion_ratio=expansion_ratio)
        self.downConv3 = DownConvolution_init(128, 256, expansion_ratio=expansion_ratio)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.bridge = UpConvolution_init(256, 512, expansion_ratio=expansion_ratio)
        self.upConv1 = UpConvolution_init(512, 256, expansion_ratio=expansion_ratio)
        self.upConv2 = UpConvolution_init(256, 128, expansion_ratio=expansion_ratio)
        self.upConv3 = UpConvolution_init(128, 64, expansion_ratio=expansion_ratio)
        self.lastConv = LastConvolution_init(64, 32, out_size, expansion_ratio=expansion_ratio)
    
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
        A_0 = self.lastConv(torch.cat((x1, x9), 1))
        I_ns_0 = x[:,:3,:,:]+A_0*x[:,:3,:,:]
        return I_ns_0, A_0

if __name__ == "__main__":
    input = torch.randn(1, 3, 256, 256)
    mask = torch.randn(1, 1, 256, 256)
    model = NetworkInit()

    weight_map,output = model(torch.cat([input,mask],dim=1))
    print('-'*50)
    print(output.shape)
    print('#generator parameters:', sum(param.numel() for param in model.parameters()))

