import torch
import model
import redundancy
import test
from graph import *

def pruneLayer(network, layer_idx, count): # layer_idx: int, count: int
    network = network.cpu()
    dim = 0         # corresponding dim of FILTER WEIGHT to prune [out_ch, in_ch, k1, k2]

    for i in range(len(network.encoder)):
        layer = network.encoder[i]

        if isinstance(layer, torch.nn.Conv2d):
            if dim == 1:    # Prune filters' DEPTH
                new_= getNewConv(layer, dim, channel_index)
                network.encoder[i] = new_
                dim ^= 1

            if i == layer_idx:
                channel_index = getFilterIdx(layer.weight.data, count)     # filter indices to be pruned [0, 25, 55]
                new_ = getNewConv(layer, dim, channel_index)
                network.encoder[i] = new_
                dim ^= 1

    # update to check last conv layer pruned
    if layer_idx == 28: # The last conv layer
        network.classifier[0] = getNewLinear(network.classifier[0], channel_index)

    return network

def getFilterIdx(kernel, num_elimination):
    sum_of_kernel = torch.sum(torch.abs(kernel.view(kernel.size(0), -1)), dim=1)
    _, args = torch.sort(sum_of_kernel)

    return args[:num_elimination].tolist()

def removeIdx(tensor, dim, index):
    if tensor.is_cuda:
        tensor = tensor.cpu()

    size_ = list(tensor.size()) # (512, 256, 7, 7)
    new_size = tensor.size(dim) - len(index)  # 512 - 3
    size_[dim] = new_size
    new_size = size_            # (509, 256, 7, 7)

    select_index = list(set(range(tensor.size(dim))) - set(index))
    new_tensor = torch.index_select(tensor, dim, torch.tensor(select_index))

    return new_tensor   # (509, 256, 7, 7)

def getNewConv(conv, dim, channel_index):
    if dim == 0:
        new_conv = torch.nn.Conv2d(in_channels=conv.in_channels,
                                   out_channels=int(conv.out_channels - len(channel_index)),
                                   kernel_size=conv.kernel_size,
                                   stride=conv.stride, padding=conv.padding, dilation=conv.dilation)

        new_conv.weight.data = removeIdx(conv.weight.data, dim, channel_index)
        new_conv.bias.data = removeIdx(conv.bias.data, dim, channel_index)

        return new_conv

    elif dim == 1:
        new_conv = torch.nn.Conv2d(in_channels=int(conv.in_channels - len(channel_index)),
                                   out_channels=conv.out_channels,
                                   kernel_size=conv.kernel_size,
                                   stride=conv.stride, padding=conv.padding, dilation=conv.dilation)
        
        new_weight = removeIdx(conv.weight.data, dim, channel_index)
        new_conv.weight.data = new_weight
        new_conv.bias.data = conv.bias.data

        return new_conv

def getNewLinear(linear, channel_index):
    new_linear = torch.nn.Linear(in_features=int(linear.in_features - len(channel_index)),
                                out_features=linear.out_features,
                                bias=linear.bias is not None)
    new_linear.weight.data = removeIdx(linear.weight.data, 1, channel_index)
    new_linear.bias.data = linear.bias.data

    return new_linear

if __name__ == '__main__':
    model = model.VGG16()
    model.load_state_dict(torch.load('checkpoints/checkpoint.pt'))
    model.eval()

    gamma = 0.023
    w1 = 0.35
    layers = list()

    with torch.no_grad():
        print('Creating graphs for each conv. layer')

        for i in range(len(model.encoder)):
            layer = model.encoder[i]

            if(isinstance(layer, torch.nn.Conv2d)):
                graph = constructGraph(layer, gamma)
                red = redundancy.redundancy(graph, w1)
                print(f"\tLayer index: {i}, redundancy: {red}")

                layers.append({'idx': i,
                                'graph': graph,
                                'redVal': red})

        print('Pruning starts...')

        for j in range(10):
            maxRed = 1
            maxLayer = 0
            maxIndex = 0

            for i in range(len(layers)):
                if(layers[i].get('redVal') >= maxRed):
                    maxRed = layers[i].get('redVal')
                    maxLayer = layers[i]['idx']
                    maxIndex = i

            if maxRed == 1:
                break

            pruneNumber = (int) (np.sqrt(maxRed) * 2)
            print(f'\tRemoving {pruneNumber} filters from layer {maxLayer}')

            model = pruneLayer(model, maxLayer, pruneNumber)

            # Reconstruct the graph for the pruned layer
            graph = constructGraph(model.encoder[maxLayer], gamma)
            red = redundancy.redundancy(graph, w1)
            layers[maxIndex]['redVal'] = red
            layers[maxIndex]['graph'] = graph

            test.test(model)