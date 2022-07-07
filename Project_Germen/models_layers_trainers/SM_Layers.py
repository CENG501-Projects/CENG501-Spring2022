import torch
import torch.nn as nn
import numpy as np
import random
import math


def greedy_selection(scores):
    # dim = -1 so it can work both on CONV and FC
    indexes = torch.argmax(scores, dim=-1,keepdim = True)
    return indexes
    
SM = nn.Softmax(-1)

def probabilistic_sampling(scores):
    #e_scores = torch.exp(scores)
    #e_scores = e_scores / e_scores.sum(dim=-1).unsqueeze(-1)
    
    e_scores = SM(scores)

    # Need to reshape since sampling doesn't work dimension wise
    e_scores_shape = list(e_scores.shape)
    k = e_scores_shape.pop()
    e_scores = torch.reshape(e_scores,(np.prod(e_scores_shape),k))

    indexes = torch.multinomial(e_scores, 1)
    indexes = torch.reshape(indexes, e_scores_shape)
    indexes = indexes.unsqueeze(-1)

    return indexes
    pass

class Conv2d_SM(nn.Module):
    def __init__(self, in_channels, kernel_size,padding, out_channels, K, score_selection = "GS"):
        super(Conv2d_SM, self).__init__()
        
        if score_selection == "GS":
            self.score_function = greedy_selection
        elif score_selection == "PS":
            self.score_function = probabilistic_sampling
        else:
            raise("score_selection is not proper!")

        # Creating classic conv
        self.conv = nn.Conv2d(in_channels=in_channels,
                                kernel_size=kernel_size,
                                padding=padding,
                                out_channels=out_channels)

        # Setting discrete weights
        w_shape = list(self.conv.weight.shape)

        # Calculating glorot standart deviation
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(self.conv.weight.data) # THIS IS IMPORTANT, extract std BEFORE K
        std =  math.sqrt(2.0 / float(fan_in + fan_out)) # Std of Glorot Normal Dst
        a = std * math.sqrt(3) # Bound from standard deviation for uniform Glorot uniform Dst

        w_shape.append(K)
        self.all_weights = torch.empty(w_shape).to("cuda")
        
        
        self.all_weights = torch.nn.init.uniform_(self.all_weights,-std,std)

        # Setting initial scores
        self.weight_scores = torch.empty(w_shape).to("cuda")
            
        

        torch.nn.init.uniform_(self.weight_scores, a=0.0, b=std)

        
        
    def update_scores(self, lr):
        # Update scores according to gradients on "self.X"
        grads_lr = -1 * lr * torch.mul(self.conv.weight.grad.data, self.conv.weight.data)
        self.weight_scores.scatter_(-1, self.indexes, grads_lr.unsqueeze_(-1), reduce="add")
        
        pass

    def set_weights(self):
        # Find indexes of the "will be" chosen weights
        self.indexes = self.score_function(self.weight_scores).to("cuda")


        # Set the weights according to indexes #PROBLEM
        self.conv.weight.data = torch.gather(self.all_weights, -1 , self.indexes).squeeze_().to("cuda")

        pass

    def forward(self,x):
        
        
        self.set_weights()
        
        
        return self.conv(x)

class Linear_SM(nn.Module):
    def __init__(self, in_features, out_features, K, score_selection = "GS"):
        super(Linear_SM, self).__init__()
        
        if score_selection == "GS":
            self.score_function = greedy_selection
        elif score_selection == "PS":
            self.score_function = probabilistic_sampling
        else:
            raise("score_selection is not proper!")

        # Creating classic linear
        self.fc1 = nn.Linear(in_features=in_features,
                                out_features=out_features)

        # Setting discrete weights
        w_shape = list(self.fc1.weight.shape)

        # Calculating glorot standart deviation
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(self.fc1.weight.data)
        std =  math.sqrt(2.0 / float(fan_in + fan_out)) # Std of Glorot Normal Dst
        a = std * math.sqrt(3) # Bound from standard deviation for uniform Glorot uniform Dst

        w_shape.append(K)
        self.all_weights = torch.empty(w_shape).to("cuda")
        
        
        self.all_weights = torch.nn.init.uniform_(self.all_weights,-std,std)

        # Setting initial scores
        self.weight_scores = torch.empty(w_shape).to("cuda")
            
        

        torch.nn.init.uniform_(self.weight_scores, a=0.0, b=std*0.1)

        
    def update_scores(self, lr):
        # Update scores according to gradients on "self.X"
        grads_lr = -1 * lr * torch.mul(self.fc1.weight.grad.data, self.fc1.weight.data)
        self.weight_scores.scatter_(-1, self.indexes, grads_lr.unsqueeze_(-1), reduce="add")
        
        pass

    def set_weights(self):
        # Find indexes of the "will be" chosen weights
        self.indexes = self.score_function(self.weight_scores).to("cuda")

        # Set the weights according to indexes
        self.fc1.weight.data = torch.gather(self.all_weights, -1 , self.indexes).squeeze_().to("cuda")

        pass

    def forward(self,x):
        

        self.set_weights()
        
        # Gradients will be captured on the following tensor
        X = self.fc1(x)
        
        
        return X
