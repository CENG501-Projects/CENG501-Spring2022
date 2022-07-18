import torch
from torch import nn
from .data_structure import *
import torchvision

torch.set_printoptions(profile="full")
import numpy as np
import random


class LSTM(nn.Module):
    """ LSTM Network class """

    def __init__(self, input_dim, hidden_dim, out_dim, layer_dim, device='cuda', dropout=(False, 0.3)):
        """
        :param input_dim:  the size of the input
        :param hidden_dim: the number of neurons in the hidden state
        :param out_dim: the size of the output
        :param layer_dim: the number of hidden layers in lstm
        :param device: model device
        :param dropout: boolean value that specified that dropout regularizer used or not
        """
        super().__init__()
        random.seed(501)
        np.random.seed(501)
        torch.manual_seed(501)

        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.device = device

        # Embedding layer to embed the words
        self.embedding = nn.Embedding(input_dim, hidden_dim)

        # Dropout layer
        self.dropout = dropout
        if self.dropout[0]:
            self.dropout_lyr_1 = nn.Dropout(self.dropout[1])
            self.dropout_lyr_2 = nn.Dropout(self.dropout[1])

        # LSTM layer
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim,
                            num_layers=layer_dim, batch_first=True)

        # Linear layers
        self.hidden_layer_1 = nn.Linear(self.hidden_dim, hidden_dim)
        self.hidden_layer_2 = nn.Linear(self.hidden_dim, out_dim)

        # Activation Layer
        self.activation_1 = nn.ReLU()

    def forward(self, x):
        """
        LSTM forward function
        :param x: input data
        :return: probability of classes
        """
        # Get batch size and init hidden layers
        batch_size = x.shape[0]
        hidden = self.init_hidden(batch_size)

        # Input & LSTM Layer pass
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out[:, -1]

        # Use dropout if dropout key specified as True
        if self.dropout[0]:
            lstm_out = self.dropout_lyr_1(lstm_out)

        # Hidden & Output Layer pass
        lstm_out = self.activation_1(self.hidden_layer_1(lstm_out))

        # Use dropout if dropout key specified as True
        if self.dropout[0]:
            lstm_out = self.dropout_lyr_2(lstm_out)

        tag_scores = self.hidden_layer_2(lstm_out)

        # Get probability of classes
        probs = nn.functional.log_softmax(tag_scores, dim=1)
        return probs

    def init_hidden(self, batch_dim):
        """
        Initialize hidden state
        :param batch_dim: dimension of batch
        :return:
        """
        h0 = torch.zeros((self.layer_dim, batch_dim, self.hidden_dim)).to(self.device)
        c0 = torch.zeros((self.layer_dim, batch_dim, self.hidden_dim)).to(self.device)
        hidden = (h0, c0)
        return hidden


class MLP(nn.Module):
    """
        Multilayer Perceptron Class
    """

    def __init__(self, mlp_type, input_dim, hidden_dim, output_dim, dropout=(False, 0.3)):
        """
        :param mlp_type: The dataset type that is used (Polarity or MNIST)
        :param input_dim:  the size of the input
        :param hidden_dim: the number of neurons in the hidden state
        :param output_dim: the size of the output
        :param dropout: boolean value that specified that dropout regularizer used or not
        """
        super().__init__()
        random.seed(501)
        np.random.seed(501)
        torch.manual_seed(501)

        self.mlp_type = mlp_type

        # Used Embedding Layer if Polarity Dataset is used
        if mlp_type == MLPType.MNIST:
            self.input_layer = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_layer = nn.Embedding(input_dim, hidden_dim)

        # 2 Hidden layers
        self.hidden_layer_1 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_layer_2 = nn.Linear(hidden_dim, hidden_dim)

        # Activation Layers
        self.activation_1 = nn.ReLU()
        self.activation_2 = nn.ReLU()

        # Output Layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Dropout layer
        self.dropout = dropout
        if self.dropout[0]:
            self.dropout_lyr_1 = nn.Dropout(self.dropout[1])
            self.dropout_lyr_2 = nn.Dropout(self.dropout[1])

    def forward(self, x):
        """
        MLP forward function
        :param x: input data
        :return: probability of classes
        """
        # Remove the dimensions of input of size 1.
        x = torch.squeeze(x)

        # Input & Hidden Layer pass
        x = nn.functional.relu(self.input_layer(x))
        h_1 = self.activation_1(self.hidden_layer_1(x))

        # Use dropout if dropout key specified as True
        if self.dropout[0]:
            h_1 = self.dropout_lyr_1(h_1)

        # Hidden & Output Layer pass
        h_2 = self.activation_2(self.hidden_layer_2(h_1))

        # Use dropout if dropout key specified as True
        if self.dropout[0]:
            h_2 = self.dropout_lyr_2(h_2)

        y_pred = self.output_layer(h_2)

        # if Polarity dataset is used, Get last prediction
        if self.mlp_type == MLPType.Polarity:
            y_pred = y_pred[:, -1]

        # Get probability of classes
        y_pred = nn.functional.log_softmax(y_pred, dim=1)
        return y_pred


class AE(nn.Module):
    """
        Autoencoder Class
        reference: https://www.geeksforgeeks.org/implementing-an-autoencoder-in-pytorch/
    """

    def __init__(self, dropout=(False, 0.5)):
        """
        :param dropout: boolean value that specified that dropout regularizer used or not
        """
        super().__init__()
        random.seed(501)
        np.random.seed(501)
        torch.manual_seed(501)
        self.dropout = dropout

        # Linear encoder
        self.input_layer = nn.Linear(28 * 28, 256)
        self.hidden_layer_1 = nn.Linear(256, 128)
        self.hidden_layer_2 = nn.Linear(128, 64)
        self.hidden_layer_3 = nn.Linear(64, 32)
        self.hidden_layer_4 = nn.Linear(32, 16)

        # Linear Decoder
        self.hidden_layer_5 = nn.Linear(16, 32)
        self.hidden_layer_6 = nn.Linear(32, 64)
        self.hidden_layer_7 = nn.Linear(64, 128)
        self.hidden_layer_8 = nn.Linear(128, 256)
        self.hidden_layer_9 = nn.Linear(256, 256)
        self.out_layer = nn.Linear(256, 28 * 28)
        self.sigmoid = nn.Sigmoid()

        # Adding drooput layer only after input worked better
        # instead of adding droput after every encoder layer (before ReLu's)
        # Dropout layer
        self.dropout = dropout
        if self.dropout[0]:
            self.dropout_lyr = nn.Dropout(dropout[1])

        # Activation Layers
        self.activation_1 = nn.ReLU()
        self.activation_2 = nn.ReLU()
        self.activation_3 = nn.ReLU()
        self.activation_4 = nn.ReLU()
        self.activation_5 = nn.ReLU()
        self.activation_6 = nn.ReLU()
        self.activation_7 = nn.ReLU()
        self.activation_8 = nn.ReLU()
        self.activation_9 = nn.ReLU()

    def forward(self, x):
        """
        Autoencoder forward function
        :param x: input data
        :return: reconstructed data
        """
        # Remove the dimensions of input of size 1.
        x = torch.squeeze(x)

        # Input layer pass
        h_1 = self.input_layer(x)

        # Use dropout if dropout key specified as True
        if self.dropout[0]:
            h_1 = self.dropout_lyr(h_1)

        # Linear Encoder pass
        h_2 = self.activation_1(self.hidden_layer_1(h_1))
        h_3 = self.activation_2(self.hidden_layer_2(h_2))
        h_4 = self.activation_3(self.hidden_layer_3(h_3))
        h_5 = self.activation_4(self.hidden_layer_4(h_4))

        # Linear decoder pass
        h_6 = self.activation_5(self.hidden_layer_5(h_5))
        h_7 = self.activation_6(self.hidden_layer_6(h_6))
        h_8 = self.activation_7(self.hidden_layer_7(h_7))
        h_9 = self.activation_8(self.hidden_layer_8(h_8))
        h_10 = self.activation_9(self.hidden_layer_9(h_9))

        # Out layer pass
        out = self.out_layer(h_10)
        out = self.sigmoid(out)
        return out


class LeNet5(nn.Module):
    def __init__(self, n_classes=10, dropout=(False, 0.5)):
        """
        LeNet5 Class(LeCun et al. 1998)
        reference: https://towardsdatascience.com/implementing-yann-lecuns-lenet-5-in-pytorch-5e05a0911320
        :param n_classes: the number of output classes
        :param dropout: boolean value that specified that dropout regularizer used or not
        """
        super().__init__()
        random.seed(501)
        np.random.seed(501)
        torch.manual_seed(501)

        layers = []

        # Add dropout if dropout key specified as True
        if dropout[0]:
            layers.append(nn.Dropout(dropout[1]))

        layers.extend([nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
                       nn.Tanh(),
                       nn.AvgPool2d(kernel_size=2)])

        # Add dropout if dropout key specified as True
        if dropout[0]:
            layers.append(nn.Dropout(dropout[1]))

        layers.extend([nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
                       nn.Tanh(),
                       nn.AvgPool2d(kernel_size=2)])

        # Add dropout if dropout key specified as True
        if dropout[0]:
            layers.append(nn.Dropout(dropout[1]))

        layers.extend([nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
                       nn.Tanh()])

        # Add layers to feature extractor
        self.feature_extractor = nn.Sequential(*layers)

        # Hidden layers
        self.hidden_layer1 = nn.Linear(in_features=120, out_features=84)
        self.hidden_layer2 = nn.Linear(in_features=84, out_features=n_classes)

        # Activation Layer
        self.activation_1 = nn.Tanh()

    def forward(self, x):
        """
        LeNet forward function
        :param x: input data
        :return: probability of classes
        """

        # Extract Feature
        x = self.feature_extractor(x)
        out = torch.flatten(x, 1)

        # Hidden & Output Layer pass
        out = self.activation_1(self.hidden_layer1(out))
        tag_scores = self.hidden_layer2(out)

        # Get probability of classes
        probs = nn.functional.log_softmax(tag_scores, dim=1)
        return probs


class VGG(nn.Module):
    """
    Pretrained VGG Class
    reference: https://medium.com/@buiminhhien2k/solving-cifar10-dataset-with-vgg16-pre-trained-architect-using-pytorch-validation-accuracy-over-3f9596942861
    """
    def __init__(self, dropout=(False, 0.5)):
        """
        :param dropout: boolean value that specified that dropout regularizer used or not
        """
        super().__init__()
        random.seed(501)
        np.random.seed(501)
        torch.manual_seed(501)

        # Load pretrained VGG from torchvision
        self.base_vgg = torchvision.models.vgg16(pretrained=True)

        # Update Last Layer dimension
        last_layer = self.base_vgg.classifier[-1].in_features
        self.base_vgg.classifier[-1] = nn.Linear(last_layer, 10)

        # Add dropout if dropout key specified as True
        if dropout[0]:
            feats_list = list(self.base_vgg.features)
            new_feats_list = []
            for feat in feats_list:
                new_feats_list.append(feat)
                if isinstance(feat, nn.Conv2d):
                    new_feats_list.append(nn.Dropout(p=dropout[1], inplace=True))
            # Modify convolution layers
            self.base_vgg.features = nn.Sequential(*new_feats_list)

    def forward(self, x):
        """
        VGG forward function
        :param x: input data
        :return: probability of classes
        """
        # VGG network pass
        x = self.base_vgg(x)

        # Get probability of classes
        probs = nn.functional.log_softmax(x, dim=1)
        return probs
