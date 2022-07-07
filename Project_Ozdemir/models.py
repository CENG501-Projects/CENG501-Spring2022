import torch
import torch.nn as nn
from layers import DIMPLayer, Discriminator


class DIMP(nn.Module):
    def __init__(self, num_layer, input_dim, hidden_dim):
        super(DIMP, self).__init__()

        self.layer_list = nn.ModuleList()
        self.layer_list.append(DIMPLayer(input_dim, hidden_dim))
        for l in range(num_layer-1):
            self.layer_list.append(DIMPLayer(hidden_dim, hidden_dim))

        self.disc = Discriminator(hidden_dim)


    def forward(self, edge_index, x, shuffled_x):
        h = x
        for layer in self.layer_list:
            h = layer(h, edge_index)

        s = torch.mean(h, 0)

        shuffled_h = shuffled_x
        for layer in self.layer_list:
            shuffled_h = layer(shuffled_h, edge_index)

        d_h = self.disc(h, s)
        d_shuffled_h = self.disc(shuffled_h, s)

        disc_out = torch.cat((d_h, d_shuffled_h), 0)

        return disc_out


    def get_embeddings(self, edge_index, x):
        with torch.no_grad():
            h = x
            for layer in self.layer_list:
                h = layer(h, edge_index)

        return h.detach().numpy()

