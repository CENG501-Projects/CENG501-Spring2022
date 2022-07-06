import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ======== Instance Discrimination Criterion  ========

# PC_loss and FC_loss classes are adapted from SimCLR implementation in https://github.com/Spijkervet/SimCLR


class PC_loss(nn.Module):
    def __init__(self, args):
        super(PC_loss, self).__init__()
        self.batch_size = args.batch_size
        self.c_subhed_num = args.c_subhed_num

        self.mask = self.mask_correlated_samples(args.batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2*batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, q):
        losses = 0
        for i in range(self.c_subhed_num):
            q_i = q[:, i, 0]
            q_j = q[:, i, 1]

            N = 2*self.batch_size

            z = torch.cat((q_i, q_j), dim=0)

            # corresponds to log(q^T qi), equation 9
            sim = torch.log(torch.matmul(z, z.T))

            sim_i_j = torch.diag(sim, self.batch_size)
            sim_j_i = torch.diag(sim, -self.batch_size)

            positive_samples = torch.cat(
                (sim_i_j, sim_j_i), dim=0).reshape(N, 1)
            negative_samples = sim[self.mask].reshape(N, -1)

            labels = torch.zeros(N).to(positive_samples.device).long()
            logits = torch.cat((positive_samples, negative_samples), dim=1)
            loss = self.criterion(logits, labels)
            loss /= N
            losses += loss

        return losses/self.c_subhed_num


class Entropy(nn.Module):
    def __init__(self, args):
        super(Entropy, self).__init__()
        self.c_subhed_num = args.c_subhed_num

    def entropy(self, x):
        x = x*torch.log(x)
        x = -x.sum(dim=1)
        return x.mean()

    def forward(self, q):
        entropies = 0
        for i in range(self.c_subhed_num):
            q_i = q[:, i, 0]
            q_j = q[:, i, 1]

            h_i = self.entropy(q_i)
            h_j = self.entropy(q_j)
            entropies += ((h_i + h_j)/2)

        return entropies/self.c_subhed_num


class FC_loss(nn.Module):
    def __init__(self, args):
        super(FC_loss, self).__init__()
        self.batch_size = args.batch_size
        self.temperature = args.temperature

        self.mask = self.mask_correlated_samples(args.batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2*batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        N = 2*self.batch_size

        z = torch.cat((z_i, z_j), dim=0)

        # corresponds to f(x, xi)/tau, equation 4
        sim = self.similarity_f(z.unsqueeze(
            1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

# ======== =============================  ========


# ======== Model with SimCLR base ========

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, input_dim, bias=False)
        self.layer2 = nn.Linear(input_dim, output_dim, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


class CRLC(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, args):
        super(CRLC, self).__init__()

        self.device = args.device
        self.c_subhed_num = args.c_subhed_num
        self.c_head_dim = args.c_head_dim

        if args.model == "resnet34":
            self.encoder = models.resnet34(pretrained=False)
            self.feature_dim = 512
        self.encoder.fc = Identity()

        self.rl_head = MLP(self.feature_dim, args.rl_head_dim)

        self.c_heads = nn.ModuleList(
            [MLP(self.feature_dim, self.c_head_dim) for i in range(args.c_subhed_num)])

        self.gamma = 0.01
        self.r = torch.ones(
            args.c_head_dim, device=args.device)/self.c_head_dim
        self.r.requires_grad = False

    def forward(self, x_i, x_j):
        batch_size = len(x_i)

        if x_j is None:     # means evaluation
            h_i = self.encoder(x_i)
            # rl_i = self.rl_head(h_i)
            q_i = F.softmax(self.c_heads[0](h_i), dim=-1)
            q_i = (1 - self.gamma)*q_i + self.gamma*self.r
            return q_i
            # cluster_head_probs = torch.zeros(
            #     batch_size, self.c_subhed_num, self.c_head_dim, device=self.device)
            #
            # for i, c_head in enumerate(self.c_heads):
            #     q_i = F.softmax(c_head(h_i), dim=-1)
            #     q_i = (1 - self.gamma)*q_i + self.gamma*self.r
            #     cluster_head_probs[:, i] = q_i
            #
            # return cluster_head_probs.mean(dim=1)

        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        # representations
        rl_i = self.rl_head(h_i)
        rl_j = self.rl_head(h_j)

        # class probabilities
        cluster_head_probs = torch.zeros(
            batch_size, self.c_subhed_num, 2, self.c_head_dim, device=self.device)
        for i, c_head in enumerate(self.c_heads):
            q_i = F.softmax(c_head(h_i), dim=-1)
            q_j = F.softmax(c_head(h_j), dim=-1)

            q_i = (1 - self.gamma)*q_i + self.gamma*self.r
            q_j = (1 - self.gamma)*q_j + self.gamma*self.r

            cluster_head_probs[:, i, 0] = q_i
            cluster_head_probs[:, i, 1] = q_j

        return rl_i, rl_j, cluster_head_probs


# ======== =============================  ========
