# Copyright 2019 Karsten Roth and Biagio Brattoli
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

###################### LIBRARIES #################################################
import warnings
warnings.filterwarnings("ignore")

# import torch, random, itertools as it, numpy as np, faiss, random
import torch, random, itertools as it, numpy as np, random
from tqdm import tqdm

from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from PIL import Image

"""================================================================================================="""
############ LOSS SELECTION FUNCTION #####################
def loss_select(loss, opt, to_optim):
    """
    Selection function which returns the respective criterion while appending to list of trainable parameters if required.

    Args:
        loss:     str, name of loss function to return.
        opt:      argparse.Namespace, contains all training-specific parameters.
        to_optim: list of trainable parameters. Is extend if loss function contains those as well.
    Returns:
        criterion (torch.nn.Module inherited), to_optim (optionally appended)
    """
    if loss=='triplet':
        loss_params  = {'margin':opt.margin, 'sampling_method':opt.sampling}
        criterion    = TripletLoss(**loss_params)
    elif loss=='npair':
        loss_params  = {'l2':opt.l2npair}
        criterion    = NPairLoss(**loss_params)
    elif loss=='marginloss':
        loss_params  = {'margin':opt.margin, 'nu': opt.nu, 'beta':opt.beta, 'n_classes':opt.num_classes, 'sampling_method':opt.sampling}
        criterion    = MarginLoss(**loss_params)
        to_optim    += [{'params':criterion.parameters(), 'lr':opt.beta_lr, 'weight_decay':0}]
    elif loss=='crossentropy':
        loss_params  = {'n_classes':opt.num_classes, 'inp_dim':opt.embed_dim}
        criterion    = CEClassLoss(**loss_params)
        to_optim    += [{'params':criterion.parameters(), 'lr':opt.lr, 'weight_decay':0}]
    elif loss=="ranking":
        loss_params = {'N': opt.ranking_N, 'M': opt.ranking_bs, 's': opt.ranking_s, 'alfa': opt.ranking_alfa, 'beta': opt.ranking_beta, 'lambda_factor': opt.ranking_lambda}
        criterion = RankingLoss(**loss_params)
    else:
        raise Exception('Loss {} not available!'.format(loss))

    return criterion, to_optim





"""================================================================================================="""
######### MAIN SAMPLER CLASS #################################
class TupleSampler():
    """
    Container for all sampling methods that can be used in conjunction with the respective loss functions.
    Based on batch-wise sampling, i.e. given a batch of training data, sample useful data tuples that are
    used to train the network more efficiently.
    """
    def __init__(self, method='random'):
        """
        Args:
            method: str, name of sampling method to use.
        Returns:
            Nothing!
        """
        self.method = method
        if method=='semihard':
            self.give = self.semihardsampling
        if method=='softhard':
            self.give = self.softhardsampling
        elif method=='distance':
            self.give = self.distanceweightedsampling
        elif method=='npair':
            self.give = self.npairsampling
        elif method=='random':
            self.give = self.randomsampling

    def randomsampling(self, batch, labels):
        """
        This methods finds all available triplets in a batch given by the classes provided in labels, and randomly
        selects <len(batch)> triplets.

        Args:
            batch:  np.ndarray or torch.Tensor, batch-wise embedded training samples.
            labels: np.ndarray or torch.Tensor, ground truth labels corresponding to batch.
        Returns:
            list of sampled data tuples containing reference indices to the position IN THE BATCH.
        """
        if isinstance(labels, torch.Tensor): labels = labels.detach().numpy()
        unique_classes = np.unique(labels)
        indices        = np.arange(len(batch))
        class_dict     = {i:indices[labels==i] for i in unique_classes}

        sampled_triplets = [list(it.product([x],[x],[y for y in unique_classes if x!=y])) for x in unique_classes]
        sampled_triplets = [x for y in sampled_triplets for x in y]

        sampled_triplets = [[x for x in list(it.product(*[class_dict[j] for j in i])) if x[0]!=x[1]] for i in sampled_triplets]
        sampled_triplets = [x for y in sampled_triplets for x in y]

        #NOTE: The number of possible triplets is given by #unique_classes*(2*(samples_per_class-1)!)*(#unique_classes-1)*samples_per_class
        sampled_triplets = random.sample(sampled_triplets, batch.shape[0])
        return sampled_triplets


    def semihardsampling(self, batch, labels, margin=0.2):
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().numpy()
        bs = batch.size(0)
        #Return distance matrix for all elements in batch (BSxBS)
        distances = self.pdist(batch.detach()).detach().cpu().numpy()

        positives, negatives = [], []
        anchors = []
        for i in range(bs):
            l, d = labels[i], distances[i]
            neg = labels!=l; pos = labels==l

            anchors.append(i)
            pos[i] = False
            p      = np.random.choice(np.where(pos)[0])
            positives.append(p)

            #Find negatives that violate tripet constraint semi-negatives
            neg_mask = np.logical_and(neg,d>d[p])
            neg_mask = np.logical_and(neg_mask,d<margin+d[p])
            if neg_mask.sum()>0:
                negatives.append(np.random.choice(np.where(neg_mask)[0]))
            else:
                negatives.append(np.random.choice(np.where(neg)[0]))

        sampled_triplets = [[a, p, n] for a, p, n in zip(anchors, positives, negatives)]
        return sampled_triplets

    def softhardsampling(self, batch, labels):
        """
        This methods finds all available triplets in a batch given by the classes provided in labels, and select
        triplets based on semihard sampling introduced in 'https://arxiv.org/pdf/1503.03832.pdf'.

        Args:
            batch:  np.ndarray or torch.Tensor, batch-wise embedded training samples.
            labels: np.ndarray or torch.Tensor, ground truth labels corresponding to batch.
        Returns:
            list of sampled data tuples containing reference indices to the position IN THE BATCH.
        """
        if isinstance(labels, torch.Tensor): labels = labels.detach().numpy()
        bs = batch.size(0)
        #Return distance matrix for all elements in batch (BSxBS)
        distances = self.pdist(batch.detach()).detach().cpu().numpy()

        positives, negatives = [], []
        anchors = []
        for i in range(bs):
            l, d = labels[i], distances[i]
            anchors.append(i)
            #1 for batchelements with label l
            neg = labels!=l; pos = labels==l
            #0 for current anchor
            pos[i] = False

            #Find negatives that violate triplet constraint semi-negatives
            neg_mask = np.logical_and(neg,d<d[np.where(pos)[0]].max())
            #Find positives that violate triplet constraint semi-hardly
            pos_mask = np.logical_and(pos,d>d[np.where(neg)[0]].min())

            if pos_mask.sum()>0:
                positives.append(np.random.choice(np.where(pos_mask)[0]))
            else:
                positives.append(np.random.choice(np.where(pos)[0]))

            if neg_mask.sum()>0:
                negatives.append(np.random.choice(np.where(neg_mask)[0]))
            else:
                negatives.append(np.random.choice(np.where(neg)[0]))

        sampled_triplets = [[a, p, n] for a, p, n in zip(anchors, positives, negatives)]
        return sampled_triplets


    def distanceweightedsampling(self, batch, labels, lower_cutoff=0.5, upper_cutoff=1.4):
        """
        This methods finds all available triplets in a batch given by the classes provided in labels, and select
        triplets based on distance sampling introduced in 'Sampling Matters in Deep Embedding Learning'.

        Args:
            batch:  np.ndarray or torch.Tensor, batch-wise embedded training samples.
            labels: np.ndarray or torch.Tensor, ground truth labels corresponding to batch.
            lower_cutoff: float, lower cutoff value for negatives that are too close to anchor embeddings. Set to literature value. They will be assigned a zero-sample probability.
            upper_cutoff: float, upper cutoff value for positives that are too far away from the anchor embeddings. Set to literature value. They will be assigned a zero-sample probability.
        Returns:
            list of sampled data tuples containing reference indices to the position IN THE BATCH.
        """
        if isinstance(labels, torch.Tensor): labels = labels.detach().cpu().numpy()
        bs = batch.shape[0]

        distances    = self.pdist(batch.detach()).clamp(min=lower_cutoff)



        positives, negatives = [],[]
        labels_visited = []
        anchors = []

        for i in range(bs):
            neg = labels!=labels[i]; pos = labels==labels[i]
            q_d_inv = self.inverse_sphere_distances(batch, distances[i], labels, labels[i])
            #Sample positives randomly
            pos[i] = 0
            positives.append(np.random.choice(np.where(pos)[0]))
            #Sample negatives by distance
            negatives.append(np.random.choice(bs,p=q_d_inv))

        sampled_triplets = [[a,p,n] for a,p,n in zip(list(range(bs)), positives, negatives)]
        return sampled_triplets


    def npairsampling(self, batch, labels):
        """
        This methods finds N-Pairs in a batch given by the classes provided in labels in the
        creation fashion proposed in 'Improved Deep Metric Learning with Multi-class N-pair Loss Objective'.

        Args:
            batch:  np.ndarray or torch.Tensor, batch-wise embedded training samples.
            labels: np.ndarray or torch.Tensor, ground truth labels corresponding to batch.
        Returns:
            list of sampled data tuples containing reference indices to the position IN THE BATCH.
        """
        if isinstance(labels, torch.Tensor):    labels = labels.detach().cpu().numpy()

        label_set, count = np.unique(labels, return_counts=True)
        label_set  = label_set[count>=2]
        pos_pairs  = np.array([np.random.choice(np.where(labels==x)[0], 2, replace=False) for x in label_set])
        neg_tuples = []

        for idx in range(len(pos_pairs)):
            neg_tuples.append(pos_pairs[np.delete(np.arange(len(pos_pairs)),idx),1])

        neg_tuples = np.array(neg_tuples)

        sampled_npairs = [[a,p,*list(neg)] for (a,p),neg in zip(pos_pairs, neg_tuples)]
        return sampled_npairs


    def pdist(self, A):
        """
        Efficient function to compute the distance matrix for a matrix A.

        Args:
            A:   Matrix/Tensor for which the distance matrix is to be computed.
            eps: float, minimal distance/clampling value to ensure no zero values.
        Returns:
            distance_matrix, clamped to ensure no zero values are passed.
        """
        prod = torch.mm(A, A.t())
        norm = prod.diag().unsqueeze(1).expand_as(prod)
        res = (norm + norm.t() - 2 * prod).clamp(min = 0)
        return res.clamp(min = 0).sqrt()


    def inverse_sphere_distances(self, batch, dist, labels, anchor_label):
        """
        Function to utilise the distances of batch samples to compute their
        probability of occurence, and using the inverse to sample actual negatives to the resp. anchor.

        Args:
            batch:        torch.Tensor(), batch for which the sampling probabilities w.r.t to the anchor are computed. Used only to extract the shape.
            dist:         torch.Tensor(), computed distances between anchor to all batch samples.
            labels:       np.ndarray, labels for each sample for which distances were computed in dist.
            anchor_label: float, anchor label
        Returns:
            distance_matrix, clamped to ensure no zero values are passed.
        """
        bs,dim       = len(dist),batch.shape[-1]

        #negated log-distribution of distances of unit sphere in dimension <dim>
        log_q_d_inv = ((2.0 - float(dim)) * torch.log(dist) - (float(dim-3) / 2) * torch.log(1.0 - 0.25 * (dist.pow(2))))
        #Set sampling probabilities of positives to zero
        log_q_d_inv[np.where(labels==anchor_label)[0]] = 0

        q_d_inv     = torch.exp(log_q_d_inv - torch.max(log_q_d_inv)) # - max(log) for stability
        #Set sampling probabilities of positives to zero
        q_d_inv[np.where(labels==anchor_label)[0]] = 0

        ### NOTE: Cutting of values with high distances made the results slightly worse.
        # q_d_inv[np.where(dist>upper_cutoff)[0]]    = 0

        #Normalize inverted distance for probability distr.
        q_d_inv = q_d_inv/q_d_inv.sum()
        return q_d_inv.detach().cpu().numpy()




"""================================================================================================="""
### Standard Triplet Loss, finds triplets in Mini-batches.
class TripletLoss(torch.nn.Module):
    def __init__(self, margin=1, sampling_method='random'):
        """
        Basic Triplet Loss as proposed in 'FaceNet: A Unified Embedding for Face Recognition and Clustering'
        Args:
            margin:             float, Triplet Margin - Ensures that positives aren't placed arbitrarily close to the anchor.
                                Similarl, negatives should not be placed arbitrarily far away.
            sampling_method:    Method to use for sampling training triplets. Used for the TupleSampler-class.
        """
        super(TripletLoss, self).__init__()
        self.margin             = margin
        self.sampler            = TupleSampler(method=sampling_method)

    def triplet_distance(self, anchor, positive, negative):
        """
        Compute triplet loss.

        Args:
            anchor, positive, negative: torch.Tensor(), resp. embeddings for anchor, positive and negative samples.
        Returns:
            triplet loss (torch.Tensor())
        """
        return torch.nn.functional.relu((anchor-positive).pow(2).sum()-(anchor-negative).pow(2).sum()+self.margin)

    def forward(self, batch, labels):
        """
        Args:
            batch:   torch.Tensor() [(BS x embed_dim)], batch of embeddings
            labels:  np.ndarray [(BS x 1)], for each element of the batch assigns a class [0,...,C-1]
        Returns:
            triplet loss (torch.Tensor(), batch-averaged)
        """
        #Sample triplets to use for training.
        sampled_triplets = self.sampler.give(batch, labels)
        #Compute triplet loss
        loss             = torch.stack([self.triplet_distance(batch[triplet[0],:],batch[triplet[1],:],batch[triplet[2],:]) for triplet in sampled_triplets])

        return torch.mean(loss)



"""================================================================================================="""
### Standard N-Pair Loss.
class NPairLoss(torch.nn.Module):
    def __init__(self, l2=0.02):
        """
        Basic N-Pair Loss as proposed in 'Improved Deep Metric Learning with Multi-class N-pair Loss Objective'

        Args:
            l2: float, weighting parameter for weight penality due to embeddings not being normalized.
        Returns:
            Nothing!
        """
        super(NPairLoss, self).__init__()
        self.sampler = TupleSampler(method='npair')
        self.l2      = l2

    def npair_distance(self, anchor, positive, negatives):
        """
        Compute basic N-Pair loss.

        Args:
            anchor, positive, negative: torch.Tensor(), resp. embeddings for anchor, positive and negative samples.
        Returns:
            n-pair loss (torch.Tensor())
        """
        return torch.log(1+torch.sum(torch.exp(anchor.mm((negatives-positive).transpose(0,1)))))

    def weightsum(self, anchor, positive):
        """
        Compute weight penalty.
        NOTE: Only need to penalize anchor and positive since the negatives are created based on these.

        Args:
            anchor, positive: torch.Tensor(), resp. embeddings for anchor and positive samples.
        Returns:
            torch.Tensor(), Weight penalty
        """
        return torch.sum(anchor**2+positive**2)

    def forward(self, batch, labels):
        """
        Args:
            batch:   torch.Tensor() [(BS x embed_dim)], batch of embeddings
            labels:  np.ndarray [(BS x 1)], for each element of the batch assigns a class [0,...,C-1]
        Returns:
            n-pair loss (torch.Tensor(), batch-averaged)
        """
        #Sample N-Pairs
        sampled_npairs = self.sampler.give(batch, labels)
        #Compute basic n=pair loss
        loss           = torch.stack([self.npair_distance(batch[npair[0]:npair[0]+1,:],batch[npair[1]:npair[1]+1,:],batch[npair[2:],:]) for npair in sampled_npairs])
        #Include weight penalty
        loss           = loss + self.l2*torch.mean(torch.stack([self.weightsum(batch[npair[0],:], batch[npair[1],:]) for npair in sampled_npairs]))

        return torch.mean(loss)




"""================================================================================================="""
### MarginLoss with trainable class separation margin beta. Runs on Mini-batches as well.
class MarginLoss(torch.nn.Module):
    def __init__(self, margin=0.2, nu=0, beta=1.2, n_classes=100, beta_constant=False, sampling_method='distance'):
        """
        Basic Margin Loss as proposed in 'Sampling Matters in Deep Embedding Learning'.

        Args:
            margin:          float, fixed triplet margin (see also TripletLoss).
            nu:              float, regularisation weight for beta. Zero by default (in literature as well).
            beta:            float, initial value for trainable class margins. Set to default literature value.
            n_classes:       int, number of target class. Required because it dictates the number of trainable class margins.
            beta_constant:   bool, set to True if betas should not be trained.
            sampling_method: str, sampling method to use to generate training triplets.
        Returns:
            Nothing!
        """
        super(MarginLoss, self).__init__()
        self.margin             = margin
        self.n_classes          = n_classes
        self.beta_constant     = beta_constant

        self.beta_val = beta
        self.beta     = beta if beta_constant else torch.nn.Parameter(torch.ones(n_classes)*beta)

        self.nu                 = nu

        self.sampling_method    = sampling_method
        self.sampler            = TupleSampler(method=sampling_method)


    def forward(self, batch, labels):
        """
        Args:
            batch:   torch.Tensor() [(BS x embed_dim)], batch of embeddings
            labels:  np.ndarray [(BS x 1)], for each element of the batch assigns a class [0,...,C-1]
        Returns:
            margin loss (torch.Tensor(), batch-averaged)
        """
        if isinstance(labels, torch.Tensor): labels = labels.detach().cpu().numpy()

        sampled_triplets = self.sampler.give(batch, labels)

        #Compute distances between anchor-positive and anchor-negative.
        d_ap, d_an = [],[]
        for triplet in sampled_triplets:
            train_triplet = {'Anchor': batch[triplet[0],:], 'Positive':batch[triplet[1],:], 'Negative':batch[triplet[2]]}

            pos_dist = ((train_triplet['Anchor']-train_triplet['Positive']).pow(2).sum()+1e-8).pow(1/2)
            neg_dist = ((train_triplet['Anchor']-train_triplet['Negative']).pow(2).sum()+1e-8).pow(1/2)

            d_ap.append(pos_dist)
            d_an.append(neg_dist)
        d_ap, d_an = torch.stack(d_ap), torch.stack(d_an)

        #Group betas together by anchor class in sampled triplets (as each beta belongs to one class).
        if self.beta_constant:
            beta = self.beta
        else:
            beta = torch.stack([self.beta[labels[triplet[0]]] for triplet in sampled_triplets]).type(torch.cuda.FloatTensor)

        #Compute actual margin postive and margin negative loss
        pos_loss = torch.nn.functional.relu(d_ap-beta+self.margin)
        neg_loss = torch.nn.functional.relu(beta-d_an+self.margin)

        #Compute normalization constant
        pair_count = torch.sum((pos_loss>0.)+(neg_loss>0.)).type(torch.cuda.FloatTensor)

        #Actual Margin Loss
        loss = torch.sum(pos_loss+neg_loss) if pair_count==0. else torch.sum(pos_loss+neg_loss)/pair_count

        #(Optional) Add regularization penalty on betas.
        if self.nu: loss = loss + beta_regularisation_loss.type(torch.cuda.FloatTensor)

        return loss




"""================================================================================================="""
class CEClassLoss(torch.nn.Module):
    def __init__(self, inp_dim, n_classes):
        """
        Basic Cross Entropy Loss for reference. Can be useful.
        Contains its own mapping network, so the actual network can remain untouched.

        Args:
            inp_dim:   int, embedding dimension of network.
            n_classes: int, number of target classes.
        Returns:
            Nothing!
        """
        super(CEClassLoss, self).__init__()
        self.mapper  = torch.nn.Sequential(torch.nn.Linear(inp_dim, n_classes))
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, batch, labels):
        """
        Args:
            batch:   torch.Tensor() [(BS x embed_dim)], batch of embeddings
            labels:  np.ndarray [(BS x 1)], for each element of the batch assigns a class [0,...,C-1]
        Returns:
            cross-entropy loss (torch.Tensor(), batch-averaged by default)
        """
        return self.ce_loss(self.mapper(batch), labels.type(torch.cuda.LongTensor))


"""================================================================================================="""
class RankingLoss(torch.nn.Module):
    """
    Container for Ranking Loss as proposed in "Deep Metric Learning with Self-Supervised Ranking" 
    """
    def __init__(self, N, M, s, alfa, beta, lambda_factor):
        self.N = N
        self.M = M
        self.s = s
        self.alfa = alfa
        self.beta = beta
        self.lambda_factor = lambda_factor

        super(RankingLoss, self).__init__()

    def calculate_similarity_vector(self, embeddings):
        """
        This function calculates the pairwise similarity vector. Pairwise similarity vector 
        contains the similarities between the embeddings of the orijinal image and augmented views.

        Args:
            embeddings: Embeddings matrix of shape [M*(N+1)*D]. Function assumes that embeddings are ordered
            according to augmentation strengths within each N+1 length block.

        Returns:
            -
        """
        # Take the embedding vector for the non-augmented orijinal sample
        z_prime_0 = embeddings[0, :]
        # Shape check
        # print("Shape of z_prime_0:", z_prime_0.shape)

        # Pack the other embeddings into a matrix
        augmented_embeddings = embeddings[1:, :]
        # Shape check
        # print("Shape of augmented_embeddings:", augmented_embeddings.shape)

        # Calculate pairwise similarities
        self.pairwise_sim = torch.nn.functional.cosine_similarity(augmented_embeddings, z_prime_0)

        # We have normalized embeddings, so below code will give the cosine similartiy
        # pairwise_sim = torch.matmul(z_prime_0, torch.transpose(augmented_embeddings, 0, 1))
        # self.pairwise_sim = pairwise_sim.squeeze()
        # Shape check
        # print("Shape of pairwise_sim:", self.pairwise_sim.shape)

    def calculate_Lsort(self):
        """
        This function calculates the L_sort loss for a single augmented block which contains 
        embeddings of the orijinal image and its augmentions

        Args:
            -
        Returns:
            L_sort: Loss calculated over an augmented block
        """
        # Get the difference of similarites
        pairwise_sim_a = self.pairwise_sim[1: ]
        pairwise_sim_b = self.pairwise_sim[0:-1]
        sim_diff = pairwise_sim_a - pairwise_sim_b
        # Shape check
        # print("Shape of sim_diff:", sim_diff.shape)
        exponent = self.s * (sim_diff + self.alfa * torch.ones(self.N-1, device=sim_diff.device))
        exponential = torch.exp(exponent)

        L_sort = (1/self.s) * torch.log(1 + torch.sum(exponential))

        return L_sort

    def calculate_Lpos(self):
        """
        This function calculates the L_pos loss for a single augmented block which contains 
        embeddings of the orijinal image and its augmentations 

        Args:
            -
        Returns:
            L_pos: Loss calculated over an augmented block 
        """
        exponent = -self.s * (self.pairwise_sim - self.beta * torch.ones(self.N, device=self.pairwise_sim.device))
        exponential = torch.exp(exponent)

        L_pos = (1/self.s) * torch.log(1 + torch.sum(exponential))

        return L_pos

    def forward(self, embeddings):
        """
        This function calculates the Ranking loss avaraged over a mini-batch 
        
        Args:
            embeddings: Embeddings matrix of shape [M*(N+1)*D]. Function assumes that embeddings are ordered
            according to augmentation strengths within each N+1 length block.

        Returns:
            L_ranking: Ranking loss 
        """
        D = embeddings.shape[-1]
        # print("Shape of embeddings:", embeddings.shape)
        reshaped_embeddings = torch.reshape(embeddings, (self.M, self.N+1, D))
        # Shape check
        # print("Shape of reshaped_embeddings:", reshaped_embeddings.shape)
        L_sort_total = 0
        L_pos_total = 0

        # Avarege over Mini-batch
        for m in range(self.M):
            self.calculate_similarity_vector(reshaped_embeddings[m, :, :])
            L_sort_total += self.calculate_Lsort()
            L_pos_total += self.calculate_Lpos()
        L_sort_total /= self.M
        L_pos_total /= self.M

        L_ranking = L_sort_total + self.lambda_factor*L_pos_total

        return L_ranking




