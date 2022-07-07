import torch
import torch.nn as nn
import numpy as np

from models import DIMP
from dataset import load_data
from utils import parser, export_data, export_embeddings, visualize_loss
from train import train
from test import classify_nodes, cluster_nodes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args = parser.parse_args()
criterion = nn.BCEWithLogitsLoss()
np.random.seed(args.seed)
torch.manual_seed(args.seed)



def process(dname):

    data = load_data(dname)
    export_data(dname, data)

    train_mask = data.train_mask
    test_mask = data.test_mask
    feature_dim = data.num_features
    num_classes = data.num_classes

    model = DIMP(args.num_layer, feature_dim, args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    loss_history = train(model, criterion, optimizer, args.epochs, data)
    visualize_loss(dname, loss_history)

    edge_index = data.edge_index
    x = data.x
    lbs = data.y

    train_lbs = lbs[train_mask == True].detach().numpy()
    test_lbs = lbs[test_mask == True].detach().numpy()

    emb = model.get_embeddings(edge_index, x)
    export_embeddings(dname, emb)

    train_emb = emb[train_mask == True]
    test_emb = emb[test_mask == True]

    return num_classes, train_emb, train_lbs, test_emb, test_lbs



def test(num_classes, train_emb, train_lbs, test_emb, test_lbs):

    _, cls_acc = classify_nodes(args.C, train_emb, train_lbs, test_emb, test_lbs)
    _, NMI, ARI = cluster_nodes(num_classes, args.seed, train_emb, test_emb, test_lbs)

    return cls_acc, NMI, ARI



if __name__ == "__main__":

    dname = args.dname

    num_classes, train_emb, train_lbs, test_emb, test_lbs = process(dname)
    acc, NMI, ARI = test(num_classes, train_emb, train_lbs, test_emb, test_lbs)
    print(f"\nResults for the {dname} Dataset")
    print("-" * 24)
    print(f"Acc: {acc} \t NMI: {NMI} \t ARI:{ARI}")