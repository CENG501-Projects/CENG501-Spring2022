import argparse
import numpy as np
import pickle
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--dname', type=str, default='Cora', choices=['Cora', 'CiteSeer', 'PubMed'], help='Dataset name.')
parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
parser.add_argument('--epochs', type=int, default=20,
                    help='Number of epochs.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.01,
                    help='Weight decay.')
parser.add_argument('--hidden_dim', type=int, default=256,
                    help='Dimension of the hidden units.')
parser.add_argument('--num_layer', type=int, default=4,
                    help='Number of layers.')      
parser.add_argument('--C', type=int, default=10,
                    help='SVM regularization parameter.')           
                    


def calc_accuracy(preds, lbs):
    return np.sum(preds == lbs) / len(preds)


def export_data(dname, data):
    print(f"Saving the data as data/{dname}.pkl...")
    with open(f'data/{dname}.pkl', 'wb') as file:
        pickle.dump(data, file)


def import_data(dname):
    with open(f'data/{dname}.pkl', 'rb') as file:
        data = pickle.load(file)

    return data


def export_embeddings(dname, emb):
    print(f"Saving the embeddings as embeddings/{dname}.pkl...")
    with open(f'embeddings/{dname}.pkl', 'wb') as file:
        pickle.dump(emb, file)


def import_embeddings(dname):
    with open(f'embeddings/{dname}.pkl', 'rb') as file:
        emb = pickle.load(file)
    
    return emb


def visualize_loss(dname, loss_history):
    print(f"Saving the loss history figure as figures/{dname}.png...")
    plt.plot(loss_history)
    plt.xlabel('Epoch number')
    plt.ylabel('Loss value')
    plt.savefig(f'figures/{dname}.png')