import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import add_self_loops


class PairData(Data):
	def __init__(self, edge_index, features, shuffled_features, labels, n_features, n_classes, train_mask, test_mask):
		super().__init__()

		self.edge_index = edge_index
		self.x = features
		self.shuffled_x = shuffled_features
		self.y = labels
		self.num_features = n_features
		self.num_classes = n_classes
		self.train_mask = train_mask
		self.test_mask = test_mask



def load_data(dname):

	dataset = Planetoid(root='/tmp/'+ dname, 
		name=dname, 
		split="random",
		num_train_per_class=40)

	num_classes = dataset.num_classes

	data = dataset.data

	num_features = data.num_node_features
	num_nodes = data.num_nodes
	train_mask = data.train_mask
	test_mask = ~train_mask

	features = data.x
	random_idx = torch.randperm(features.shape[0])
	shuffled_features = features[random_idx].view(features.size())

	edge_index, _ = add_self_loops(data.edge_index, num_nodes=num_nodes)

	labels = data.y

	pair_data = PairData(edge_index, features, shuffled_features, labels, num_features, num_classes, train_mask, test_mask)

	return pair_data