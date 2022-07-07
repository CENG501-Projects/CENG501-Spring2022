import time
import torch
from torchvision.datasets import CelebA
from utils.functions import get_attribute_indices
from utils.functions import generate_latent_vectors
from utils.functions import extract_orthogonal_basis_set
from utils.functions import compute_distances
from utils.functions import get_max_min_values


if __name__ == '__main__':
    attr_indices = get_attribute_indices()
    attr_transform = lambda attr: attr[attr_indices]
    dataset = CelebA(root='datasets', split='test', target_transform=attr_transform)
    attr_names = [attr_name for i, attr_name in enumerate(dataset.attr_names) if i in attr_indices]

    log_idx = 6
    F, _ = extract_orthogonal_basis_set(idx=log_idx)
    wtest, ytest = generate_latent_vectors(partition='test')

    index = 300
    topk = 5

    M = len(attr_indices)
    pq = torch.ones(M)
    aq = torch.full(size=(M, ), fill_value=-1)
    # aq[attr_names.index('Blond_Hair')] = 1

    amax, amin = get_max_min_values(idx=log_idx)
    distances = compute_distances(F, wtest, wtest[index], pq, aq, amax, amin)[:topk]
    for i, dist in distances:
        print(f'Image-index: {i} -- Distance: {dist}')
        img = dataset[i][0]
        img.show()
        time.sleep(2)
