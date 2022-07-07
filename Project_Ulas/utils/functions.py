import copy
import math
import os
import random
from argparse import Namespace
import dlib
import torch
import torchvision.transforms as T
from torchvision.datasets import CelebA
from pixel2style2pixel.models.psp import pSp
from pixel2style2pixel.scripts.align_all_parallel import align_face
########################################################################################################################
def set_seed(seed=2023):
    random.seed(seed)
    torch.manual_seed(seed)
########################################################################################################################
def get_encoder(verbose=False):
    weights_path = f'weights{os.sep}psp_ffhq_encode.pt'
    ckpt = torch.load(weights_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = weights_path
    opts['output_size'] = 1024
    if verbose: print(opts)
    return pSp(Namespace(**opts))
########################################################################################################################
def get_attribute_indices():
    dataset = CelebA(root='datasets')
    masked_attr = ['Blurry', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Necklace', 'Wearing_Necktie']
    masked_indices = [dataset.attr_names.index(ma) for ma in masked_attr]
    return [i for i in range(dataset.attr.shape[1]) if i not in masked_indices]
########################################################################################################################
def generate_latent_vectors(partition, N=int(2e4)):
    latents_path = f'weights{os.sep}{partition}_latent_vectors.pt'
    if os.path.exists(latents_path):
        tmp = torch.load(latents_path, map_location='cpu')
        return tmp['wset'], tmp['yset']

    set_seed()

    img_transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    attr_indices = get_attribute_indices()
    attr_transform = lambda attr: attr[attr_indices]
    dataset = CelebA(root='datasets', split=partition, target_transform=attr_transform)

    encoder = get_encoder()
    encoder.eval()
    encoder.cuda()

    dataset_length = len(dataset) if partition != 'train' else N
    with torch.no_grad(): latent_length = encoder(img_transform(dataset[0][0]).unsqueeze(0).cuda()).numel()
    wset = torch.empty(size=(dataset_length, latent_length), dtype=torch.float)
    yset = torch.empty(size=(dataset_length, len(attr_indices)), dtype=torch.long)

    predictor = dlib.shape_predictor(f'weights{os.sep}shape_predictor_68_face_landmarks.dat')
    random_indices = range(dataset_length) if partition != 'train' else random.sample(range(len(dataset)), k=N)
    with torch.no_grad():
        for i, index in enumerate(random_indices):
            _, label, img_path = dataset[index]
            image = img_transform(align_face(filepath=img_path, predictor=predictor))
            wset[i] = encoder(image.unsqueeze(0).cuda()).reshape(-1)
            yset[i] = label
            if (i + 1) % 1000 == 0: print(f'Generated the latent vector of {i + 1}-th image.')

    torch.save({
        'wset': wset,
        'yset': yset
    }, latents_path)
    return wset, yset
########################################################################################################################
def find_nearest_orthogonal_set(fset):
    cset = (fset ** 2).sum(dim=0).sqrt()
    F = fset / cset
    A, V = torch.linalg.eig((F.T @ F).inverse())
    Fhat = F @ (V @ A.diag().sqrt() @ V.inverse()).real
    return cset * Fhat
########################################################################################################################
def hinge_loss(pred, target):
    loss = 1 - target * pred
    loss[loss < 0] = 0
    return loss

def extract_orthogonal_basis_set(idx, lr=1e-2, reg=5e-3, epochs=int(1e5), stop_step=500, num_images=int(2e4)):
    basis_path = f'weights{os.sep}basis_set{idx}.pt'
    if os.path.exists(basis_path):
        tmp = torch.load(basis_path, map_location='cpu')
        return tmp['fset'], tmp['bset']

    set_seed()

    wset, yset = generate_latent_vectors(partition='train')
    yset[yset == 0] = -1
    wset, yset = wset.cuda(), yset.cuda()

    wvalset, yvalset = generate_latent_vectors(partition='valid')
    yvalset[yvalset == 0] = -1
    wvalset, yvalset = wvalset.cuda(), yvalset.cuda()

    dplus, M = wset.shape[1], yset.shape[1]

    fset = torch.rand(dplus, M, requires_grad=True, device='cuda')
    torch.nn.init.xavier_uniform_(fset)
    bset = torch.zeros(M, requires_grad=True, device='cuda')

    train_log_file = open(f'logs{os.sep}train_log{idx}.txt', 'w')
    train_log_file.write(f'lr: {lr} -- reg: {reg} -- epochs: {epochs} -- stop_step: {stop_step} -- num_images: {num_images}\n')

    count_loss_increase, best_val_loss = 0, 1e10
    for epoch in range(1, epochs + 1):
        train_loss_vector = torch.empty(M, dtype=torch.float)
        for m in range(M):
            loss = hinge_loss(wset @ fset[:, m] + bset[m], yset[:, m]).mean() + reg * fset[:, m].abs().sum()
            loss.backward()
            train_loss_vector[m] = loss.item()
            with torch.no_grad():
                fset[:, m] -= lr * fset.grad[:, m]
                bset[m] -= lr * bset.grad[m]
                fset.grad = bset.grad = None
        with torch.no_grad():
            fset = find_nearest_orthogonal_set(fset).requires_grad_(True)
            val_loss = hinge_loss(wvalset @ fset + bset, yvalset).sum(dim=1).mean().item()

        train_loss = train_loss_vector.sum().item()
        train_log_file.write(f'Epoch: {epoch} -- Train Loss: {train_loss} -- Val Loss: {val_loss}\n')

        if val_loss < best_val_loss:
            count_loss_increase = 0
            best_val_loss = val_loss
            best_fset = fset.detach().cpu()
            best_bset = bset.detach().cpu()
            print(f'Best weights at Epoch: {epoch} -- Train Loss: {train_loss} -- Val Loss: {val_loss}')
        else:
            count_loss_increase += 1
            if count_loss_increase == stop_step:
                print(f'Early stopping at Epoch: {epoch}')
                break
    train_log_file.close()
    best_fset /= (best_fset ** 2).sum(dim=0).sqrt()
    torch.save({
        'fset': best_fset,
        'bset': best_bset
    }, basis_path)
    return best_fset, best_bset
########################################################################################################################
def get_max_min_values(idx):
    wtrain, _ = generate_latent_vectors(partition='train')
    F, _ = extract_orthogonal_basis_set(idx=idx)
    result = wtrain @ F
    return result.max(dim=0)[0], result.min(dim=0)[0]

def compute_distances(F, wtest, wq, pq, aq, amax, amin):
    alpha = aq * (amax - amin) + amin
    alpha[(aq != 0) & (aq != 1)] = 0
    dF = wq.reshape(1, -1) @ F + alpha - wtest @ F
    dI = (wq - wtest) @ (torch.eye(len(wq)) - F @ F.T)
    distances = ((dF @ pq.diag()) * dF).sum(dim=1) + (dI ** 2).sum(dim=1)
    distances = list(enumerate(distances))
    distances.sort(key=lambda x: x[1])
    return distances
########################################################################################################################
def generate_queries(ytest, attr_index, attr_value=1, query_size=1000):
    chose_indices = []
    for i in range(len(ytest)):
        y = ytest[i].clone()
        y[attr_index] = attr_value
        if torch.where((y == ytest).all(dim=1))[0].numel() > 1: chose_indices.append(i)
    return chose_indices[:query_size]

def compute_nDCG(attr_name='Smiling', attr_value=1, topk=5, query_size=1000, idx=6):
    attr_indices = get_attribute_indices()
    attr_transform = lambda attr: attr[attr_indices]
    dataset = CelebA(root='datasets', split='test', target_transform=attr_transform)
    attr_names = [attr_name for i, attr_name in enumerate(dataset.attr_names) if i in attr_indices]
    wtest, ytest = generate_latent_vectors(partition='test')
    query_indices = generate_queries(ytest, attr_names.index(attr_name), attr_value, query_size)

    M = len(attr_indices)
    aq = torch.full(size=(M,), fill_value=-1)
    aq[attr_names.index(attr_name)] = attr_value
    pq = torch.ones(M)
    amax, amin = get_max_min_values(idx=idx)
    F, _ = extract_orthogonal_basis_set(idx=idx)

    ndcg_list = []
    for j, query in enumerate(query_indices, start=1):
        wq, yq = wtest[query], ytest[query]
        distances = compute_distances(F, wtest, wtest[query], pq, aq, amax, amin)[:topk]  # call it with different pq's
        relevancies = [(yq == ytest[i]).sum().item() for i, sc in distances]  # 1/dist can also be used.

        ideal = copy.deepcopy(relevancies)
        ideal.sort(reverse=True)
        dcg  = relevancies[0] + sum([rel / math.log2(i) for i, rel in enumerate(relevancies[1:], start=2)])
        ndcg = ideal[0] + sum([rel / math.log2(i) for i, rel in enumerate(ideal[1:], start=2)])
        ndcg_list.append(dcg / ndcg)

        print(f'Query-{j} has been processed.')
    return sum(ndcg_list) / len(ndcg_list)
########################################################################################################################
