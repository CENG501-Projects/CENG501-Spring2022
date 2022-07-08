import _init_paths

import logging as lg
import torch
from torchvision import datasets, transforms

from bootstrap import bootstrap as bs
from ccnn import ccnn
from cnn import cnn

# Globals for configuration:
BATCH_SIZE = 256
NUM_WORKERS = 1
DATASET_PATH = "datasets/"
CNN_LR = 5e-4
# List of datasets that can be used
DATASETS_LIST = ["MNIST", "FashionMNIST", "CIFAR10"]
DATASET = "MNIST"
DATASET_TRAIN_CNT = -1
DATASET_TEST_CNT = 1000

lg.basicConfig(format='%(levelname)s\t- %(asctime)s\t- %(message)s',
               datefmt='%m/%d/%Y-%H:%M:%S',
               level=lg.DEBUG)

if __name__ == "__main__":
    if DATASET not in DATASETS_LIST:
        raise ValueError("Given dataset is not supported: " + DATASET)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    lg.info("Using device:" + str(device))
    
    train_transform_list = [
        # transforms.RandomCrop(...)
    ]
    test_transform_list = []
    
    # Add tensor conversion
    train_transform_list.append(transforms.ToTensor())
    test_transform_list.append(transforms.ToTensor())
    
    # Compose transformations
    train_transform = transforms.Compose(train_transform_list)
    test_transform = transforms.Compose(test_transform_list)
    
    lg.info("Setting up the dataset...")
    
    if DATASET == "MNIST":
        train_dset = datasets.MNIST(DATASET_PATH, train=True, transform=train_transform, download=True)
        test_dset = datasets.MNIST(DATASET_PATH, train=False, transform=test_transform, download=True)
        class_cnt = 10
    elif DATASET == "FashionMNIST":
        train_dset = datasets.FashionMNIST(DATASET_PATH, train=True, transform=train_transform, download=True)
        test_dset = datasets.FashionMNIST(DATASET_PATH, train=False, transform=test_transform, download=True)
        class_cnt = 10
    elif DATASET == "CIFAR10":
        train_dset = datasets.CIFAR10(DATASET_PATH, train=True, transform=train_transform, download=True)
        test_dset = datasets.CIFAR10(DATASET_PATH, train=False, transform=test_transform, download=True)
        class_cnt = 10
    else:
        raise ValueError("Unrecognized dataset name: " + DATASET)
    
    if DATASET_TRAIN_CNT > 0:
        train_dset = torch.utils.data.Subset(train_dset, range(0, DATASET_TRAIN_CNT))
    
    if DATASET_TEST_CNT > 0:
        test_dset = torch.utils.data.Subset(test_dset, range(0, DATASET_TEST_CNT))
    
    train_dl = torch.utils.data.DataLoader(
        dataset=train_dset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    test_dl = torch.utils.data.DataLoader(
        dataset=test_dset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    
    lg.info("Using dataset: " + DATASET)
    
    lg.info("Begin bootstrap with CCNN...")
    
    ccnn_model = ccnn.CCNN(
        train_dl=train_dl,
        test_dl=test_dl,
        train_img_cnt=len(train_dset),
        test_img_cnt=len(test_dset),
        device=torch.device("cpu")  # Temporarily use CPU only. TODO: Change after implementing torch tensors
    )
    
    bs.ccnn_bootstrap(ccnn_model, train_dset, test_dl, BATCH_SIZE, device=device)

    lg.info("Begin bootstrap with LeNet (described in paper)...")
    
    cnn_model = cnn.LeNet(n_classes=class_cnt, paper_params=True).to(device)
    cnn_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=CNN_LR)
    cnn_criterion = torch.nn.CrossEntropyLoss()

    bs.cnn_bootstrap(cnn_model, train_dset, test_dl, cnn_criterion, cnn_optimizer, BATCH_SIZE, num_class=class_cnt,
                     device=device)

    lg.info("Begin bootstrap with LeNet (better parameters)...")
    
    cnn_model = cnn.LeNet(n_classes=class_cnt, paper_params=False).to(device)
    cnn_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=CNN_LR)
    # cnn_criterion: Unchanged
    
    bs.cnn_bootstrap(cnn_model, train_dset, test_dl, cnn_criterion, cnn_optimizer, BATCH_SIZE, num_class=class_cnt,
                     device=device)

    lg.info("Finished the experiments.")
