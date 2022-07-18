from matplotlib import transforms
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader
from typing import Union, Tuple

def prepare_loader(
    image_size: int,
    batch_size: int,
    root: str='./data',
    train_or_test: str='train') -> Union[Tuple[Union[int, int, DataLoader]], Tuple[Union[DataLoader, DataLoader]]]:
    """
    Creates dataloader for CIFAR-10 data
    
    Parameters
    ---------
    image_size : int
        Resize value for both height and width
    batch_size: int
        Size of each batch
    root : str
        Path to load the data
    train_or_test: str
        Defining whether preparing the training or test data loader
        
    Returns
    -------
    data_loader : Union[Tuple[Union[int, int, DataLoader]], Tuple[Union[DataLoader, DataLoader]]]
         Data loader with batches
    """
    
    if train_or_test == 'train':
        transform = transforms.Compose([
                                transforms.Resize((image_size, image_size)),
                                transforms.ToTensor()])
        
        train_dataset = datasets.CIFAR10(root=root,
                                             train=True, 
                                             transform=transform,
                                             download=True)
        
        train_loader = train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)
        
        num_of_images = len(train_dataset)
        num_of_batches = len(train_loader)
        return (num_of_images, train_loader, num_of_batches)
    
    else:
        test_transform_1 = transforms.Compose([
                                       transforms.Resize((image_size, image_size)),
                                       transforms.RandomRotation((90, 90)),
                                       transforms.RandomRotation((180, 180)),
                                       transforms.RandomRotation((270, 270)),
                                       transforms.ToTensor()
                                      ])

        test_transform_2 = transforms.Compose([
                                       transforms.Resize((image_size, image_size)),
                                       transforms.RandomRotation((0, 360)),
                                       transforms.ToTensor()
                                      ])


        testset_1 = torchvision.datasets.CIFAR10(root=root, 
                                       train=False,
                                       download=True, 
                                       transform=test_transform_1)

        testloader_1 = torch.utils.data.DataLoader(testset_1, 
                                          batch_size=batch_size,
                                         shuffle=False)

        testset_2 = torchvision.datasets.CIFAR10(root=root, 
                                       train=False,
                                       download=True, 
                                       transform=test_transform_2)

        testloader_2 = torch.utils.data.DataLoader(testset_2, 
                                          batch_size=batch_size,
                                         shuffle=False)
    
        return (testloader_1, testloader_2)