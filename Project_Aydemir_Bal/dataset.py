import torch
import torchvision


class TransformsSimCLR:
    """
    from https://github.com/Spijkervet/SimCLR/blob/cd85c4366d2e6ac1b0a16798b76ac0a2c8a94e58/simclr/modules/transformations/simclr.py
    A stochastic data augmentation module that transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, size):
        s = 1
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        self.train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(size=size),
                torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                torchvision.transforms.RandomApply([color_jitter], p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.ToTensor(),
            ]
        )

        self.test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=size),
                torchvision.transforms.ToTensor(),
            ]
        )

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)


def get_dataset(args):
    if args.dataset == "CIFAR10":

        train_dataset = torchvision.datasets.CIFAR10(
            root=args.data_path, train=True, download=True, transform=TransformsSimCLR(size=32))
        test_dataset = torchvision.datasets.CIFAR10(
            root=args.data_path, train=False, download=True, transform=TransformsSimCLR(size=32))
        train_dataset = torch.utils.data.ConcatDataset(
            [train_dataset, test_dataset])

        train_dataset2 = torchvision.datasets.CIFAR10(
            root=args.data_path, train=True, download=False, transform=TransformsSimCLR(size=32).test_transform)
        test_dataset = torchvision.datasets.CIFAR10(
            root=args.data_path, train=False, download=False, transform=TransformsSimCLR(size=32).test_transform)
        test_dataset = torch.utils.data.ConcatDataset(
            [train_dataset2, test_dataset])

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

        return train_dataloader, test_dataloader
