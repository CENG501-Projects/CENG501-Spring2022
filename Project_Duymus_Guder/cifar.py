import torchvision
from torchvision import transforms

def loadCIFAR10(train=True):
    root = './datasets/'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))
    ])
    dataset = torchvision.datasets.CIFAR10(root=root, train=train, transform=transform, download=True)

    return dataset