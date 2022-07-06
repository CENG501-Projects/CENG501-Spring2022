import torchvision
from torchvision import transforms

def loadCIFAR10(train=True):
    root = './datasets/'
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(root=root, train=train, transform=transform, download=True)

    return dataset