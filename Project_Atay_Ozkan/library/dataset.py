from torchvision import transforms
from torchvision import datasets
import numpy as np
from .util import *
import string


class Sentence_Polarity_Dataset:
    """
        Sentence Polarity class that used to get positive and negative processed sentences
    """
    def __init__(self, pos_files, neg_files):
        """
            :param pos_files: file path that consists positive sentences
            :param neg_files: file path that consists negative sentences
        """

        # Preprocess the positive sentences
        # Reading files & Removing the non-word characters & Doing labelling
        self.positive_examples = list(open(pos_files, "r", encoding='utf-8').readlines())
        self.positive_examples = [s.strip() for s in self.positive_examples]
        self.positive_examples = [s.translate(str.maketrans('', '', string.punctuation)) for s in
                                  self.positive_examples]

        # Preprocess the negative sentences
        # Reading files & Removing the non-word characters & Doing labelling
        self.negative_examples = list(open(neg_files, "r", encoding='utf-8').readlines())
        self.negative_examples = [s.strip() for s in self.negative_examples]
        self.negative_examples = [s.translate(str.maketrans('', '', string.punctuation)) for s in
                                  self.negative_examples]

    def get_data(self):
        """
            Getting sentences and labels
            :return: sentence and label
        """
        # Merging Positive and Negative Sentences
        sentence = self.positive_examples + self.negative_examples
        sentence = [clean_str(sent) for sent in sentence]

        # Generate labels (1 means positive 0 mean negative)
        positive_labels = [1 for _ in self.positive_examples]
        negative_labels = [0 for _ in self.negative_examples]
        label = np.concatenate([positive_labels, negative_labels], 0)

        return sentence, label


class MNIST_Dataset:
    """
        MNIST class that consists of handwritten digits
    """
    def __init__(self, is_image_2d, img_size):
        """
        Initialize & Normalize MNIST dataset that is in the torchvision.datasets module
        :param is_image_2d: specified image dimension (1d or 2d)
        :param img_size: desired images size
        """

        # Function to resize the downloaded image
        if is_image_2d:
            transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
        else:
            transform = transforms.Compose([transforms.Resize((1, img_size * img_size)), transforms.ToTensor()])

        # Get train set from torchvision.datasets
        self.trainset = datasets.MNIST(
            root="./dataset",
            train=True,
            download=True,
            transform=transform)

        # Get test set from torchvision.datasets
        self.testset = datasets.MNIST(
            root="./dataset",
            train=False,
            download=True,
            transform=transform)

    def get_train_dataset(self):
        """
            Get train set of MNIST Dataset
            :return: train set
        """
        return self.trainset

    def get_test_dataset(self):
        """
            Get test set of MNIST Dataset
            :return: test set
        """
        return self.testset


class CIFAR_10_Dataset:
    """
        CIFAR-10 class that consists of 60000 32x32 colour images in 10 classes
    """
    def __init__(self):
        """
            Initialize & Normalize CIFAR-10 dataset that is in the torchvision.datasets module
        """

        #  Function to transform the downloaded image
        tensor_transform = transforms.Compose([
            transforms.Resize(size=(32, 32)),  # Resize images for training time
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            )
        ])

        # Get train set from torchvision.datasets
        self.trainset = datasets.CIFAR10(root="./dataset",
                                         train=True,
                                         download=True,
                                         transform=tensor_transform)

        # Get test set from torchvision.datasets
        self.testset = datasets.CIFAR10(root="./dataset",
                                        train=False,
                                        download=True,
                                        transform=tensor_transform)

    def get_train_dataset(self):
        """
            Get train set of CIFAR-10 Dataset
            :return: train set
        """
        return self.trainset

    def get_test_dataset(self):
        """
            Get test set of CIFAR-10 Dataset
            :return: test set
        """
        return self.testset
