import torch
import numpy as np
from numpy import random
import torch.nn as nn
import torchvision
import torchvision.transforms as torchTransforms
import torchvision.models as models
from torchvision import datasets
import torch.nn.functional as F
import torch.utils as utils

from torch.utils.data import (
    Dataset,
    DataLoader,
) 

from cnn import LeNet

learning_rate = 5e-4
batch_size = 32


transforms = torchTransforms.Compose([torchTransforms.Resize((32, 32)),
                                 torchTransforms.ToTensor()])

train_fashion_mnist = datasets.FashionMNIST(root='fashion_mnist_data', 
                               train=True, 
                               transform=transforms,
                               download=True)

train_fashion_mnist_loader = DataLoader(dataset=train_fashion_mnist, 
                          batch_size=batch_size, 
                          shuffle=True)

valid_fashion_mnist = datasets.FashionMNIST(root='fashion_mnist_data', train=False, transform=transforms)

valid_fashion_mnist_loader = DataLoader(dataset=valid_fashion_mnist, batch_size=16, shuffle=False)


train_fashion_mnist_shuffled = datasets.FashionMNIST(root='fashion-mnist-data-shuffled', 
                            train=True, 
                            transform=transforms.ToTensor(),
                            target_transform=lambda y: torch.randint(0, 10, (1,)).item())

train_fashion_mnist_shuffled_loader = DataLoader(dataset=train_fashion_mnist_shuffled, 
                          batch_size=batch_size, 
                          shuffle=True, download=True)


def train_flip(model, train_loader_first, train_loader_shuffled, criterion, optimizer, epochs=30, device=None):
  model.train()
  for ep in range(epochs):
    train_loss = 0
    for images, labels in train_loader_first:
      images, labels = images.to(device), labels.to(device)
      predictions, _ = model(images) 

      loss = criterion(predictions, labels) 
      train_loss += loss.item() * images.size(0)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    epoch_loss = train_loss / len(train_loader_first.dataset)
    print("Epoch: " + str(ep+1) + " Loss: "  + str(epoch_loss))

  for ep in range(epochs):
    train_loss = 0
    for images, labels in train_loader_shuffled:
	  np.random.shuffle(labels)
      images, labels = images.to(device), labels.to(device)
      predictions, _ = model(images) 

      loss = criterion(predictions, labels) 
      train_loss += loss.item() * images.size(0)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    epoch_loss = train_loss / len(train_loader_shuffled.dataset)
    print("Epoch: " + str(ep+1) + " Loss: "  + str(epoch_loss))

  torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, "train_flip_mnist.pth")
  return model, optimizer, epoch_loss



def test_train_flip(model, test_loader, criterion, device=None):
  model.eval()
  valid_loss = 0
  correctOut = 0.0
  total = 0

  for images, labels in test_loader:
      images, labels = images.to(device), labels.to(device)
      predictions, probs = model(images)

      _, outClass = predictions.max(1)

      loss = criterion(predictions, labels) 

      valid_loss += loss.item() * images.size(0) 

      total += labels.size(0)
      correctOut += (outClass == labels).sum()

  loss = valid_loss / len(test_loader.dataset)
  accuracy = (float) (correctOut / total)
  print("Loss in test: " + str(loss))
  print("Accuracy in test: {:.2f}%".format(100*accuracy))
  print("-------------------")


# flip model trained on Fashion MNIST
flip_model_mnist = LeNet(10).to(device)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = flip_model_mnist
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

model,_,_ = train_flip(model, train_fashion_mnist_loader,
                       train_fashion_mnist_loader,
                       criterion, optimizer, epochs=30, device=device)

test_train_flip(model, train_fashion_mnist_loader, criterion, device=device)

