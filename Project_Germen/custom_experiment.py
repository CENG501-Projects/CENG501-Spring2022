
import argparse
from multiprocessing.dummy import active_children

parser = argparse.ArgumentParser(description="Model Arguments")

parser.add_argument(
    "--model",
    type=str,
    default = "CONV_6_SM", 
    help = "Model name"
)

parser.add_argument(
    "--K",
    type=int,
    default = 8,
    help = "Number of discrete weights"
)

parser.add_argument(
    "--selector",
    type=str,
    default = "GS",
    help = "Selector function"
)

parser.add_argument(
    "--lr",
    type = float,
    default = 0.2,
    help = "Learning rate"
)

parser.add_argument(
    "--epochs",
    type=int,
    default = 200,
    help = "Number of epochs"
)

args = parser.parse_args()

import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as TF
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import math
from models_layers_trainers.SM_Models import CONV_2_SM, CONV_2,  CONV_4_SM, CONV_4,  CONV_6_SM, CONV_6
import copy

random.seed(501)
np.random.seed(501)    
torch.manual_seed(501)

if torch.cuda.is_available():
  print("Cuda (GPU support) is available and enabled!")
  device = torch.device("cuda")
else:
  print("Cuda (GPU support) is not available :(")
  device = torch.device("cpu")


#######################################################
########### Loading datasets
#######################################################

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 128

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

validset, trainset = torch.utils.data.random_split(trainset, [5000, 45000])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size,
                                          shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
                                       
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#######################################################
########### Validation
#######################################################
def validation():
    correct = 0
    total = 0

    with torch.no_grad():
        for data in validloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

#######################################################
########### Training function - SLOT MACHINE
#######################################################

def train_SM(model, criterion, optimizer, epochs, dataloader, lr, verbose=True):
  """
    Define the trainer function. We can use this for training any model.
    The parameter names are self-explanatory.

    Returns: the loss history.
  """
  model.to(device)
  loss_history = []
  acc_history = []
  for epoch in range(epochs):
    for i, data in enumerate(dataloader, 0):    
      
      # Our batch:
      inputs, labels = data
      inputs = inputs.to(device)
      labels = labels.to(device)

      # zero the gradients as PyTorch accumulates them
      optimizer.zero_grad()

      # Obtain the scores
      outputs = model(inputs)

      # Calculate loss
      loss = criterion(outputs.to(device), labels)

      # Backpropagate
      loss.backward()

      # Update the scores
      model.update_scores(lr) # MOST IMPORTANT CHANGE 

      loss_history.append(loss.item())
    
    if verbose: print(f'Epoch {epoch} / {epochs}: avg. loss of last 5 iterations {np.sum(loss_history[:-6:-1])/5}')

    acc_history.append(validation())
    best_acc = max(acc_history)
    best_epoch = acc_history.index(best_acc)
    if best_epoch == epoch:
        best_model = copy.deepcopy(model)

  return loss_history, acc_history, best_model

#######################################################
########### Training function - NORMAL
#######################################################

def train(model, criterion, optimizer, epochs, dataloader, verbose=True):
  """
    Define the trainer function. We can use this for training any model.
    The parameter names are self-explanatory.

    Returns: the loss history.
  """
  loss_history = []
  acc_history = []
  for epoch in range(epochs):
    for i, data in enumerate(dataloader, 0):    
      
      # Our batch:
      inputs, labels = data
      inputs = inputs.to(device)
      labels = labels.to(device)

      # zero the gradients as PyTorch accumulates them
      optimizer.zero_grad()

      # Obtain the scores
      outputs = model(inputs)

      # Calculate loss
      loss = criterion(outputs.to(device), labels)

      # Backpropagate
      loss.backward()

      # Update the weights
      optimizer.step()

      loss_history.append(loss.item())
    
    if verbose: print(f'Epoch {epoch} / {epochs}: avg. loss of last 5 iterations {np.sum(loss_history[:-6:-1])/5}')

    acc_history.append(validation())
    best_acc = max(acc_history)
    best_epoch = acc_history.index(best_acc)
    if best_epoch == epoch:
        best_model = copy.deepcopy(model)

  return loss_history, acc_history, best_model




###### Setting the model
MN = args.model # Extracting model name

if MN == "CONV_2_SM":
    model = CONV_2_SM(args.K,args.selector)
elif MN == "CONV_2":
    model = CONV_2()
if MN == "CONV_4_SM":
    model = CONV_4_SM(args.K,args.selector)
elif MN == "CONV_4":
    model = CONV_4()
if MN == "CONV_6_SM":
    model = CONV_6_SM(args.K,args.selector)
elif MN == "CONV_6":
    model = CONV_6()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)

model = model.to(device)
epochs = args.epochs

####### Setting the trainer
if MN[-2:] == "SM":
    loss_history, acc_history, best_model = train_SM(model, criterion, optimizer, epochs, trainloader, lr = args.lr) 
else:
    loss_history, acc_history, best_model = train(model, criterion, optimizer, epochs, trainloader)

# Disclaimer: This code piece is taken from PyTorch examples.

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = best_model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy_2 = correct / total 

print('Accuracy out of 5000 images is ' + str(accuracy))
best_acc = max(acc_history)
best_epoch = acc_history.index(best_acc)
print("Best epoch for validation and its test accuracy: " + str(best_epoch) + " - " + str(accuracy_2))

model.accuracy = accuracy
model.loss_history = loss_history
model.acc_history = acc_history
model.validbest = (best_epoch, best_acc)
model.best_model = best_model

import pickle
model_file_name = args.model + "_K_" + str(args.K) + "_Selector_" + args.selector + "_LR_" + str(args.lr) + "_Epochs_" + str(args.epochs) + "_ACC_" + str(accuracy)
with open(model_file_name, "wb") as pf:
    pickle.dump(model,pf)
