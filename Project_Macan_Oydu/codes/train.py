import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from ShadowRemoverNetwork import ShadowRemoverNetwork
from dataset import dataset

if torch.cuda.is_available():
  print("Cuda (GPU support) is available and enabled!")
  device = torch.device("cuda")
  torch.cuda.empty_cache()
  torch.cuda.set_per_process_memory_fraction(fraction=1., device=None)
else:
  print("Cuda (GPU support) is not available :(")
  device = torch.device("cpu")

def loss(I_ns_history, A_history, M, gt):
    loss = 0
    gamma_d = torch.arange(5)
    gamma_reg = 0.000001
    for batch in range(gt.shape[0]):
      for i in range(len(I_ns_history)):
        loss += torch.mean(gamma_d[i]*torch.norm(I_ns_history[i][batch,:,:,:]-gt[batch,:,:,:])+gamma_reg*torch.norm((1-M[batch,:,:,:])*A_history[i][batch,:,:,:]))
    loss = loss/gt.shape[0]
    return loss

def train(model, criterion, optimizer, scheduler, epochs, dataloader, verbose=True):
  loss_history = []
  for epoch in range(epochs):
    for i, data in enumerate(dataloader, 0):
      
      # Our batch:
      I_s, M, gt = data
      I_s = I_s.to(device)
      M = M.to(device)
      gt = gt.to(device)
      print(gt)
      # zero the gradients as PyTorch accumulates them
      optimizer.zero_grad()
      
      # Obtain the outputs
      I_ns_history, A_history = model(I_s, M)
      
      # Calculate loss
      loss = criterion(I_ns_history, A_history, M, gt)
      
      # Backpropagate
      loss.backward()

      # Update the weights
      optimizer.step()

      # Update the learning rate
      scheduler.step()

      loss_history.append(loss.item())
      
    if verbose: print(f'Epoch {epoch} / {epochs}: avg. loss of last 5 iterations {np.sum(loss_history[:-6:-1])/5}')

  return loss_history

if __name__ == "__main__":
  learning_rate = 0.00005
  batch_size = 2
  epochs = 150
  transformation = torchvision.transforms.Compose((torchvision.transforms.Resize([128,128]),torchvision.transforms.ToTensor()))
  train_dataset = dataset(image_dir="ISTD_Dataset/train/train_A", mask_dir="ISTD_Dataset/train/train_B", groud_truth_dir="ISTD_Dataset/train/train_C", transformation=transformation)
  train_loader = DataLoader(train_dataset,shuffle=True,batch_size=batch_size,num_workers=1)

  model = ShadowRemoverNetwork()
  model.to(device)
  
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150) #T_max is not given in the paper
  train(model, loss, optimizer, scheduler, epochs, train_loader, verbose=True)
  torch.save(model.state_dict(), 'ShadowRemoverNetwork.ckpt')