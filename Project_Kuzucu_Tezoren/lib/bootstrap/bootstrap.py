import logging as lg
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from numpy import random
from ccnn import ccnn


random_generator = random.default_rng(501)

lg.basicConfig(format='%(levelname)s\t- %(asctime)s\t- %(message)s',
               datefmt='%m/%d/%Y-%H:%M:%S',
               level=lg.DEBUG)


def train_model_one_boot(model, sample_loader, criterion, optimizer, epochs=5, device=None):
    model.train()
    epoch_loss = 0
    
    for ep in range(epochs):
        train_loss = 0
        for images, labels in sample_loader:
            images, labels = images.to(device), labels.to(device)
            predictions, _ = model(images)
            
            loss = criterion(predictions, labels)
            train_loss += loss.item() * images.size(0)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        epoch_loss = train_loss / len(sample_loader.dataset)
        lg.info("\tEpoch: " + str(ep + 1) + " Loss: " + str(epoch_loss))
    return model, optimizer, epoch_loss


def compute_preds_test(model, test_dataloader, criterion, dist, likelihood_arr, device=None):
  model.eval()
  valid_loss = 0
  correct_out = 0.0
  total = 0
  likelihood_est = 0.0

  for images, labels in test_dataloader:
      images, labels = images.to(device), labels.to(device)
      predictions, probs = model(images)

      dist = np.append(dist, probs.cpu().detach().numpy())

      _, out_class = predictions.max(1)

      loss = criterion(predictions, labels)

      prednp = out_class.cpu().detach().numpy()
      labelnp = labels.cpu().detach().numpy()

      likelihood_est += -np.sum(labelnp*np.log(prednp+1e-9))
      likelihood_arr = np.append(likelihood_arr,
                                 -np.sum(labelnp * np.log(prednp+1e-9) / len(test_dataloader.dataset)))

      valid_loss += loss.item() * images.size(0)

      total += labels.size(0)
      correct_out += (out_class == labels).sum()

  loss = valid_loss / len(test_dataloader.dataset)
  likelihood_est /= len(test_dataloader.dataset)
  accuracy = float(correct_out / total)
  lg.info("\tLoss in test: " + str(loss))
  lg.info("\tAccuracy in test: {:.2f}%".format(100*accuracy))
  # lg.info("-------------------")
  
  return model, loss, 100*accuracy, dist, likelihood_est, likelihood_arr


def get_sample_data(train_dataset: Dataset, repetitions, x=200, batch_size=16):
    sample_indices = random_generator.integers(low=0, high=len(train_dataset), size=x)
    
    for ind in sample_indices:
        repetitions[str(ind)] += 1
    
    train_subset = Subset(train_dataset, sample_indices)
    train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True)
    
    return train_loader, repetitions


def cnn_bootstrap(model, train_data: Dataset, valid_dataloader: DataLoader, criterion, optimizer,
                  batch_size=16, X=200, conf_level=95, B=100, epochs=5, num_class=10, device=None):
    # Train model for the whole dataset for one epoch (?)
    # train_model(model, train_data, criterion)
    
    # Save init state dict
    # torch.save(model.state_dict(), "last_state_dict.pth")
    
    alpha = (100 - conf_level) / 2
    alpha_ = 100 - alpha
    
    # predictions
    dist = np.array([])
    likelihood_dist = np.array([])
    accumulated_loss = 0.0
    likelihood_sum = 0.0
    
    repeats = 0
    repetitions = dict()
    for i in range(len(train_data)):
        repetitions[str(i)] = 0
        
    # sample B times
    for b in range(B):
        sample_data, repetitions = get_sample_data(train_data, repetitions,
                                                   X, batch_size)

        lg.info(f"Begin bootstrap {b + 1}.")
        # load from previous sampling
        # model.load_state_dict("last_state_dict.pth")
        train_model_one_boot(model, sample_data, criterion, optimizer, epochs, device)
        
        # save for loading it next sampling
        torch.save(model.state_dict(), "last_state_dict.pth")
        
        lg.info(f"Test scores for bootstrap {b + 1}:")
        # get test scores
        _, loss, acc, dist, likelihood, likelihood_dist = compute_preds_test(model, valid_dataloader, criterion, dist,
                                                                             device)
        accumulated_loss += loss
        likelihood_sum += likelihood
    
    lower = np.percentile(dist, alpha)
    upper = np.percentile(dist, alpha_)
    std_err_dist = np.std(dist) / np.sqrt(len(dist))
    # std_err_like_dist = np.std(likelihood_dist) / np.sqrt(len(likelihood_dist))
    lg.info(f"(Lower bound: {lower}, Upper bound: {upper})")
    lg.info(f"Average interval length is: {(upper - lower) / (B * num_class)}")
    
    # Torch implementation, gives different results from the paper
    # lg.info(f"Average log likelihood is: {accumulated_loss / B}")
    # Numpy implementation, gives results similar to those in the paper
    lg.info(f"Average log likelihood is: {likelihood_sum / B}")
    
    lg.info(f"Standard error for average interval length: {std_err_dist}")
    # lg.info(f"Standard error for average likelihood: {std_err_like_dist}")
    for _, v in repetitions.items():
        if v > 1:
            repeats += 1
    lg.info(f"{repeats} inputs seen more than once")


def ccnn_bootstrap(model, train_data: Dataset, valid_dataloader: DataLoader, batch_size=16, X=200, conf_level=95, B=100,
                   epochs=5, num_class=10, device=None):
    alpha = (100 - conf_level) / 2
    alpha_ = 100 - alpha
    
    # predictions
    dist = np.array([])
    likelihood_dist = np.array([])
    # accumulated_loss = 0.0
    likelihood_sum = 0.0
    
    repeats = 0
    repetitions = dict()
    for i in range(len(train_data)):
        repetitions[str(i)] = 0
        
    ccnn_state = model.state
    
    # sample B times
    for b in range(B):
        sample_data, repetitions = get_sample_data(train_data, repetitions,
                                                   X, batch_size)

        lg.info(f"Begin bootstrap {b + 1}.")
        
        # Create new CCNN, use previous state.
        # It generates a new layer & trains it.
        model = ccnn.CCNN(
            train_dl=sample_data,
            test_dl=valid_dataloader,
            train_img_cnt=len(sample_data.dataset),
            test_img_cnt=len(valid_dataloader.dataset),
            from_state=ccnn_state,
            multilayer_method="ZHANG",
            n_iter=epochs,
            # device=device
            device=torch.device("cpu")  # Temporarily use CPU only
        )
        
        # Save CCNN state for the next bootstrap.
        ccnn_state = model.state

        #lg.info(f"Test scores for bootstrap {b + 1}:")
        
        # loss = ...
        # acc = ...
        
        lg.debug(f"Got CCNN probs with shape: {model.probs.shape}")
        
        # FIXME: probs is used as a ndarray. When Torch tensor conversion is done, this will create problems.
        dist = np.append(dist, model.probs)  # Array of softmax for each class
        likelihood = model.log_likelihood
        # likelihood_dist = ...
        
        # accumulated_loss += loss
        likelihood_sum += likelihood

    lower = np.percentile(dist, alpha)
    upper = np.percentile(dist, alpha_)
    std_err_dist = np.std(dist) / np.sqrt(len(dist))
    # std_err_like_dist = np.std(likelihood_dist) / np.sqrt(len(likelihood_dist))
    lg.info(f"(Lower bound: {lower}, Upper bound: {upper})")
    lg.info(f"Average interval length is: {(upper - lower) / (B * num_class)}")

    # Torch implementation, gives different results from the paper
    # lg.info(f"Average log likelihood is: {accumulated_loss / B}")
    # Numpy implementation, gives results similar to those in the paper
    lg.info(f"Average log likelihood is: {likelihood_sum / B}")

    lg.info(f"Standard error for average interval length: {std_err_dist}")
    # lg.info(f"Standard error for average likelihood: {std_err_like_dist}")
    for _, v in repetitions.items():
        if v > 1:
            repeats += 1
    lg.info(f"{repeats} inputs seen more than once")
