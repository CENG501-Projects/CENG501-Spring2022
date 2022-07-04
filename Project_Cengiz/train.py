import argparse
from model import AdaptationGenerator, FreqDensityComparator, WaveletDiscriminator
import time  # TODO: Add epoch timing

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
# from model import *
import data_utils
import os
from tensorboardX import SummaryWriter

def main():
    parser = argparse.ArgumentParser(description='Train Downscaling kernel')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--num_epochs', default=500, type=int, help='maximum number of epochs')
    parser.add_argument('--num_decay_epochs', default=250, type=int, help='number of epochs during which lr is decayed')
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--adam_beta_1', default=0.5, type=float)
    parser.add_argument('--adam_beta_2', default=0.999, type=float)
    parser.add_argument('--val_period', default=5, type=int, help='Period of running the model on the validation data')
    parser.add_argument('--val_save_path', default=None, type=str, help='path to save validation images')
    parser.add_argument('--checkpoint_save_period', default=10, type=int, help='save model once every <period>')
    parser.add_argument('--checkpoint', default=None, type=str, help='input checkpoint')
    parser.add_argument('--dataset_path_train', default="/home/baran/Documents/datasets/DIV2K/dataset_train_128", type=str, help='location of the train dataset')
    parser.add_argument('--dataset_path_val', default="/home/baran/Documents/datasets/DIV2K/dataset_val_128", type=str, help='location of the val dataset')
    _args = parser.parse_args()

    if torch.cuda.is_available():
        print("Cuda (GPU support) is available and enabled!")
        device = torch.device("cuda")
    else:
        print("Cuda (GPU support) is not available :(")
        device = torch.device("cpu")

    scales = [3.5, 1.2]

    print("# Initializing dataset")
    dataset_generator = data_utils.FreqConsistentSRDataset(_args.dataset_path_train, scale=scales[0], ext=".png")

    print("# Initializing models")
    models = [AdaptationGenerator(), FreqDensityComparator(), WaveletDiscriminator()]

    for model in models:
        model = model.to(device)

    # Optimizer parameters from the DASR paper
    print("# Initializing optimizers")
    start_decay = _args.num_epochs - _args.num_decay_epochs
    # Optimizers
    optimizers = []
    for model in models:
        opt = torch.optim.Adam(model.parameters(), lr=_args.learning_rate, betas=[_args.adam_beta_1, _args.adam_beta_2])
        optimizers.append(opt)

    # Similar to lr_scheduler.LinearLR but start after a certain epoch. From the DASR implementation
    scheduler_rule = lambda e: 1.0 if e < start_decay else 1.0 - max(0.0, float(e - start_decay) / _args.num_decay_epochs)
    
    schedulers = []
    for opt in optimizers:
        schedulers.append(torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=scheduler_rule))

    # Load from checkpoint if it exists
    if _args.checkpoint is not None:
        print(f"Trying to load checkpoint: {_args.checkpoint}")
        checkpoint = torch.load(_args.checkpoint)
        epoch_start = checkpoint['epoch'] + 1
        iteration = checkpoint['iteration'] + 1
        models[0].load_state_dict(checkpoint['model_adapt_gen_state_dict'])
        models[1].load_state_dict(checkpoint['model_fdc_state_dict'])
        models[2].load_state_dict(checkpoint['model_wd_state_dict'])
        optimizers[0].load_state_dict(checkpoint['optimizer_adapt_gen_state_dict'])
        optimizers[1].load_state_dict(checkpoint['optimizer_fdc_state_dict'])
        optimizers[2].load_state_dict(checkpoint['optimizer_wd_state_dict'])
        schedulers[0].load_state_dict(checkpoint['scheduler_adapt_gen_state_dict'])
        schedulers[1].load_state_dict(checkpoint['scheduler_fdc_state_dict'])
        schedulers[2].load_state_dict(checkpoint['scheduler_wd_state_dict'])
        print('Continuing training at epoch %d' % epoch_start)
    else:
        epoch_start = 1
        iteration = 1
    
    # Initialize tensorboard
    tensorboard_summary_path = "./tensorboard"
    if not os.path.exists(tensorboard_summary_path):
        os.makedirs(tensorboard_summary_path)
    writer = SummaryWriter(tensorboard_summary_path)
    print(f"Tensorboard summary path: {tensorboard_summary_path}")

    train(models, optimizers, schedulers, dataset_generator, scales, epoch_start, iteration, _args, writer, device, verbose=True)
    

def train(models, optimizers, schedulers, dataset_generator, scales,epoch_start, iteration, _args, writer, device, verbose=False):
    adaptation_gen, fdc, wavelet_discriminator = models
    dataloader = DataLoader(dataset_generator, batch_size=_args.batch_size, shuffle=True)
    dataset_generator_val = data_utils.FreqConsistentSRDataset(_args.dataset_path_val, scale=scales[0], ext=".png")
    dataloader_val = DataLoader(dataset_generator_val, batch_size=_args.batch_size, shuffle=False)
    scale_idx = 0
    print("Start train")
    for epoch in range(epoch_start, _args.num_epochs + 1):
        # .train() is used for converting the model to the train mode
        # It is necessary for layers like dropout, and batchnorm 
        for model in models:
            model.train()

        for data in dataloader:    
            # Get inputs from dataloader
            x, x_random, x_random_down, x_random_up = data
            x = x.to(device)
            x_random = x_random.to(device)
            x_random_down = x_random_down.to(device)
            x_random_up = x_random_up.to(device)
            
            # Generate LR image
            g_x = adaptation_gen(x) # n, 3, 128, 128

            # Generate loss outputs
            fdc_p = fdc(x_random, g_x) # n, 1, 16, 16
            fdc_d = fdc(x_random_down, g_x) # n, 1, 16, 16
            fdc_u = fdc(x_random_up, g_x) # n, 1, 16, 16
            wavelet_out = wavelet_discriminator(x_random, g_x) # n, 1
            
            # Calculate FDC loss
            fdc_loss_p = torch.abs(fdc_p - 1)
            fdc_loss_d = torch.abs(fdc_d - 1)
            fdc_loss_u = torch.abs(fdc_u - 1)
            fdc_loss = torch.mean(fdc_loss_p + fdc_loss_d + fdc_loss_u)
            fdc_loss = fdc_loss.view(-1)
            # Calculate wavelet loss = (WD(G(x)) - 1)**2
            wavelet_loss = torch.mean(torch.square(wavelet_out - 1))
            wavelet_out = wavelet_loss.view(-1)

            # Parameters are from the paper
            lambda_1 = 1
            lambda_2 = 0.001
            loss = lambda_1 * fdc_loss + lambda_2 * wavelet_loss

            # Clear previous gradients
            for opt in optimizers:
                opt.zero_grad()
                        
            # Backpropagate
            loss.backward()

            # Update the weights
            for opt in optimizers:
                opt.step()
            
            # Log losses
            writer.add_scalar("loss/fdc_loss", fdc_loss, iteration)
            writer.add_scalar("loss/wavelet_loss", wavelet_loss, iteration)
            writer.add_scalar("loss/total_loss", loss, iteration)
            iteration += 1
        
        writer.add_scalar("param/learning_rate", torch.Tensor(schedulers[0].get_last_lr()), epoch)
        writer.add_scalar("param/scale", scales[scale_idx], epoch)

        if(epoch % _args.val_period == 0):
            # Validation mode
            for model in models:
                model.eval()
            
            # TODO: calculate validation loss
            with torch.no_grad():
                for data in dataloader_val:    
                    # Get inputs from dataloader
                    x, x_random, x_random_down, x_random_up = data
                    x = x.to(device)  # Could put .to(device) functions inside Transforms
                    x_random = x_random.to(device)
                    x_random_down = x_random_down.to(device)
                    x_random_up = x_random_up.to(device)
                    
                    # Generate LR image
                    g_x = adaptation_gen(x) # n, 3, 32, 32

                    # Generate loss outputs
                    fdc_p = fdc(x_random, g_x) # n, 1, 4, 4
                    fdc_d = fdc(x_random_down, g_x) # n, 1, 4, 4
                    fdc_u = fdc(x_random_up, g_x) # n, 1, 4, 4
                    wavelet_out = wavelet_discriminator(x_random, g_x) # n, 1
                    
                    # Calculate FDC loss
                    fdc_loss_p = torch.abs(fdc_p - 1)
                    fdc_loss_d = torch.abs(fdc_d - 1)
                    fdc_loss_u = torch.abs(fdc_u - 1)
                    fdc_loss = torch.mean(fdc_loss_p + fdc_loss_d + fdc_loss_u)
                    fdc_loss = fdc_loss.view(-1)
                    # Calculate wavelet loss = (WD(G(x)) - 1)**2
                    wavelet_loss = torch.mean(torch.square(wavelet_out - 1))
                    wavelet_out = wavelet_loss.view(-1)

                    # Parameters are from the paper
                    lambda_1 = 1
                    lambda_2 = 0.001
                    loss_val = lambda_1 * fdc_loss + lambda_2 * wavelet_loss
                    
                    # Log losses
                    writer.add_scalar("loss/fdc_loss_val", fdc_loss, epoch)
                    writer.add_scalar("loss/wavelet_loss_val", wavelet_loss, epoch)
                    writer.add_scalar("loss/total_loss_val", loss_val, epoch)
            if verbose:
                print(f"Epoch {epoch} / {_args.num_epochs}: Val Loss = {loss_val.item()}")
                # TODO validation image
                if(_args.val_save_path is not None):
                    # TODO save validation images
                    pass

        if verbose: 
            print(f"Epoch {epoch} / {_args.num_epochs}: Loss = {loss.item()}")

        # Update lr if necessary
        for scheduler in schedulers:
            scheduler.step()
        
        if epoch != 0 and epoch % _args.checkpoint_save_period == 0:
            checkpoint_path = f"./checkpoints/epoch_{epoch}.tar"
            if not os.path.exists("./checkpoints"):
                os.makedirs("./checkpoints")
            state_dict = {
                "epoch": epoch,
                "iteration": iteration,
                "model_adapt_gen_state_dict": models[0].state_dict(),
                "model_fdc_state_dict": models[1].state_dict(),
                "model_wd_state_dict": models[2].state_dict(),
                "optimizer_adapt_gen_state_dict": optimizers[0].state_dict(),
                "optimizer_fdc_state_dict": optimizers[1].state_dict(),
                "optimizer_wd_state_dict": optimizers[2].state_dict(),
                "scheduler_adapt_gen_state_dict": schedulers[0].state_dict(),
                "scheduler_fdc_state_dict": schedulers[0].state_dict(),
                "scheduler_wd_state_dict": schedulers[0].state_dict()
            }
            torch.save(state_dict, checkpoint_path)

        # Experimental (magic) threshold epoch for the curriculum learning
        if scale_idx == 0 and epoch > _args.num_epochs // 2:
            print(f"Updating scale {scales[scale_idx]}->{scales[scale_idx + 1]}")
            dataset_generator.set_scale(scales[scale_idx + 1])
            dataset_generator_val.set_scale(scales[scale_idx + 1])
            dataloader = DataLoader(dataset_generator, batch_size=_args.batch_size, shuffle=True)
            dataloader_val = DataLoader(dataset_generator_val, batch_size=_args.batch_size, shuffle=True)
            scale_idx += 1       


if __name__ == "__main__":
    main()