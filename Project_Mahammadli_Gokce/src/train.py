import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from models import LeNet5, ResNet18, ResidualBlock
from dataloader import prepare_loader
from time import perf_counter
from functools import wraps
from typing import Callable, TypeVar
from typing_extensions import ParamSpec

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def args_from_terminal() -> argparse.Namespace:
    """
    Allows to take arguments from terminal with corresponding flags and stors in args variable.
    Parameters
    ----------
    None
    
    Returns
    -------
    args : argparse.Namespace
        Argparse Namespace object containing the arguments with corresponding values
    """

    parser = argparse.ArgumentParser(description='Model Training Program.')
    optional = parser._action_groups.pop() 
    required = parser.add_argument_group('required arguments')
    
    required.add_argument(
        '--rrl',
        help='use this command to train model with RRL layer.',
        action='store_true')
    
    required.add_argument(
        '--no-rrl',
        help='use this command to train model without RRL layer.',
        action='store_false'
    )
    
    required.set_defaults(feature=False)
    
    required.add_argument(
        '--model_name',
        help='name of the model to train: lenet5 or resnet18.',
        default='lenet5',
        type=str
    )
    
    optional.add_argument(
        '--image_size',
        help='height and width of the input image. should be divisable by three.',
        default=33,
        type=int)
    
    optional.add_argument(
        '--batch_size',
        help='size of each batch for training.', 
        default=128, 
        type=int)

    optional.add_argument(
        '--epochs',
        help='number of epochs to train.', 
        default=10, 
        type=int)

    optional.add_argument(
        '--lr',
        help='learning rate for the optimizer.', 
        default=0.001, 
        type=float)

    optional.add_argument(
        '--model_destination',
        default='./',
        type=str)
    
    optional.add_argument(
        '--plot', 
        help='pass it without any value to draw training loss and accuracy.', 
        action='store_true')

    optional.add_argument(
        '--plot_destination',
        help='if you have passed --plot_training argument, use this flag with corresponsing path to store the plot.', 
        default='./',
        type=str)
    
    optional.add_argument(
        '--verbose',
        help='If True, loss and accuracy during training will be printed out. default True',
        action='store_true')
    

    parser._action_groups.append(optional)
    args = parser.parse_args()
    return args

# decorator for calculating training time

P = ParamSpec('P')
R = TypeVar('R')

def timer(func: Callable[P, R]) -> Callable[P, R]:
  """Calculates the runtime of the the given function"""
  
  @wraps(func)
  def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
    start = perf_counter()
    func(*args, **kwargs)
    train_time = perf_counter() - start
    return train_time
  return wrapper

@timer
def trainer(model_name: str,
            epochs: int, 
            lr : float,
            image_size : int,
            batch_size : int,
            rrl: bool=False,
            verbose: bool=True,
            plot: bool=True,
            model_destination: str='./',
            plot_destination: str='./') -> float:
    """
    Training given model architecture
    
    Parameters
    ---------
    model_name : str
        Name of the model to train, lenet5 or resnet18
    epochs : int
        Number of epochs to train
    lr : float
        Learning rate for the optimizer
    image_size : int
        Value to resize images during loading
    batch_size : int
        Size of each batch
    rrl : bool
        If set True, RRL layer will be used inside the chosen model
    verbose : bool
        If True, loss and accuracy during training will be printed out
    plot : bool
        If True, training accuracy and loss will be plotted
    model_destination : str
        Path to store the trained model
    plot_destinatinon : str
        Path to store the plots
    
    Returns:
    ------
    train_time : float
        Time spent for the traning from the decorator
    """
    
    train_losses = []
    train_accuracy = []
    
    if model_name == 'lenet5':
        model = LeNet5(rrl).to(device)
        
    elif model_name == 'resnet18':
        model = ResNet18(ResidualBlock, rrl=rrl).to(device)
        
    else:
        raise Exception("Wrong model name, please include either 'lenet5' or 'resnet18'")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    num_of_images, num_of_batches, train_loader = prepare_loader(
        image_size=image_size,
        batch_size=batch_size,
        train_or_test='train')

    for epoch in range(epochs):
        train_corr = 0

        # Run the training batches
        for b, (X_train, y_train) in enumerate(train_loader):
            b += 1
            X_train = X_train.to(device)
            y_train = y_train.to(device)

            # Apply the model
            y_pred = model(X_train)
            loss = criterion(y_pred.to(device), y_train)
        
            # the number of correct predictions
            predicted = torch.max(y_pred.data, 1)[1]
            batch_corr = (predicted == y_train).sum()
            train_corr += batch_corr.item()
                
            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print interim results
            if verbose and b % 100 == 0:
                print(f"Epoch [{epoch}/{epochs}], Step [{b}/{num_of_batches}] Loss: {loss.item():.2f} training accuracy: {(train_corr * 100 / (b * batch_size)):.3f}")
                
        train_losses.append(loss.item())
        train_accuracy.append((train_corr / num_of_images) * 100)
    
    if rrl:
        torch.save(model, model_destination + model_name + '_with_rrl.pth')
    else:
        torch.save(model, model_destination + model_name + '_without_rrl.pth')

    if plot:
        plt.plot(train_accuracy)
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.savefig(
            plot_destination + model_name + '_accuracy.png', 
            facecolor='white', 
            edgecolor='none')
        
        plt.show()

        plt.plot(train_losses)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig(
            plot_destination + model_name + "_loss.png",
            facecolor='white',
            edgecolor='none')
        
        plt.show()
        

if __name__ == '__main__':
    args = args_from_terminal()
    train_time = trainer(
        model_name=args.model_name,
        epochs=args.epochs,
        lr=args.lr,
        image_size=args.image_size,
        batch_size=args.batch_size,
        rrl = args.rrl,
        verbose = args.verbose,
        plot=args.plot,
        model_destination=args.model_destination,
        plot_destination=args.plot_destination)
    
    print(f'{args.model_name} took {(train_time / 60):.3f} minutes to train.')