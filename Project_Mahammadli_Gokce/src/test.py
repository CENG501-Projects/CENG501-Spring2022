import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import RRL
from models import LeNet5, ResNet18, ResidualBlock
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import classification_report
from dataloader import prepare_loader

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
    
    parser = argparse.ArgumentParser(description='Model Testing Program.')
    optional = parser._action_groups.pop() 
    required = parser.add_argument_group('required arguments')
    
    required.add_argument(
        '--model_name',
        help='name of the model to test: lenet5 or resnet18.',
        type=str)
    
    required.add_argument(
        '--model_path',
        help='Path to the saved, trained model.',
        type=str)
    
    required.add_argument(
        '--image_size',
        help='height and width of the input image. should be divisable by three and the same as in the training loader.',
        type=int)
    
    required.add_argument(
        '--batch_size',
        help='size of each batch for training. should be the same as in the training loader', 
        default=32, 
        type=int)

    
    optional.add_argument(
        '--report_destination',
        help='path to the folder to store the clasification report',
        default='./',
        type=str)
    
    parser._action_groups.append(optional)
    args = parser.parse_args()
    print(parser.print_help())
    return args

def print_results(
    model_name: str,
    model_path : str,
    test_loader: DataLoader, 
    report_destination: str,
    rrl : bool) -> None:
    """
    Prints out classification report based on the given model and test loader, and save the results at given path
    
    Parameters
    ---------
    model_name : str
        Name of the model to train, lenet5 or resnet18
    model_path : str:
        Path to saved, trained model
    test_loader : DataLoader
        Data loader for the test set
    report_destination : str
        Path to store the classification report
    rrl : bool
        Indicates whether model has been trained with RRL layer or not
    """
    
    if model_name == 'lenet5':
        model = LeNet5(rrl).to(device)
        
    elif model_name == 'resnet18':
        model = ResNet18(ResidualBlock, rrl=rrl).to(device)
        
    else:
        raise Exception("Wrong model name, please include either 'lenet5' or 'resnet18'")
    
    model = torch.load(model_path)
    model.eval()
    
    y_pred_ls = []
    y_test_ls = []

    with torch.no_grad():
        for _, (X_test, y_test) in enumerate(test_loader):
            y_test_ls.extend(y_test.tolist())
            X_test = X_test.to(device)
            y_test = y_test.to(device)

            y_test_pred = model(X_test)

            predicted = torch.max(y_test_pred.data, 1)[1].tolist()
            y_pred_ls.extend(predicted)
            
    report = classification_report(y_pred_ls, y_test_ls)
    with open(report_destination + model_name + '_classiciation_report.txt', 'a+') as file:
        file.write(report)
        
    print(report)
    
    
if __name__ == '__main__':
    args = args_from_terminal()
    """
    testloader_1, testloader_2 = prepare_loader(
        batch_size=args.batch_size,
        image_size=args.image_size,
        train_or_test='test')
    
    
    print("Results for CIFAR-10-rot test")
    print_results(
        model_name=args.model_name,
        model_path=args.model_path,
        test_loader=testloader_1,
        report_destination=args.report_destination)

    print("\n Results for CIFAR-10-rot+ test")
    print_results(
        model_name=args.model_name,
        model_path=args.model_path,
        test_loader=testloader_2,
        report_destination=args.report_destination)
    """