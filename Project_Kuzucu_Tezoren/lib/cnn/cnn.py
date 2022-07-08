import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    """ A LeNet model, supporting parameters described in the paper by Du et al."""
    
    def __init__(self, n_classes, in_channels=1, input_size=28, paper_params=False):
        super(LeNet, self).__init__()
        
        if input_size not in [28, 32]:
            raise ValueError("Linear layers are not defined for input image sizes other than 28 or 32.")
        
        if paper_params:  # Use parameters described in the paper
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=2, stride=1),
                nn.Tanh(),
                nn.AvgPool2d(kernel_size=2),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1),
                nn.Tanh(),
                nn.AvgPool2d(kernel_size=2),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1),
                nn.Tanh()
            )
            
            self.classifier = nn.Sequential(
                nn.Linear(in_features=(4608 if input_size == 32 else 3200), out_features=800),
                nn.Tanh(),
                nn.Linear(in_features=800, out_features=n_classes),
            )
        else:  # Use parameters giving better results on the MNIST dataset
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, stride=1),
                nn.Tanh(),
                nn.AvgPool2d(kernel_size=2),
                nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
                nn.Tanh(),
                nn.AvgPool2d(kernel_size=2),
                nn.Conv2d(in_channels=16, out_channels=120, kernel_size=4, stride=1),
                nn.Tanh()
            )
    
            self.classifier = nn.Sequential(
                nn.Linear(in_features=(480 if input_size == 32 else 120), out_features=84),
                nn.Tanh(),
                nn.Linear(in_features=84, out_features=n_classes),
            )
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs
