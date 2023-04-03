import torch
import torch.nn as nn
import torch.nn.functional as F

class OurModel(nn.Module):
    """Neural network used in Exercise 3 of Assignment 2."""
    
    def __init__(self, nr_classes=10):
        super(OurModel, self).__init__()
        
        # First convolution layer. The input channels are three colour channels (RGB), and 32 output channels are produced by the convolution.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        
        # Batch normalization for first convolution layer.
        self.bn1 = nn.BatchNorm2d(32)
        
        # Max pooling for first convolution layer.
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        
        # Second convolution layer.
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2)
        
        # Batch normalization for second convolution layer.
        self.bn2 = nn.BatchNorm2d(64)
        
        # Third convolution layer.
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)
        
        # Batch normalization for third convolution layer.
        self.bn3 = nn.BatchNorm2d(128)
        
        # Fully connected layer.
        self.fc1 = nn.Linear(128*4*4, 10)
        
        
    def forward(self, x):
        # 'conv1' layer
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)
        
        # 'conv2' layer
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        
        # 'conv3' layer
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        
        x = x.view(x.size(0), -1)   # flattening output from 'conv2' layer before passing it to the fully connected layers
        
        # 'fc1' layer
        x = self.fc1(x)
        
        return x
        
        
        