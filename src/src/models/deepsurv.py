import torch
import torch.nn as nn 


class DeepSurv(nn.Module):
    """
    """
    def __init__(self, config):
        super(DeepSurv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, 
                               out_channels=6, 
                               kernel_size=5)
        self.maxpool1 = nn.MaxPool()
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=6,
                               out_channels=18,
                               kernel_size=5)
        self.maxpool2 = nn.MaxPool()
        self.flatten = 
        self.fc1 = 
        self.fc2 = 
        self.fc3 = 

    def forward(self, x):
        pass

    def pass_one_batch(self, x):
        pass