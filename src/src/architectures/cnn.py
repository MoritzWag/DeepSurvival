import torch
import torch.nn as nn 
import pdb

class CNN(nn.Module):
    """
    """
    def __init__(self,
                out_dim,
                **kwargs):
        super(CNN, self).__init__(**kwargs)

        self.out_dim = out_dim
        self.conv1 = nn.Conv2d(in_channels=1, 
                               out_channels=6, 
                               kernel_size=5)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6,
                               out_channels=16,
                               kernel_size=5)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=256,
                             out_features=120)
        self.fc2 = nn.Linear(in_features=120,
                             out_features=84)
        self.fc3 = nn.Linear(in_features=84,
                             out_features=self.out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
