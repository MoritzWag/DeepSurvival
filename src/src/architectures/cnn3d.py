import torch 
import torch.nn as nn 
import pdb 


class CNN3d(nn.Module):
    """
    """
    def __init__(self,
                 out_dim,
                 n_dim,
                 **kwargs):
        super(CNN3d, self).__init__(**kwargs)

        self.out_dim = out_dim
        self.n_dim = n_dim
        self.conv1 = nn.Conv3d(in_channels=n_dim,
                               out_channels=6,
                               kernel_size=5)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool3d(kernel_size=2)
        self.conv2 = nn.Conv3d(in_channels=6,
                               out_channels=16,
                               kernel_size=5)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool3d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=1024,
                             out_features=512)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=512,
                             out_features=120)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=120,
                             out_features=84)
        self.relu5 = nn.ReLU()
        self.fc4 = nn.Linear(in_features=84,
                             out_features=self.out_dim)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)

        return x
