import pdb
import torch 
import torch.nn as nn

from src.architectures.ADNI.utils import (conv2d_bn_block, 
                                          conv3d_bn_block,
                                          dense_layer_bn, Identity,
                                          ResidualBottleneckBlock,
                                          ResidualBlock)


class NormalNet2D(nn.Module):
    """
    """
    def __init__(self,
                 out_dim, 
                 n_dim,
                 init_filters=32,
                 **kwargs):
        nf = init_filters
        super(NormalNet2D, self).__init__()
        
        self.out_dim = out_dim 
        self.n_dim = n_dim 
        self.conv1 = nn.Conv2d(in_channels=self.n_dim,
                               out_channels=16,
                               kernel_size=5)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=16,
                               out_channels=32,
                               kernel_size=5)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=32, 
                               out_channels=64, 
                               kernel_size=5)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = nn.Conv2d(in_channels=64, 
                               out_channels=128, 
                               kernel_size=5)
        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=2048, 
                             out_features=256)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=256,
                             out_features=64)
        self.relu6 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=64,
                             out_features=self.out_dim)

    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)
        x = self.flatten(x)
        x = self.relu5(self.fc1(x))
        x = self.relu6(self.fc2(x))
        x = self.fc3(x)

        return x


class Classifier(nn.Module):
    """
    """
    def __init__(self, 
                 out_dim,
                 in_channels,
                 dimensions,
                 params,
                 **kwargs):
        super(Classifier, self).__init__()
    

    def forward(self, x):
        pass




class ResidualClassifier(nn.Module):
    """
    """
    def __init__(self,
                 deep_params,
                 **kwargs):
        super(ResidualClassifier, self).__init__()

        self.params = deep_params
        self.out_dim = self.params['out_dim']
        self.in_channels = self.params['in_channels']
        self.dimensions = self.params['dimensions']
        self.bottleneck = self.params.get('bottleneck', False)

        hidden_channels = self.params.get('hidden_channels', [16, 32, 64, 128])
        kernel_sizes = self.params.get('kernel_sizes', [5, 3, 3, 3])
        strides = self.params.get('strides', [2, 2, 2, 2])
        padding = self.params.get('padding', [2, 1, 1, 1])

        if self.dimensions == 2:
            conv_block = conv2d_bn_block
        else:
            conv_block = conv3d_bn_block

        layers = []
        layers.append(conv_block(in_channels=self.in_channels, 
                       out_channels=hidden_channels[0],
                       kernel=kernel_sizes[0],
                       stride=strides[0],
                       padding=padding[0]))
        
        layers.append(ResidualBlock(hidden_channels[0], self.dimensions))
        layers.append(conv_block(in_channels=hidden_channels[0],
                       out_channels=hidden_channels[1],
                       kernel=kernel_sizes[1],
                       stride=strides[1],
                       padding=padding[1]))

        layers.append(ResidualBlock(hidden_channels[1], self.dimensions))
        layers.append(conv_block(in_channels=hidden_channels[1],
                       out_channels=hidden_channels[2],
                       kernel=kernel_sizes[2],
                       stride=strides[2],
                       padding=padding[2]))

        if self.bottleneck:
            layers.append(ResidualBottleneckBlock(hidden_channels[2], self.dimensions))
        else:
            layers.append(ResidualBlock(hidden_channels[2], self.dimensions))
        layers.append(conv_block(in_channels=hidden_channels[2],
                       out_channels=hidden_channels[3],
                       kernel=kernel_sizes[3],
                       stride=strides[3],
                       padding=padding[3]))
        
        if self.bottleneck:
            layers.append(ResidualBottleneckBlock(hidden_channels[3], self.dimensions))
        else:
            layers.append(ResidualBlock(hidden_channels[3], self.dimensions))
        layers.append(conv_block(in_channels=hidden_channels[3],
                       out_channels=16,
                       kernel=1,
                       stride=1,
                       padding=0))
        
        self.main = nn.Sequential(*layers)
        
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=1024, out_features=self.out_dim)
        self.tanh = nn.Tanh()



    def forward(self, x):
        out = self.main(x)
        out = self.flatten(out)
        out = self.linear(out)
        out = self.tanh(out)

        return out