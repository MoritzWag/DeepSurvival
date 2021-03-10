import pdb
import torch 
import torch.nn as nn

from src.architectures.utils import (conv2d_bn_block, 
                                          conv3d_bn_block,
                                          dense_layer_bn, Identity,
                                          ResidualBottleneckBlock,
                                          ResidualBlock)

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
                       out_channels=4,
                       kernel=1,
                       stride=1,
                       padding=0))
        
        self.main = nn.Sequential(*layers)
        
        self.flatten = nn.Flatten()
        if self.dimensions == 2:
            self.linear = nn.Linear(in_features=320, out_features=self.out_dim)
        else:
            self.linear = nn.Linear(in_features=640, out_features=self.out_dim)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(self.out_dim)

    def forward(self, x):
        out = self.main(x)
        out = self.flatten(out)
        out = self.linear(out)
        out = self.relu(out)
        # out = self.relu(self.bn(out))
        #out = self.tanh(out)

        return out