import torch 
import numpy as np 
import pdb
import torch.nn.functional as F

from torch import nn 
from src.architectures.utils import (ACTIVATION,
                                     conv2d_block, 
                                     conv2d_bn_block, 
                                     dense_layer_bn, 
                                     conv3d_block,
                                     conv3d_bn_block)


class DiscriminatorADNI(nn.Module):
    """
    """
    def __init__(self, 
                 n_dim=1, 
                 conv_dim=16, 
                 batch_norm=True,
                 dimensions=2,
                 repeat_num=5,
                 activation=ACTIVATION,
                 **kwargs):
        super(DiscriminatorADNI, self).__init__()

        if dimensions == 2:
            conv_block = conv2d_bn_block if batch_norm else conv2d_block
        else:
            conv_block = conv3d_block
        max_pool = nn.MaxPool2d(2) if int(dimensions) is 2 else nn.MaxPool3d(2)
        act = activation

        self.main = []

        self.conv1 = nn.Sequential(
            conv_block(n_dim, conv_dim, activation=act),
            conv_block(conv_dim, conv_dim, activation=act)
        )
        self.conv2 = nn.Sequential(
            max_pool,
            conv_block(conv_dim, 2*conv_dim, activation=act),
            conv_block(2*conv_dim, 2*conv_dim, activation=act)
        )

        self.conv3 = nn.Sequential(
            max_pool,
            conv_block(2*conv_dim, 4*conv_dim, activation=act),
            conv_block(4*conv_dim, 4*conv_dim, activation=act)
        )

        self.conv4 = nn.Sequential(
            max_pool,
            conv_block(4*conv_dim, 2*conv_dim, activation=act),
            conv_block(2*conv_dim, 2*conv_dim, activation=act)
        )

        self.conv5 = nn.Sequential(
            max_pool,
            conv_block(2*conv_dim, conv_dim, activation=act),
            conv_block(conv_dim, conv_dim, activation=act)
        )
        
        self.flatten = nn.Flatten()
        #self.linear = nn.Linear(in_features=1024, out_features=1)
        self.linear = nn.Linear(in_features=1280, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.flatten(x)
        x = self.linear(x)
        out = self.sigmoid(x)

        return out
