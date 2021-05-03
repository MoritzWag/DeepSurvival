import torch 
import numpy as np 
import pdb
import torch.nn.functional as F

from torch import nn 
from src.architectures.utils import (conv2d_intnorm_block,
                                     conv3d_intnorm_block,
                                     conv2d_transposed_intnorm_block,
                                     conv3d_transposed_intnorm_block,
                                     ResidualBlockInstanceNorm)


class GeneratorSIM(nn.Module):
    """
    """
    def __init__(self,
                 n_dim,
                 conv_dim,
                 dimensions,
                 c_dim=2,
                 repeat_num=6):
        super(GeneratorSIM, self).__init__()

        if dimensions == 2:
            conv_block = conv2d_intnorm_block
            conv_transpose_block = conv2d_transposed_intnorm_block
        else:
            conv_block = conv3d_intnorm_block
            conv_transpose_block = conv3d_transposed_intnorm_block
        
        layers = []
        layers.append(conv_block(in_channels=n_dim+c_dim,
                                 out_channels=conv_dim,
                                 kernel_size=7,
                                 stride=1,
                                 padding=3,
                                 bias=False))

        # down sampling layers
        curr_dim = conv_dim 
        for i in range(1):
            layers.append(conv_block(curr_dim, curr_dim*2, kernel_size=5, stride=2, padding=1, bias=False))
            curr_dim = curr_dim * 2
        
        # bottleneck layers
        for i in range(repeat_num):
            layers.append(ResidualBlockInstanceNorm(dim_in=curr_dim, dim_out=curr_dim, dimensions=dimensions))

        # up-sampling layers
        for i in range(1):
            layers.append(conv_transpose_block(in_channels=curr_dim,
                                              out_channels=curr_dim//2,
                                              kernel_size=4,
                                              stride=2,
                                              padding=1,
                                              bias=False))
            curr_dim = curr_dim // 2
        
        if dimensions == 2:
            #layers.append(nn.Conv2d(curr_dim, n_dim, kernel_size=5, stride=1, padding=3, bias=False))
            layers.append(nn.Conv2d(curr_dim, n_dim, kernel_size=3, stride=1, padding=2, bias=False))
        else:
            layers.append(nn.Conv3d(curr_dim, n_dim, kernel_size=5, stride=1, padding=3, bias=False))
        
        #layers.append(nn.Sigmoid())
        self.main = nn.Sequential(*layers)
    
    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        out = self.main(x)
        return out 