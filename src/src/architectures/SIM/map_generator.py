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
            layers.append(nn.Conv2d(curr_dim, n_dim, kernel_size=5, stride=1, padding=3, bias=False))
        else:
            layers.append(nn.Conv3d(curr_dim, n_dim, kernel_size=5, stride=1, padding=3, bias=False))

        layers.append(nn.Sigmoid())
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


# class Generator2d(nn.Module):
#     """
#     """
#     def __init__(self, 
#                  n_dim, 
#                  conv_dim, 
#                  dimensions,
#                  c_dim=2, 
#                  repeat_num=6):
#         super(Generator2d, self).__init__()

#         layers = []
#         layers.append(nn.Conv2d(n_dim+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
#         layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
#         layers.append(nn.ReLU(inplace=True))

#         # down-sampling layers
#         curr_dim = conv_dim
#         for i in range(1):
#             layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
#             layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
#             layers.append(nn.ReLU(inplace=True))
#             curr_dim = curr_dim * 2

#         # bottleneck layers
#         for i in range(repeat_num):
#             layers.append(ResidualBlock2d(dim_in=curr_dim, dim_out=curr_dim))

#         # up-sampling layers
#         for i in range(1):
#             layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
#             layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
#             layers.append(nn.ReLU(inplace=True))
#             curr_dim = curr_dim // 2

#         layers.append(nn.Conv2d(curr_dim, n_dim, kernel_size=7, stride=1, padding=3, bias=False))
#         #layers.append(nn.Tanh())
#         layers.append(nn.Sigmoid())
#         self.main = nn.Sequential(*layers)

#     def forward(self, x, c):
#         # Replicate spatially and concatenate domain information.
#         # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
#         # This is because instance normalization ignores the shifting (or bias) effect.
#         c = c.view(c.size(0), c.size(1), 1, 1)
#         c = c.repeat(1, 1, x.size(2), x.size(3))
#         x = torch.cat([x, c], dim=1)
#         return self.main(x)

    

# class Generator3d(nn.Module):
#     """
#     """
#     def __init__(self, conv_dim, c_dim=2, repeat_num=6):
#         super(Generator3d, self).__init__()

#         layers = []
#         layers.append(nn.Conv3d(1+c_dim, conv_dim, kernel_size=4, stride=1, padding=3, bias=False))
#         layers.append(nn.InstanceNorm3d(conv_dim, affine=True, track_running_stats=True))
#         layers.append(nn.ReLU(inplace=True))

#         # down-sampling layers
#         curr_dim = conv_dim
#         for i in range(2):
#             layers.append(nn.Conv3d(curr_dim, curr_dim*2, kernel_size=3, stride=2, padding=1, bias=False))
#             layers.append(nn.InstanceNorm3d(curr_dim*2, affine=True, track_running_stats=True))
#             layers.append(nn.ReLU(inplace=True))
#             curr_dim = curr_dim * 2

#         # bottleneck layers
#         # for i in range(repeat_num):
#         #     layers.append(ResidualBlock3d(dim_in=curr_dim, dim_out=curr_dim))

#         # up-sampling layers
#         for i in range(2):
#             layers.append(nn.ConvTranspose3d(curr_dim, curr_dim//2, kernel_size=3, stride=2, padding=1, bias=False))
#             layers.append(nn.InstanceNorm3d(curr_dim//2, affine=True, track_running_stats=True))
#             layers.append(nn.ReLU(inplace=True))
#             curr_dim = curr_dim // 2

#         layers.append(nn.Conv3d(curr_dim, 1, kernel_size=8, stride=1, padding=3, bias=False))
#         #layers.append(nn.Tanh())
#         layers.append(nn.Sigmoid())
#         self.main = nn.Sequential(*layers)
    

#     def forward(self, x, categ):
#         # Replicate spatially and concatenate domain information.
#         # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
#         # This is because instance normalization ignores the shifting (or bias) effect.
#         categ = categ.view(categ.size(0), categ.size(1), 1, 1, 1)
#         categ = categ.repeat(1, 1, x.size(2), x.size(3), x.size(4))
#         x = torch.cat([x, categ], dim=1)
#         return self.main(x)

    

# class ResidualBlock2d(nn.Module):
#     """Residual Block with instance normalization."""
#     def __init__(self, dim_in, dim_out):
#         super(ResidualBlock, self).__init__()
#         self.main = nn.Sequential(
#             nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

#     def forward(self, x):
#         return x + self.main(x)



# class ResidualBlock3d(nn.Module):
#     """Residual Block with instance normalization."""
#     def __init__(self, dim_in, dim_out):
#         super(ResidualBlock, self).__init__()
#         self.main = nn.Sequential(
#             nn.Conv3d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.InstanceNorm3d(dim_out, affine=True, track_running_stats=True),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.InstanceNorm3d(dim_out, affine=True, track_running_stats=True))

#     def forward(self, x):
#         return x + self.main(x)