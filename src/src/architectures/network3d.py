import torch 
import numpy as np
import pdb 
import torch.nn.functional as F 

from torch import nn 


class Generator3d(nn.Module):
    """
    """
    def __init__(self, conv_dim, c_dim=2, repeat_num=6):
        super(Generator3d, self).__init__()

        layers = []
        layers.append(nn.Conv3d(1+c_dim, conv_dim, kernel_size=4, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm3d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # down-sampling layers
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv3d(curr_dim, curr_dim*2, kernel_size=3, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm3d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # bottleneck layers
        # for i in range(repeat_num):
        #     layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # up-sampling layers
        for i in range(2):
            layers.append(nn.ConvTranspose3d(curr_dim, curr_dim//2, kernel_size=3, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm3d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv3d(curr_dim, 1, kernel_size=8, stride=1, padding=3, bias=False))
        #layers.append(nn.Tanh())
        layers.append(nn.Sigmoid())
        self.main = nn.Sequential(*layers)
    

    def forward(self, x, categ):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        categ = categ.view(categ.size(0), categ.size(1), 1, 1, 1)
        categ = categ.repeat(1, 1, x.size(2), x.size(3), x.size(4))
        x = torch.cat([x, categ], dim=1)
        return self.main(x)


class Discriminator3d(nn.Module):
    """
    """
    def __init__(self, img_size=16, conv_dim=64, c_dim=2, repeat_num=6):
        super(Discriminator3d, self).__init__()
        
        layers = []
        layers.append(nn.Conv3d(1, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv3d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2
        
        kernel_size = int(img_size //  np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv3d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.linear = nn.Linear(in_features=27, out_features=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_src = torch.flatten(out_src, start_dim=1)
        out_src = self.linear(out_src)
        out_src = self.sigmoid(out_src)

        return out_src

    
class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv3d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)