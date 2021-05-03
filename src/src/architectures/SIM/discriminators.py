import torch 
import numpy as np 
import pdb
import torch.nn.functional as F

from torch import nn 
from torch.nn.utils import spectral_norm


class DiscriminatorSIM(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, 
                 img_size=28, 
                 conv_dim=64, 
                 c_dim=2,
                 n_dim=1, 
                 repeat_num=3, 
                 **kwargs):
        super(DiscriminatorSIM, self).__init__()
        layers = []
        layers.append(nn.Conv2d(n_dim, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(img_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=kernel_size, stride=1, padding=0, bias=False)
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        return out_src