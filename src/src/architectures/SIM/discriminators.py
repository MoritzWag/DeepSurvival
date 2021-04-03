import torch 
import numpy as np 
import pdb
import torch.nn.functional as F

from torch import nn 
from torch.nn.utils import spectral_norm



# class DiscriminatorSIM(nn.Module):
#     def __init__(self,
#                  img_size,
#                  n_dim=1,
#                  conv_dim=64,
#                  dimensions=2,
#                  c_dim=2, 
#                  repeat_num=6):
#         super(DiscriminatorSIM, self).__init__()

#         self.model = nn.Sequential(
#             nn.Linear(int(img_size*img_size), 512),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(512, 256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(256, 1),
#             nn.Sigmoid(),
#         )

#     def forward(self, img):
#         img_flat = img.view(img.size(0), -1)
#         validity = self.model(img_flat)

#         return validity


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

# class DiscriminatorSIM(nn.Module):
#     """
#     """
#     def __init__(self,
#                  img_size=28,
#                  n_dim=3,
#                  conv_dim=64,
#                  dimensions=2,
#                  c_dim=2,
#                  repeat_num=6):
#         super(DiscriminatorSIM, self).__init__()

#         if dimensions == 2:
#             conv = nn.Conv2d
#         else:
#             conv = nn.Conv3d
        
#         layers = []
#         layers.append(spectral_norm(conv(n_dim, conv_dim, kernel_size=4, stride=2, padding=1)))
#         layers.append(nn.LeakyReLU(0.01))

#         curr_dim = conv_dim
#         for i in range(1, repeat_num):
#             layers.append(spectral_norm(conv(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)))
#             layers.append(nn.LeakyReLU(0.01))
#             # layers.append(conv(curr_dim, curr_dim*2,  kernel_size=4, stride=2, padding=1))
#             # layers.append(nn.LeakyReLU(0.01))
#             curr_dim = curr_dim * 2
        
#         kernel_size = int(img_size // np.power(2, repeat_num))
#         self.main = nn.Sequential(*layers)
#         # self.conv1 = conv(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv1 = spectral_norm(conv(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False))
#         if img_size == 28:
#             self.linear = nn.Linear(in_features=9, out_features=1)
#         else:
#             self.linear = nn.Linear(in_features=64, out_features=1)
#         self.sigmoid = nn.Sigmoid()
    
#     def forward(self, x):
#         h = self.main(x)
#         out_src = self.conv1(h)
#         out_src = torch.flatten(out_src, start_dim=1)
#         out_src = self.linear(out_src)
#         out_src = self.sigmoid(out_src)

#         return out_src


# class DiscriminatorSIM(nn.Module):
#     """
#     """
#     def __init__(self,
#                  img_size=28,
#                  n_dim=3,
#                  conv_dim=64,
#                  dimensions=2,
#                  c_dim=2,
#                  repeat_num=6):
#         super(DiscriminatorSIM, self).__init__()

#         if dimensions == 2:
#             conv = nn.Conv2d
#         else:
#             conv = nn.Conv3d
        
#         layers = []
#         layers.append(conv(n_dim, conv_dim, kernel_size=4, stride=2, padding=1))
#         layers.append(nn.LeakyReLU(0.01))

#         curr_dim = conv_dim
#         for i in range(1, repeat_num):
#             layers.append(conv(curr_dim, curr_dim*2,  kernel_size=4, stride=2, padding=1))
#             layers.append(nn.LeakyReLU(0.01))
#             curr_dim = curr_dim * 2
        
#         kernel_size = int(img_size // np.power(2, repeat_num))
#         self.main = nn.Sequential(*layers)
#         self.conv1 = conv(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
#         if img_size == 28:
#             self.linear = nn.Linear(in_features=9, out_features=1)
#         else:
#             self.linear = nn.Linear(in_features=64, out_features=1)
#         self.sigmoid = nn.Sigmoid()
    
#     def forward(self, x):
#         h = self.main(x)
#         out_src = self.conv1(h)
#         out_src = torch.flatten(out_src, start_dim=1)
#         out_src = self.linear(out_src)
#         out_src = self.sigmoid(out_src)

#         return out_src

# class Discriminator2d(nn.Module):
#     """
#     """
#     def __init__(self, 
#                  img_size=28, 
#                  n_dim=3,
#                  conv_dim=64, 
#                  c_dim=2, 
#                  repeat_num=6):
#         super(Discriminator2d, self).__init__()

#         layers = []
#         layers.append(nn.Conv2d(n_dim, conv_dim, kernel_size=4, stride=2, padding=1))
#         layers.append(nn.LeakyReLU(0.01))

#         curr_dim = conv_dim
#         for i in range(1, repeat_num):
#             layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
#             layers.append(nn.LeakyReLU(0.01))
#             curr_dim = curr_dim * 2
        
#         kernel_size = int(img_size //  np.power(2, repeat_num))
#         self.main = nn.Sequential(*layers)
#         self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
#         self.linear = nn.Linear(in_features=9, out_features=1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         h = self.main(x)
#         out_src = self.conv1(h)
#         out_src = torch.flatten(out_src, start_dim=1)
#         out_src = self.linear(out_src)
#         out_src = self.sigmoid(out_src)

#         return out_src


# class Discriminator3d(nn.Module):
#     """
#     """
#     def __init__(self, img_size=16, conv_dim=64, c_dim=2, repeat_num=6):
#         super(Discriminator3d, self).__init__()
        
#         layers = []
#         layers.append(nn.Conv3d(1, conv_dim, kernel_size=4, stride=2, padding=1))
#         layers.append(nn.LeakyReLU(0.01))

#         curr_dim = conv_dim
#         for i in range(1, repeat_num):
#             layers.append(nn.Conv3d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
#             layers.append(nn.LeakyReLU(0.01))
#             curr_dim = curr_dim * 2
        
#         kernel_size = int(img_size //  np.power(2, repeat_num))
#         self.main = nn.Sequential(*layers)
#         self.conv1 = nn.Conv3d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
#         self.linear = nn.Linear(in_features=27, out_features=1)
#         self.sigmoid = nn.Sigmoid()
    
#     def forward(self, x):
#         h = self.main(x)
#         out_src = self.conv1(h)
#         out_src = torch.flatten(out_src, start_dim=1)
#         out_src = self.linear(out_src)
#         out_src = self.sigmoid(out_src)

#         return out_src