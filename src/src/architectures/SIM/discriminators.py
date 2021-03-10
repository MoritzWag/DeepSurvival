import torch 
import numpy as np 
import pdb
import torch.nn.functional as F

from torch import nn 


class DiscriminatorSIM(nn.Module):
    """
    """
    def __init__(self,
                 img_size=28,
                 n_dim=3,
                 conv_dim=64,
                 dimensions=2,
                 c_dim=2,
                 repeat_num=6):
        super(DiscriminatorSIM, self).__init__()

        if dimensions == 2:
            conv = nn.Conv2d
        else:
            conv = nn.Conv3d
        
        layers = []
        layers.append(conv(n_dim, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(conv(curr_dim, curr_dim*2,  kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2
        
        kernel_size = int(img_size // np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = conv(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.linear = nn.Linear(in_features=9, out_features=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_src = torch.flatten(out_src, start_dim=1)
        out_src = self.linear(out_src)
        out_src = self.sigmoid(out_src)

        return out_src

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