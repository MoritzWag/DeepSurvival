from typing import Tuple 
import numpy as np 
import torch 
import pdb
from torch import Tensor
from torch.nn import Conv1d, Conv2d

import torch.nn.functional as F
torch.set_default_dtype(torch.float64)

def square(x: torch.Tensor) -> torch.Tensor:
    return torch.pow(x, torch.tensor([2.0], device=x.device))

class ProbConv2dInput(Conv2d):
    """
    """
    def __init__(self, 
                 in_channels:int,
                 out_channels:int,
                 kernel_size,
                 **kwargs
                ):
        super(ProbConv2dInput, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            **kwargs
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _conv2d(self, inputs, kernel):
        return F.conv2d(inputs,
                        kernel, 
                        bias=None,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        groups=self.groups)

    def forward(self, inputs, baselines):
        """
        """
        input_, mask, k = inputs
        n_players =  torch.sum(torch.ones_like(inputs[0]))
        size_coalition = torch.sum(mask[0])
        n_players = n_players / size_coalition

        # When I say k, i actually mean k coalition players so need to compensate for it 
        k = torch.unsqueeze(torch.unsqueeze(k, -1), -1)
        one = torch.as_tensor([1.0], dtype=torch.float).to(self.device)

        # 1.) apply mask complement => mask_comp
        mask_comp = input_ * (torch.ones_like(mask) - mask)
        pdb.set_trace()
        # 2.) inputs_i = mask_comp + baseline * mask
        inputs_i = mask_comp + baselines * mask

        # 3.) Calculate ghost
        ghost = torch.ones_like(input_) * (1.0 - mask)

        # # Old implementation
        # ghost = torch.ones_like(input_) * (1.0 - mask)
        # inputs_i = input_ * (1.0 - mask)
        conv = self._conv2d(input_, self.weight)
        conv_i = self._conv2d(inputs_i, self.weight)
        conv_count = self._conv2d(ghost, torch.ones_like(self.weight))

        conv_v = self._conv2d(square(inputs_i), square(self.weight))

        # Compute mean without feature i
        # Compensate for number of players in current coalition
        mu1 = torch.mul(conv_i, torch.div(k, n_players))
        # Compute mean of the distribution that also includes player i (acting as bias to expectation)
        mu2 = mu1 + (conv - conv_i)
        # Compute variance without player i
        v1 = torch.div(conv_v, conv_count) - square(torch.div(conv_i, conv_count))

        # Compensate for number or players in the coalition
        k = torch.mul(conv_count, torch.div(k, n_players))
        v1 = v1 * k * (one - (k - one) / (conv_count - one))
        # Set something different than 0 if necessary
        v1 = torch.clamp(v1, min=0.00001)
        # Since player i is only a bias, at this point the variance of the distribution than
        # includes it is the same
        v2 = v1

        if isinstance(self.bias, torch.nn.Parameter):
            b = self.bias.view(-1, 1, 1)
            mu1.add_(b)
            mu2.add_(b)

        mu1 = mu1.float()
        v1 = v1.float()
        mu2 = mu2.float()
        v2 = v2.float()
        
        return (mu1, v1), (mu2, v2)




