from typing import Tuple

import torch.nn
import pdb
from torch import Tensor
from torch.nn import functional as F


def square(x: torch.Tensor) -> torch.Tensor:
    return torch.pow(x, torch.tensor([2.0], device=x.device))


class ProbLinearInput(torch.nn.Linear):
    
    """
    Lightweight probabilistic linear input layer. Transforms sampled inputs into a normally distributed inputs.
    """

    def __init__(self, in_features, out_features, bias=True):
        super(ProbLinearInput, self).__init__(
            in_features=in_features, out_features=out_features, bias=bias
        )
        self.epsilon = 1e-7
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(
        self, inputs: Tuple[Tensor, Tensor, Tensor]
    ) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        """
        Performs probabilistic transformation in the forward pass of a linear operation.
        Args:
            inputs (): Inputs in a tuple of (Input, mask, k)
        Returns: Mean and variance of the input distribution with (mv1) and without (mv2) masked features.
        """
        # Note: Takes zero baseline as default, does not use DaspConfig
        input_, mask, k = inputs
        assert len(mask.shape) == len(
            input_.shape
        ), "Inputs must have same number of dimensions."
        one = torch.tensor([1.0], device=input_.device)
        mask_comp = one - mask
        inputs_i = input_ * mask_comp

        dot = F.linear(input_, self.weight)
        dot_i = F.linear(inputs_i.float(), self.weight)
        dot_mask = torch.sum(mask_comp, dim=1, keepdim=True)
        dot_v = F.linear(square(inputs_i), square(self.weight))
        # Compute mean without feature i
        mu = dot_i / dot_mask
        v = dot_v / dot_mask - square(mu)
        # Compensate for number of players in current coalition
        mu1 = mu * k
        # Compute mean of the distribution that also includes player i (acting as bias to expectation)
        mu2 = mu1 + (dot - dot_i)
        # Compensate for number or players in the coalition
        v1 = v * k * (one - (k - one) / (dot_mask - one))
        # Set something different than 0 if necessary
        v1 = torch.clamp(v1, min=self.epsilon)
        # Since player i is only a bias, at this point the variance of the distribution than
        # includes it is the same
        v2 = v1

        if isinstance(self.bias, torch.nn.Parameter):
            mu1 += self.bias
            mu2 += self.bias

        return (mu1, v1), (mu2, v2)