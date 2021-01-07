import os 
import torch 
import pdb 
import numpy as np
from tdqm import tqdm 

from torch import nn 



class Occlusion(nn.Module):
    """
    """
    def __init__(self, **kwargs):
        super(Occlusion, self).__init__(**kwargs)
    
    def meassure_difference(self, with_feature, without_feature):
        """
        """
        return np.sum(with_feature - without_feature, axis=1)
    
    def run(self, input_):
        """
        """
        with torch.no_grad():
            