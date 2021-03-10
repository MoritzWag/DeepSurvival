import torch 
import pdb
import numpy as np 

from torch import nn 
from src.models.base import BaseModel
from src.data.utils import safe_normalize


class Baseline(BaseModel):
    """
    """
    def __init__(self,
                 deep: nn.Module, 
                 out_dim: int, 
                 output_dim: int,
                 **kwargs):
        super(Baseline, self).__init__()

        self.deep = deep
        self.out_dim = out_dim
        self.output_dim = output_dim 
        self.linear = nn.Linear(in_features=self.out_dim, out_features=self.output_dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, images, **kwargs):
        
        unstructured = self.deep(images.float())
        riskscore = self.linear(unstructured)

        return riskscore

    def predict_on_images(self, **kwargs):
        pass


class Linear(BaseModel):
    """
    """
    def __init__(self,
                 structured_input_dim,
                 output_dim,
                 **kwargs):
        super(Linear, self).__init__()
        
        self.structured_input_dim = structured_input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(in_features=self.structured_input_dim, out_features=self.output_dim)

    
    def forward(self, tabular_data, **kwargs):
        riskscore = self.linear(tabular_data.float())
        return riskscore

    def predict_on_images(self, **kwargs):
        pass