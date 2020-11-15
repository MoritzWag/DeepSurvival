import torch 
import torch.nn as nn 

from abc import ABC, abstractmethod
from src.evaluator import Evaluator

class BaseModel(ABC, Evaluator):
    """
    """
    def __init__(self, **kwargs):
        super(BaseModel, self).__init__(**kwargs)

    @abstractmethod
    def _loss_function(self):
        pass
    
    @abstractmethod
    def _orthogonalize(self):
        pass

