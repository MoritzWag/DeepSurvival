import torch 
import torch.nn as nn 

from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    """
    def __init__(self):
        pass 
    
    @abstractmethod
    def train_one_batch(self):
        pass 

    def validate(self):
        pass 

    def predict(self):
        pass 

    def evaluate(self):
        pass 

