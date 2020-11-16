import torch 
import torch.nn as nn 

from abc import ABC, abstractmethod
from src.evaluator import Evaluator
from src.visualizer import Visualizer

class BaseModel(ABC, Evaluator, Visualizer):
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

    def accumulate_batches(self, data, cuda=False):
        """
        """
        images = []
        tabular_data = []
        for batch, data in enumerate(data):
            image = data[0]
            tabular_date = data[1]
            if cuda: 
                image = image.cuda()
                tabular_date = tabular_date.cuda()
            images.append(image)
            tabular_data.append(tabular_date)

        images = torch.cat(images)
        tabular_data = torch.cat(tabular_data)

        return images, tabular_data



