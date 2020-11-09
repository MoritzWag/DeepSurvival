import pandas as pd
import numpy as np 
import torch 
import pytorch_lightning as pl


from torch import optim
from torch.utils import DataLoader
from torch.autograd import Variable



class DeepSurvExperiment(pl.LightningModule):
    """
    """
    def __init__(self,
                 model,
                 params,
                 log_params,
                 model_hyperparams,
                 run_name,
                 experiment_name):
        super(DeepSurvExperiment, self).__init__()
    
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        pass 

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outputs):
        pass 

    def test_step(self, batch, batch_idx):
        pass

    def test_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        pass 

    def train_dataloader(self):
        pass 

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass