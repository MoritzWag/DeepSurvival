import torch
from torch import nn


class Evaluator(nn.Module):
    """
    """
    def __init__(self, **kwargs):
        super(Evaluator, self).__init__(**kwargs)
        self.scores = {}

    def cindex_metric(self):
        pass
