import torch
import pdb
import numpy as np 

from torch import nn

from sksurv.linear_model.coxph import BreslowEstimator

class Evaluator(nn.Module):
    """
    """
    def __init__(self, **kwargs):
        super(Evaluator, self).__init__(**kwargs)
        self.scores = {}

    def cindex_metric(self):
        pass

    def get_measures(self, riskscore, events, times):
        pdb.set_trace()
        riskscore = riskscore.detach().numpy()
        events = events.detach().numpy()
        times = times.detach().numpy()
        breslow = BreslowEstimator().fit(riskscore, events, times)
        survival = breslow.get_survival_function(riskscore)
        

    def get_baseline_hazard(self):
        pass

    def get_hazard_rate(self):
        pass

    def get_survival_curve(self):
        pass

        


