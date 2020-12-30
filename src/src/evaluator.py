import torch
import pdb
import numpy as np 

from torch import nn

from sksurv.linear_model.coxph import BreslowEstimator
from sksurv.metrics import concordance_index_censored, integrated_brier_score

class Evaluator(nn.Module):
    """
    """
    def __init__(self, **kwargs):
        super(Evaluator, self).__init__(**kwargs)
        self.scores = {}

    def get_metrics(self, event, time, riskscores, y, times_unique, **kwargs):
        """
        """
        surv_preds = self.get_survival_predictions(riskscores, event, time)
        concordance_index = self.concordance_index(event.astype(np.bool), time, riskscores)
        ibs = self.integrated_brier_score(y, surv_preds, times_unique)

        self.scores['cindex'] = concordance_index
        self.scores['ibs'] = ibs

    def concordance_index(self, event, time, riskscores, **kwargs):
        """
        """
        cindex = concordance_index_censored(event, time, riskscores)

        return cindex
    
    def integrated_brier_score(self, y, surv_preds, times, **kwargs):
        """
        """
        surv_preds = np.asarray([[fn(t) for t in times] for fn in surv_preds])
        score = integrated_brier_score(y, y, surv_preds, times)

        return score
    
    def get_survival_predictions(self, riskscores, event, time, **kwargs):
        """
        """
        breslow = BreslowEstimator().fit(riskscores, event, time)
        survival = breslow.get_survival_function(riskscores)

        return survival

    def get_hazards(self, **kwargs):
        """
        """
        pass 
    
    def get_cumulative_hazard(self, hazard, index):
        pass
        
    def get_baseline_hazard(self):
        pass


    
        


