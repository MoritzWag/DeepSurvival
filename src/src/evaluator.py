import torch
import pdb
import numpy as np 

from torch import nn

from sksurv.linear_model.coxph import BreslowEstimator
from sksurv.metrics import concordance_index_censored, integrated_brier_score, brier_score

class Evaluator(nn.Module):
    """
    """
    quantiles = [0.25, 0.5, 0.75]

    def __init__(self, **kwargs):
        super(Evaluator, self).__init__(**kwargs)
        self.scores = {}

    def get_metrics(self, event, time, riskscores, y, times_unique, part, **kwargs):
        """
        """
        try:
            event = event.cpu().detach().numpy()
            time = time.cpu().detach().numpy()
        except:
            pass
        surv_preds = self.get_survival_predictions(riskscores, event, time)
        concordance_index = self.concordance_index(event.astype(np.bool), time, riskscores)
        brier_scores = self.brier_score(y, surv_preds, times_unique, self.quantiles)
        #ibs = self.integrated_brier_score(y, surv_preds, times_unique)

        for key, value in zip(brier_scores.keys(), brier_scores.values()):
            self.scores[f"{key}_{part}"] = value
        self.scores[f'cindex_{part}'] = concordance_index[0]
        #self.scores['ibs'] = ibs

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
    
    def brier_score(self, y, surv_preds, time, quantiles, **kwargs):
        """
        """
        brier_scores = {}
        times = np.quantile(time, np.array(quantiles))
        for i, t in enumerate(times):
            surv_pred = [fn(t) for fn in surv_preds]
            ts, score = brier_score(y, y, surv_pred, t)
            brier_scores[f"quantile_{str(quantiles[i])}"] = score[0]

        return brier_scores

    def get_survival_predictions(self, riskscores, event, time, **kwargs):
        """
        """
        breslow = BreslowEstimator().fit(riskscores, event, time)
        survival = breslow.get_survival_function(riskscores)

        return survival

    
        


