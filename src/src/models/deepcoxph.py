
import torch 
import pdb
import numpy as np 

from torch import nn
from src.models.base import BaseModel
from src.data.utils import safe_normalize


class DeepCoxPH(BaseModel):
    """
    """
    def __init__(self,
                 deep: nn.Module, 
                 structured_input_dim: int, 
                 out_dim: int, 
                 output_dim: int,
                 orthogonalize: bool,
                 **kwargs):
        super(DeepCoxPH, self).__init__()

        self.deep = deep 
        self.output_dim = output_dim
        self.structured_input_dim = structured_input_dim
        self.orthogonalize = orthogonalize
        self.overall_dim = self.structured_input_dim + self.deep.out_dim
        self.linear = nn.Linear(in_features=self.overall_dim, out_features=self.output_dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, tabular_data, images, **kwargs):
        
        unstructured = self.deep(images)
        if self.orthogonalize:
            unstructured_orth = self._orthogonalize(tabular_data, unstructured)
            features_concatenated = torch.cat((tabular_data, unstructured_orth), axis=1)
        else:
            features_concatenated = torch.cat((tabular_data, unstructured), axis=1)
        
        riskscore = self.linear(features_concatenated.float())

        return riskscore

    def _loss_function(self, event, riskset, predictions):
        """
        """
        predictions = safe_normalize(predictions)
        pred_t = predictions.t()

        rr = self.logsumexp_masked(pred_t, riskset, axis=1, keepdim=True)

        losses = event * (rr - predictions)
        loss = torch.mean(losses)

        return loss


    def logsumexp_masked(self, riskscores, mask, axis=0, keepdim=True):
        """
        """
        mask = mask.to(self.device)
        risk_scores_masked = riskscores * mask
        amax, _ = torch.max(risk_scores_masked, dim=axis, keepdim=True)
        risk_scores_shift = risk_scores_masked - amax

        exp_masked = risk_scores_shift.exp() * mask
        exp_sum = torch.sum(exp_masked, axis=axis, keepdim=True)
        output = amax + torch.log(exp_sum)
        if not keepdim:
            output = torch.squeeze(output, axis=axis)
        return output

    def _accumulate_batches(self, data):
        """
        """
        images = []
        tabular_data = []
        events = []
        times = []
        for batch, data in enumerate(data):
            image = data['images']
            tabular_date = data['tabular_data']
            # image = data['images'].to(self.device)
            # tabular_date = data['tabular_data'].to(self.device)
            event = data['event']
            time = data['time']
            # if cuda: 
            #     image = image.c
            #     tabular_date = tabular_date.cuda()
            images.append(image)
            tabular_data.append(tabular_date)
            events.append(event)
            times.append(time)

        images = torch.cat(images)
        tabular_data = torch.cat(tabular_data)
        events = torch.cat(events)
        times = torch.cat(times)

        dict_batches = {'images': images.float(), 'tabular_data': tabular_data,
                        'event': events, 'time': times}

        return dict_batches
    
    def _sample_batch(self, data, num_obs):
        """
        """
        acc_batch = self._accumulate_batches(data=data)
        indeces = torch.randint(0, acc_batch['images'].shape[0] - 1, size=(num_obs, ))
        
        for value, key in zip(acc_batch.values(), acc_batch.keys()):
            acc_batch[key] = value[indeces]
        
        return acc_batch

    def predict_on_images(self, images, tabular_data, **kwargs):
        """
        """
        unstructured = self.deep(images)
        if self.orthogonalize:
            unstructured_orth = self._orthogonalize(tabular_data, unstructured)
        else:
            unstructured_orth = unstructured
        
        weights = self.linear.weight.data
        unstructured_weights = weights[:, 0:unstructured_orth.shape[1]]
        out = unstructured_orth.matmul(unstructured_weights.t())
        
        return out
