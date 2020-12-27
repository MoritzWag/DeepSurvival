
import torch 
import pdb

from torch import nn
from src.models.base import BaseModel



class DeepCoxPH(BaseModel):
    """
    """
    def __init__(self,
                 deep: nn.Module, 
                 structured_input_dim: int, 
                 output_dim: int,
                 orthogonalize: bool,
                 **kwargs):
        super(DeepCoxPH, self).__init__(**kwargs)

        self.deep = deep 
        self.output_dim = output_dim
        self.structured_input_dim = structured_input_dim
        self.orthogonalize = orthogonalize
        self.overall_dim = self.structured_input_dim + self.deep.fc3.out_features
        self.linear = nn.Linear(in_features=self.overall_dim, out_features=self.output_dim)

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

        pred_t = predictions.t()

        rr = self.logsumexp_masked(pred_t, riskset, axis=1, keepdim=True)

        losses = torch.multiply(event, rr - predictions)
        loss = torch.mean(losses)

        return {'loss': loss}


    def logsumexp_masked(self, risk_scores, mask, axis=0, keepdim=None):
        """
        """

        mask = torch.from_numpy(mask)
        risk_scores_masked = torch.multiply(risk_scores, mask)
        amax = torch.max(risk_scores_masked, dim=axis, keepdim=True)
        risk_scores_shift = risk_scores_masked - amax[0]

        exp_masked = torch.multiply(torch.exp(risk_scores_shift), mask)
        exp_sum = torch.sum(exp_masked, axis=axis, keepdim=True)
        output = amax[0] + torch.log(exp_sum)
        if not keepdim:
            output = torch.squeeze(output, axis=axis)
        return output

    def _accumulate_batches(self, data, cuda=False):
        """
        """
        images = []
        tabular_data = []
        events = []
        times = []
        for batch, data in enumerate(data):
            image = data[0]
            tabular_date = data[1]
            event = data[2]
            time = data[3]
            if cuda: 
                image = image.cuda()
                tabular_date = tabular_date.cuda()
            images.append(image)
            tabular_data.append(tabular_date)
            events.append(event)
            times.append(time)

        images = torch.cat(images)
        tabular_data = torch.cat(tabular_data)
        events = torch.cat(events)
        times = torch.cat(times)

        dict_batches = {'images': images.float(), 'tabular_data': tabular_data,
                        'events': events, 'times': times}

        return dict_batches

