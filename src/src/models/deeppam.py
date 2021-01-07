
import torch 
import pdb
import numpy as np

from torch import nn
from torch import functional as F
from src.models.base import BaseModel


class DeepPAM(BaseModel):
    """
    """
    def __init__(self,
                 deep: nn.Module, 
                 structured_input_dim: int,
                 output_dim: int,
                 orthogonalize: bool,
                 **kwargs):
        super(DeepPAM, self).__init__(**kwargs)

        self.deep = deep 
        self.output_dim = output_dim
        self.structured_input_dim = structured_input_dim
        self.orthogonalize = orthogonalize
        self.overall_dim = self.structured_input_dim + self.deep.fc3.out_features
        self.linear = nn.Linear(in_features=self.overall_dim, out_features=self.output_dim)

    def forward(self, tabular_data, images, offset, index, splines, **kwargs):

        # 1.) pass unstructured part through deep network
        unstructured = self.deep(images)

        # 2.) bring latent representation into ped format
        # 3.) additionally, orthogonalize for each interval seperately
        unstructured_ped = self.latent_to_ped(unstructured, tabular_data.to(self.device), index.to(self.device))

        # 4.) concatenate unstructured_ped with structured 
        features_concatenated = torch.cat((tabular_data, unstructured_ped), axis=1)
        
        # 5.) pass through last linear layer
        out = self.linear(features_concatenated.float())
        out = out.squeeze(1)

        # 6.) add offset and splines
        splines = torch.sum(splines, dim=1)
        out = out + offset.to(self.device) + splines.to(self.device)
        hazard = torch.exp(out)

        return hazard

    def _loss_function(self, pred_status, true_status, epsilon=1e-07):
        """poisson_loss
        """
        loss = torch.mean(pred_status - true_status * torch.log(pred_status + epsilon))

        return loss

    def latent_to_ped(self, unstructured, structured, index):
        """bring latent representation into ped format
        """
        num_of_intervals = torch.unique(index, return_counts=True)[1]

        structured_compressed = []
        for idx in range(num_of_intervals.shape[0]):
            num_obs = torch.sum(num_of_intervals[0:idx+1])
            structured_obs = structured[num_obs -1]
            structured_compressed.append(structured_obs)
        
        structured = torch.vstack(structured_compressed)
        unstructured_orth = self._orthogonalize(structured, unstructured)
        unstructured_ped = []
        for idx in range(unstructured.shape[0]):
            unstructured_instance = unstructured_orth[idx].unsqueeze(0)
            ped_instance = torch.repeat_interleave(unstructured_instance,
                                                   repeats=num_of_intervals[idx],
                                                   dim=0)
            unstructured_ped.append(ped_instance)
        
        unstructured_ped = torch.vstack(unstructured_ped)
        
        return unstructured_ped
    
    def _accumulate_batches(self, data, cuda=False):
        """
        """
        images = []
        tabular_data = []
        offsets = []
        ped_statuses = []
        indeces = []
        splines = []
        for batch, data in enumerate(data):
            image = data['images']
            tabular_date = data['tabular_data']
            offset = data['offset']
            ped_status = data['ped_status']
            index = data['index']
            spline = data['splines']

            if cuda: 
                image = image.cuda()
                tabular_date = tabular_date.cuda()
            images.append(image)
            tabular_data.append(tabular_date)
            offsets.append(offset)
            ped_statuses.append(ped_status)
            indeces.append(index)
            splines.append(spline)

        images = torch.cat(images)
        tabular_data = torch.cat(tabular_data)
        offsets = torch.cat(offsets)
        ped_statuses = torch.cat(ped_statuses)
        indeces = torch.cat(indeces)
        splines = torch.cat(splines)

        dict_batches = {'images': images.float(), 'tabular_data': tabular_data, 'offset': offsets,
                        'ped_status': ped_statuses, 'index': indeces, 'splines': splines}

        return dict_batches
    
    def _sample_batch(self, data, num_obs):
        """
        """
        acc_batch = self._accumulate_batches(data=data)
        samples = torch.randint(0, acc_batch['images'].shape[0] - 1, size=(num_obs, )).numpy()
        index = acc_batch['index'].cpu().detach().numpy()
        indeces = []
        for sample in samples:
            idx = np.where(index==sample)[0]
            indeces.append(idx)
        
        #pdb.set_trace()
        indeces = np.hstack(indeces)
        indeces = torch.from_numpy(indeces).to(self.device)
        
        # index images 
        acc_batch['images'] = acc_batch['images'][samples, :, :, :]
        for value, key in zip(acc_batch.values(), acc_batch.keys()):
            if key != 'images':
                acc_batch[key] = value[indeces]
            else:
                continue
        
        return acc_batch

    def predict_on_images(self, images, tabular_data):
        """
        """
        unstructured = self.deep(images)
        unstructured_orth = unstructured
        #unstructured_orth = self._orthogonalize(tabular_data, unstructured)
        weights = self.linear.weight.data 
        unstructured_weights = weights[:, 0:unstructured_orth.shape[1]]
        out = unstructured_orth.matmul(unstructured_weights.t())
        out = torch.exp(out)

        return out
