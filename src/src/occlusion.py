import os 
import torch 
import pdb 
import numpy as np
from tqdm import tqdm 

from torch import nn 



class Occlusion(nn.Module):
    """
    """
    def __init__(self,
                 model,
                 player_generator, 
                 **kwargs):
        super(Occlusion, self).__init__(**kwargs)

        self.model = model
        self.player_generator = player_generator
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _measure_difference(self, pred, out):
        """
        """
        return torch.sum(pred - out, axis=1)
    
    def run(self, images, tabular_data, baselines):
        """
        """
        with torch.no_grad():
            if self.player_generator is None:
                self.player_generator = DefaultPlayerIterator(images)
            
            num_players = self.player_generator.n_players
            pred = self.model(tabular_data, images)
            one = torch.tensor([1.0]).to(self.device)
            attributions = torch.zeros(images.shape).to(self.device)

            # I do not need tile mask
            with tqdm(range(num_players)) as progress_bar:
                for i, (mask, mask_output) in enumerate(tqdm(self.player_generator)):
                    #pdb.set_trace()
                    mask, feat_mask = mask
                    mask_output, feat_output = mask_output

                    # from numpy to tensor
                    mask = torch.from_numpy(mask).to(self.device)
                    feat_mask = torch.from_numpy(feat_mask).to(self.device)
                    mask_output = torch.from_numpy(mask_output).to(self.device)
                    feat_output = torch.from_numpy(feat_output).to(self.device)
                    mask_out = mask_output.clone()
                    #mask_out = mask_output.repeat(tile_mask).to(self.device)

                    # loop for each image:
                    for idx in range(images.shape[0]):
                        img = images.clone()
                        bl_img = images.clone()
                        bl_img[idx] = baselines[idx]

                        feat_i = tabular_data.mul(one - feat_mask).add(tabular_data * feat_mask)
                        images_i = img.mul(one - mask).add(bl_img * mask)

                        out = self.model(feat_i.float(), images_i.float())
                        attr = self._measure_difference(pred, out)

                        mask_attr = mask_out * attr[idx]
                        attributions[idx] += mask_attr.squeeze(0)
                        
        
        return attributions.cpu().detach().numpy()

