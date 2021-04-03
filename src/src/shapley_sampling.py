
import os
import torch
import pdb
import numpy as np 

from tqdm import tqdm
from torch import nn
from torchvision.utils import save_image
from typing import List, Tuple

class ShapleySampling(nn.Module):
    """
    """
    def __init__(self, **kwargs):
        super(ShapleySampling, self).__init__(**kwargs)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def sv_sampling(self, 
                    model, 
                    images, 
                    tabular_data, 
                    baseline, 
                    n_steps=1000):
        """
        """
        batch_size = images.shape[0]
        img_height, img_width = images.shape[2], images.shape[3]
        n_players = images.shape[2] * images.shape[3]
        pbar = tqdm(total=n_steps*n_players)
        with torch.no_grad():
            attributions = np.zeros((batch_size, img_height, img_width, 1), dtype=float)
            for _ in range(n_steps):
                perm = torch.randperm(n_players, device=self.device)
                img_in = baseline.repeat(2, 1, 1, 1)
                tabular_in = tabular_data.repeat(2, 1)
                for end in range(n_players):
                    idx = perm[end]

                    i, j = np.unravel_index(idx.detach().cpu().numpy(), (img_height, img_width))
                    img_in[:batch_size, :, i, j] = images[..., i, j]
                    # pred = model(img_in, tabular_data)
                    pred = model(tabular_in, img_in)
                    pred_w_i, pred_wo_i = torch.chunk(pred, 2)
                    delta = pred_w_i - pred_wo_i
                    attributions[:, i, j] += delta.detach().cpu().numpy()

                    img_in[batch_size:, :, i, j] = images[..., i, j]

                    pbar.update()
            attributions /= n_steps
            attributions = np.transpose(attributions, axes=(0, 3, 1, 2))
        
        return attributions