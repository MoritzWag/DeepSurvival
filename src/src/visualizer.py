import torch 
import pdb 
import numpy as np 
import os
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from torch import nn 

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class Visualizer(nn.Module):
    """
    """
    def __init__(self, **kwargs):
        super(Visualizer, self).__init__(**kwargs)
    
    def visualize_attributions(self, 
                               images, 
                               attributions,
                               method,
                               storage_path,
                               run_name):
        """
        """
        fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))
        cmap_bound = np.abs(attributions).max()

        for i in range(len(images)):
            # original image
            ax[i, 0].imshow(images[i, :, :, :].squeeze(), cmap='gray')
            ax[i, 0].set_title('Original Image')

            # attributions 
            attr = attributions[i, :, :, :].squeeze()
            im = ax[i, 1].imshow(attr, vmin=-cmap_bound, vmax=cmap_bound, cmap='PiYG')

            # positive attributions
            attr_pos = attr.clip(0, 1).squeeze()
            im = ax[i, 2].imshow(attr_pos, vmin=-cmap_bound, vmax=cmap_bound, cmap='PiYG')
            
            # negative attributions
            attr_neg = attr.clip(-1, 0).squeeze()
            im = ax[i, 3].imshow(attr_neg, vmin=-cmap_bound, vmax=cmap_bound, cmap='PiYG')

        ax[0, 1].set_title('Attributions')
        ax[0, 2].set_title('Positive attributions')
        ax[0, 3].set_title('Negative attributions')

        for ax in fig.axes:
            ax.axis('off')
    
        fig.colorbar(im, cax=fig.add_axes([0.95, 0.25, 0.03, 0.5]))

        storage_path = os.path.expanduser(storage_path)
        storage_path = f"{storage_path}/{run_name}"
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)
        
        plt.savefig(f"{storage_path}/{method}.png", dpi=300)

        attributions = self.overlay_function(images=images, 
                                             attributions=attributions)
        

        storage_path = os.path.expanduser(storage_path)
        storage_path = f"{storage_path}/{run_name}"
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)
        
        vutils.save_image(attributions.data, 
                          f"{storage_path}/{method}.png")


    def overlay_function(self, images, attributions):
        return np.clip(0.7 * images + 0.5 * attributions, 0, 255)

