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
    
    def visualize_integrated_gradients(self, 
                                       images, 
                                       integrated_gradients, 
                                       storage_path,
                                       run_name):
        """
        """
        fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))
        cmap_bound = np.abs(integrated_gradients).max()

        for i in range(len(images)):
            # original image
            ax[i, 0].imshow(images[i, :, :, :].squeeze(), cmap='gray')
            ax[i, 0].set_title('Original Image')

            # attributions 
            attr = integrated_gradients[i, :, :, :].squeeze()
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

        plt.show()
        storage_path = os.path.expanduser(storage_path)
        storage_path = f"{storage_path}/{run_name}"
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)
        
        plt.savefig(storage_path)

        attributions = self.overlay_function(images=images, 
                                             integrated_gradients=integrated_gradients)
        

        storage_path = os.path.expanduser(storage_path)
        storage_path = f"{storage_path}/{run_name}"
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)
        
        vutils.save_image(attributions.data, 
                          f"{storage_path}/integrated_gradients.png")



        

    def overlay_function(self, images, integrated_gradients):
        return np.clip(0.7 * images + 0.5 * integrated_gradients, 0, 255)