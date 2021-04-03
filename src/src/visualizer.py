import torch 
import pdb 
import numpy as np 
import os
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from torch import nn 
from src.data.utils import hsl2rgb

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class Visualizer(nn.Module):
    """
    """
    def __init__(self, **kwargs):
        super(Visualizer, self).__init__(**kwargs)
    
    def visualize_attributions(self, 
                               images, 
                               attributions,
                               rgb_trained, 
                               method,
                               storage_path,
                               run_name):
        """
        """
        if torch.is_tensor(images):
            images = images.cpu().detach().numpy()
            #attributions = attributions.cpu().detach().numpy()

        fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(12, 12))
        cmap_bound = np.abs(attributions).max()

        for i in range(len(images)):

            if images.shape[1] == 1:

                # original image
                ax[i, 0].imshow(images[i, :, :, :].squeeze(), cmap='gray')
                ax[i, 0].set_title('Original Image')

                # attributions 
                attr = attributions[i, :, :, :].squeeze()
                im = ax[i, 1].imshow(attr, vmin=-cmap_bound, vmax=cmap_bound, cmap='seismic')

                # positive attributions
                attr_pos = attr.clip(0, 1).squeeze()
                im = ax[i, 2].imshow(attr_pos, vmin=-cmap_bound, vmax=cmap_bound, cmap='seismic')
                
                # negative attributions
                attr_neg = attr.clip(-1, 0).squeeze()
                im = ax[i, 3].imshow(attr_neg, vmin=-cmap_bound, vmax=cmap_bound, cmap='seismic')
            else:
                img = images[i, :, :, :]
                shapes = img.shape
                img = np.transpose(img, axes=(1, 2, 0))
                if not rgb_trained:
                    im = hsl2rgb(img)
                #img = img.reshape(shapes[1], shapes[2], shapes[0])
                
                ax[i, 0].imshow(im, cmap='gray')
                ax[i, 0].set_title('Original Image')

                # attributions
                attr = attributions[i, :, :, :]
                attr = np.transpose(attr, axes=(1, 2, 0))
                #attr = attributions[i, :, :, :].reshape(shapes[1], shapes[2], shapes[0]).squeeze()
                attr = convert_to_grayscale(attr)
                im = ax[i, 1].imshow(attr, vmin=-cmap_bound, vmax=cmap_bound, cmap='seismic')

                # positive attributions
                attr_pos = attr.clip(0, 1).squeeze()
                im = ax[i, 2].imshow(attr_pos, vmin=-cmap_bound, vmax=cmap_bound, cmap='seismic')
                
                # negative attributions
                attr_neg = attr.clip(-1, 0).squeeze()
                im = ax[i, 3].imshow(attr_neg, vmin=-cmap_bound, vmax=cmap_bound, cmap='seismic')



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
        
        plt.savefig(f"{storage_path}/{method}.png", dpi=300, bbox_inches='tight')

        attributions = self.overlay_function(images=images, 
                                             attributions=attributions)
        

        storage_path = os.path.expanduser(storage_path)
        storage_path = f"{storage_path}/{run_name}"
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)
        
        # vutils.save_image(attributions.data, 
        #                   f"{storage_path}/{method}.png")


    def overlay_function(self, images, attributions):
        return np.clip(0.7 * images + 0.5 * attributions, 0, 255)

    def plot_riskscores(self, 
                        riskscores,
                        storage_path,
                        run_name,
                        epoch):
        """
        """
        try:
            riskscores = riskscores.detach().cpu().numpy()
        except:
            pass
        plt.close()

        plt.hist(riskscores, bins='auto')

        storage_path = os.path.expanduser(storage_path)
        storage_path = f"{storage_path}/{run_name}"
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)

        plt.savefig(f"{storage_path}/hist_rs_{epoch}.png")

    def plot_gen_origin_rs(self, 
                           rs_origin,
                           rs_generated,
                           storage_path,
                           run_name,
                           epoch):
        """
        """
        try:
            rs_origin = rs_origin.detach().cpu().numpy().squeeze()
            rs_generated = rs_generated.detach().cpu().numpy().squeeze()
        except:
            pass 

        plt.close()
        plt.scatter(x=rs_origin, y=rs_generated)
        plt.xlabel("Riskscore - Original")
        plt.ylabel("Riskscore - Generated")
        plt.ylim(np.min(rs_origin), np.max(rs_origin))
        plt.xlim(np.min(rs_origin), np.max(rs_origin))

        storage_path = os.path.expanduser(storage_path)
        storage_path = f"{storage_path}/{run_name}"
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)

        plt.savefig(f"{storage_path}/scatter_orig_gen_{epoch}.png")
        plt.close()
    

def convert_to_grayscale(attributions):
    """
    """
    return np.average(attributions, axis=2)