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
        if torch.is_tensor(images):
            images = images.cpu().detach().numpy()
            #attributions = attributions.cpu().detach().numpy()

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
        
        # vutils.save_image(attributions.data, 
        #                   f"{storage_path}/{method}.png")


    def overlay_function(self, images, attributions):
        return np.clip(0.7 * images + 0.5 * attributions, 0, 255)

    def visualize_results(self, 
                          x_origin, 
                          x_generated,
                          rs_origin,
                          rs_generated,
                          storage_path,
                          run_name,
                          step):
        """
        """
        x_origin = x_origin.cpu().detach().numpy()
        x_generated = x_generated.cpu().detach().numpy()
        rs_origin = rs_origin.cpu().detach().numpy()
        rs_generated = rs_generated.cpu().detach().numpy()
        fig, ax = plt.subplots(nrows=x_origin.shape[0], ncols=2, figsize=(10, 10))
        #pdb.set_trace()
        for i in range(len(x_origin)):
            rs_gen = round(rs_generated[i][0], 3)
            rs_orig = round(rs_origin[i][0], 3)
            ax[i, 0].imshow(x_generated[i, :, :, :].squeeze(), cmap='gray')
            #ax[i, 0].set_title(f"GI - RS: {rs_gen}")
            ax[i, 0].set_title("GI - RS: {:.3f}".format(rs_gen))

            ax[i, 1].imshow(x_origin[i, :, :, :].squeeze(), cmap='gray')
            ax[i, 1].set_title("OI - RS: {:.3f}".format(rs_orig))

        for ax in fig.axes:
            ax.axis('off')
        
        storage_path = os.path.expanduser(storage_path)
        storage_path = f"{storage_path}/{run_name}"
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)
        
        plt.savefig(f"{storage_path}/baseline_{step}.png", dpi=300)


    def visualize_3D_slices(self,
                            slices,
                            x_origin,
                            x_generated,
                            rs_origin,
                            rs_generated,
                            storage_path,
                            run_name,
                            step):
        """
        """
        plt.close()
        num_samples = len(x_origin)
        all_slices = []

        for img_one, img_two in zip(x_origin.detach().cpu(), x_generated.detach().cpu()):
            content_slices = torch.sum(img_one, dim=(0, 2, 3)) != 0
            img_one = img_one[:, content_slices]
            img_two = img_two[:, content_slices]

            indices, _ = torch.sort(torch.randperm(img_one.shape[1])[:slices])
            all_slices.extend(img_one[:, indices].permute(1, 0, 2, 3))
            all_slices.extend(img_two[:, indices].permute(1, 0, 2, 3))

        fig = plt.figure(figsize=(10, 10))
        grid = ImageGrid(
            fig,
            111,
            nrows_ncols=(num_samples * 2, slices),
            axes_pad=0.1,
        )

        for ax, arr in zip(grid, all_slices):
            if arr.shape[0] == 2:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    ax.imshow(arr[0, :, :], cmap='gray', vmin=0, vmax=1)
                    mask = np.where(arr[1, :, :] < 0.25, np.nan, arr[1, :, :])
                    ax.imshow(mask, cmap='cool', alpha=0.1, vmin=0.25, vmax=1)
                    ax.axis('off')

            else:
                ax.imshow(arr[0, :, :], vmin=0, vmax=1)
                ax.axis('off')

        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = deepcopy(Image.open(buf))
        buf.close()
        plt.close(fig)

        storage_path = os.path.expanduser(storage_path)
        storage_path = f"{storage_path}/{run_name}"
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)

        fig.savefig(f"{storage_path}/baseline3d_{step}.png")


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
        plt.ylim(np.min(rs_generated), np.max(rs_generated))
        plt.xlim(np.min(rs_origin), np.max(rs_origin))

        storage_path = os.path.expanduser(storage_path)
        storage_path = f"{storage_path}/{run_name}"
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)

        plt.savefig(f"{storage_path}/scatter_orig_gen_{epoch}.png")
        plt.close()
    