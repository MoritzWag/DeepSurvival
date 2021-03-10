import torch 
import pdb
import time 
import os
import io
import datetime 
import pandas as pd
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from matplotlib import colors
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image
from torch import nn
from torchvision.utils import save_image

from src.postprocessing import plot_train_progress

class BaselineGenerator(nn.Module):
    """
    """
    def __init__(self,
                 discriminator,
                 generator,
                 survival_model,
                 data_type,
                 c_dim,
                 img_size,
                 generator_params,
                 discriminator_params,
                 trainer_params,
                 logging_params):
        super(BaselineGenerator, self).__init__()

        self.discriminator = discriminator
        self.generator = generator
        self.survival_model = survival_model
        self.data_type = data_type

        self.c_dim = c_dim
        self.img_size = img_size
        self.trainer_params = trainer_params
        self.logging_params = logging_params

        # discriminator params
        self.d_lr = discriminator_params['learning_rate']
        self.d_beta1 = discriminator_params['beta1']
        self.d_beta2 = discriminator_params['beta2']
        self.d_repeat_num = discriminator_params['repeat_num']
        self.d_conv_dim = discriminator_params['conv_dim']
        self.d_n_dim = discriminator_params['n_dim']
        self.d_dimensions = discriminator_params['dimensions']

        # generator params
        self.g_lr = generator_params['learning_rate']
        self.g_beta1 = generator_params['beta1']
        self.g_beta2 = generator_params['beta2']
        self.g_repeat_num = generator_params['repeat_num']
        self.g_conv_dim = generator_params['conv_dim']
        self.g_n_dim = generator_params['n_dim']
        self.g_dimensions = discriminator_params['dimensions']
        

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.build_model()
        self.train_history = pd.DataFrame()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    
    def build_model(self):
        """
        """
        # configure discriminator
        self.D = self.discriminator(n_dim=self.d_n_dim, 
                                    conv_dim=self.d_conv_dim, 
                                    dimensions=self.d_dimensions,
                                    repeat_num=self.d_repeat_num)

        # configure generator
        self.G = self.generator(n_dim=self.g_n_dim, 
                                conv_dim=self.g_conv_dim, 
                                dimensions=self.g_dimensions,
                                repeat_num=self.g_repeat_num)

        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.d_beta1, self.d_beta2])
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.g_beta1, self.g_beta2])

        for p in self.survival_model.parameters():
            p.requires_grad = False
        
        # put them on device
        self.survival_model.to(self.device).float()
        self.put_parallel()
    
    def reset_grad(self):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
    
    def put_parallel(self):
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.G = nn.DataParallel(self.G).float()
            self.D = nn.DataParallel(self.D).float()

        self.G.to(self.device).float()
        self.D.to(self.device).float()

    def train(self, train_gen, val_gen):
        """
        """
        print("start training")
        start_time = time.time()
        for step in range(self.trainer_params['n_steps']):
            self.D.train()
            self.G.train()

            try:
                batch = next(train_gen_iter)
            except:
                train_gen_iter = iter(train_gen)
                batch = next(train_gen_iter)
            images = batch['images'].float().to(self.device)
            #images.requires_grad = True
            tabular_data = batch['tabular_data'].float().to(self.device)
            #tabular_data.requires_grad = True

            # derive labels for batch
            label_target, label_origin = self.generate_labels(images=images, tabular_data=tabular_data)
            
            # put data on device
            label_target = label_target.to(self.device)
            label_origin = label_origin.to(self.device)
            
            ############################################################
            #
            # 1.) Train the Discriminator
            #
            ############################################################
            
            # Compute loss with real images
            out_src = self.D(images)
            d_loss_real =  - torch.mean(out_src)

            # Compute loss with fake images
            x_mask = self.G(images, label_target.float())
            output_mask = x_mask
            #x_fake = images + output_mask
            x_fake = x_mask
            # x_fake = self.tanh(x_fake)


            out_src = self.D(x_fake)
            d_loss_fake = torch.mean(out_src)
            #out_domain = self.survival_model.predict_on_images(x_fake, tabular_data)
            #out_domain = self.survival_model(tabular_data, x_fake)

            # Compute loss for gradient penalty
            if images.ndim == 5:
                alpha = torch.rand(images.size(0), 1, 1, 1, 1).to(self.device)
            else:
                alpha = torch.rand(images.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * images.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src = self.D(x_hat.to(self.device).float())
            d_loss_gp = self.gradient_penalty(out_src, x_hat)
            
            # Backward and optimize
            d_loss = d_loss_real + d_loss_fake + self.trainer_params['lambda_gp'] * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_gp'] = d_loss_gp.item()

            ############################################################
            #
            # 2.) Train the Generator
            #
            ############################################################

            if (step + 1) % self.trainer_params['n_critic'] == 0:
                # Original-to-target domain 
                x_mask = self.G(images, label_target.float())
                output_mask = x_mask
                #x_fake = images + output_mask
                x_fake = x_mask
                # x_fake = self.tanh(x_fake)

                out_src = self.D(x_fake)
                #out_domain = self.survival_model(tabular_data, x_fake)
                out_domain = self.survival_model.predict_on_images(x_fake, tabular_data)
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = self.domain_loss(out_domain, label_target, alpha= self.trainer_params['alpha'], tolerance=self.trainer_params['tolerance'],)

                # Target-to-original domain

                x_mask_recon = self.G(x_fake, label_origin)
                output_mask_recon = x_mask_recon
                #x_reconstruct = x_fake + output_mask_recon
                x_reconstruct = x_mask_recon
                # x_reconstruct = self.tanh(x_reconstruct)

                g_loss_rec = self.reconstruction_loss(images, x_reconstruct)

                # map_loss = torch.abs(x_mask).mean()
                

                # Backward and optimize
                # + 0.0001 * map_loss
                g_loss = g_loss_fake + self.trainer_params['lambda_rec'] * g_loss_rec \
                        + self.trainer_params['lambda_cls'] * g_loss_cls 
                        # + self.trainer_params['lambda_map'] * map_loss \
                    

                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()
                #loss['G/loss_norm'] = map_loss.item()

                train_history = pd.DataFrame([[value for value in loss.values()]],
                                                    columns=[key for key in loss.keys()])
                self.train_history = self.train_history.append(train_history, ignore_index=True)

            # print out training info
            if (step  + 1) % self.trainer_params['log_step'] == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, step+1, self.trainer_params['n_steps'])
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)
                print(f"max value: {torch.max(x_mask_recon)} | min value: {torch.min(x_mask_recon)}")
            
            if (step + 1) % self.trainer_params['sample_step'] == 0:
                self.validate(val_gen=val_gen, step=step)

        plot_train_progress(self.train_history, storage_path="./baseline_generator")

    def validate(self, val_gen, step):
        """
        """
        self.D.eval()
        self.G.eval()

        try:
            batch = next(val_gen_iter)
        except:
            val_gen_iter = iter(val_gen)
            batch = next(val_gen_iter)
        images = batch['images'].float().to(self.device)
        tabular_data = batch['tabular_data'].float().to(self.device)
        # derive labels for batch
        label_target, label_origin = self.generate_labels(images=images, tabular_data=tabular_data)
        
        # put data on device
        label_target = label_target.to(self.device)
        label_origin = label_origin.to(self.device)

        with torch.no_grad():
            x_mask = self.G(images, label_target.float())
            #x_generated = images + x_mask
            x_generated = x_mask
            #x_generated = self.tanh(x_generated)

            # derive predicted riskscores for original and generated images
            rs_origin = self.survival_model.predict_on_images(images, tabular_data)
            #rs_origin = self.survival_model(tabular_data, images)

            rs_generated = self.survival_model.predict_on_images(x_generated, tabular_data)
            #rs_generated = self.survival_model(tabular_data, x_generated)
            
            self.survival_model.plot_gen_origin_rs(rs_origin=rs_origin,
                                                    rs_generated=rs_generated,
                                                    storage_path=self.logging_params['storage_path'],
                                                    run_name=self.logging_params['run_name'],
                                                    epoch=step)

            # concatenate such that original images and generated images are plotted together!
            samples = np.random.randint(images.shape[0], size=4)
            x_generated = x_generated[samples, :, :, :]
            x_mask = x_mask[samples, :, :, :]
            images = images[samples, :, :, :]
            rs_origin = rs_origin[samples]
            rs_generated = rs_generated[samples]

            # self.survival_model.plot_riskscores(riskscores=rs_origin,
            #                                     storage_path=self.logging_params['storage_path'],
            #                                     run_name=self.logging_params['run_name'],
            #                                     epoch=step)
            # self.survival_model.plot_riskscores(riskscores=rs_generated)

            

            if images.ndim != 5:
                self.visualize_results(x_origin=images,
                                       x_generated=x_generated,
                                       x_mask=x_mask,
                                       rs_origin=rs_origin,
                                       rs_generated=rs_generated,
                                       storage_path=self.logging_params['storage_path'],
                                       run_name=self.logging_params['run_name'], 
                                       step=step)
            else:
                self.visualize_3D_slices(slices=5,
                                         x_origin=images,
                                         x_generated=x_generated,
                                         rs_origin=rs_origin,
                                         rs_generated=rs_generated,
                                         storage_path=self.logging_params['storage_path'],
                                         run_name=self.logging_params['run_name'],
                                         step=step)

    # def test(self, batch):
    #     """
    #     """
    #     images = batch['images'].to(self.device)
    #     tabular_data = batch['tabular_data'].to(self.device)

    #     # derive labels for batch
    #     label_target, label_origin = self.generate_labels(images=images, tabular_data=tabular_data)
    #     # put data on device
    #     label_target = label_target.to(self.device)
    #     label_origin = label_origin.to(self.device)

    #     x_generated = self.G(images, label_target.float())

    #     # classification results: 

    #     return x_generated
        
    def gradient_penalty(self, y, x):
        """
        """
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]
        
        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        
        return torch.mean((dydx_l2norm - 1)**2)

    def adversarial_loss(self, pred, true):
        """
        """
        return nn.BCELoss(pred, true)

    #alpha = 0.9
    def domain_loss(self, preds, target_label, alpha=0.9, tolerance=0.25):
        """Adaptive quantile loss.
        """

        errors = torch.zeros(size=preds.shape)
        losses = []
        for idx in range(preds.shape[0]):
            if target_label[idx, 1] == 1.:
                tol = -1 * tolerance
            else:
                tol = tolerance
            errors[idx] = torch.abs(preds[idx] - tol)
            if target_label[idx, 1] == 1.:
                if preds[idx] > tol:
                    loss = alpha * errors[idx]
                else:
                    loss = (1 - alpha) * errors[idx]
            else:
                if preds[idx] > tol:
                    loss = (1 - alpha) * errors[idx]
                else:
                    loss = alpha * errors[idx]
            losses.append(loss)

        loss = torch.mean(torch.stack(losses))
        
        return loss

    def reconstruction_loss(self, x_real, x_reconstruct):
        """Cyclic reconstruction loss between reconstructed 
        to original domain from generated images from target domain.
        """
        loss = torch.mean(torch.abs(x_real - x_reconstruct))
        return loss


    def generate_labels(self, images, tabular_data):
        """
        """
        prediction = self.survival_model.predict_on_images(images=images, tabular_data=tabular_data)
        log_prediction = prediction
        # try:
        #     log_prediction = torch.log(prediction).squeeze(1)
        # except:
        #     pdb.set_trace()
        label_target = torch.zeros(size=(log_prediction.shape[0], 2))
        label_origin = torch.zeros(size=(log_prediction.shape[0], 2))
        for idx in range(label_target.shape[0]):
            if log_prediction[idx] > 0: 
                label_target[idx, 1] = 1.0
                label_origin[idx, 0] = 1.0 
            else:
                label_target[idx, 0] = 1.0 
                label_origin[idx, 1] = 1.0
    
        return label_target.float(), label_origin.float()

    def visualize_results(self, 
                          x_origin, 
                          x_generated,
                          x_mask,
                          rs_origin,
                          rs_generated,
                          storage_path,
                          run_name,
                          step):
        """
        """
        x_origin = x_origin.cpu().detach().numpy()
        x_generated = x_generated.cpu().detach().numpy()
        x_mask = x_mask.cpu().detach().numpy()
        rs_origin = rs_origin.cpu().detach().numpy()
        rs_generated = rs_generated.cpu().detach().numpy()
        fig, ax = plt.subplots(nrows=x_origin.shape[0], ncols=3, figsize=(10, 10))

        for i in range(len(x_origin)):
            rs_gen = round(rs_generated[i][0], 3)
            rs_orig = round(rs_origin[i][0], 3)
            if x_generated.shape[1] == 1:
                ax[i, 0].imshow(x_generated[i, :, :, :].squeeze(), cmap='gray', vmin=0, vmax=1)
            else:
                shapes = x_generated.shape
                x_gen = x_generated.reshape(-1, shapes[2], shapes[3], shapes[1])
                ax[i, 0].imshow(x_gen[i, :, :, :])

            ax[i, 0].set_title("GI - RS: {:.3f}".format(rs_gen))

            if x_origin.shape[1] == 1:
                ax[i, 1].imshow(x_origin[i, :, :, :].squeeze(), cmap='gray')
            else:
                shapes = x_origin.shape
                x_orig = x_origin.reshape(-1, shapes[2], shapes[3], shapes[1])
                ax[i, 1].imshow(x_orig[i, :, :, :])


            ax[i, 1].set_title("OI - RS: {:.3f}".format(rs_orig))

            if x_mask.shape[1] == 1:
                ax[i, 2].imshow(-x_mask[i, :, :, :].squeeze(), cmap='gray')
            else:
                shapes = x_mask.shape
                x_m = x_mask.reshape(-1, shapes[2], shapes[3], shapes[1])
                ax[i, 2].imshow(x_m[i, :, :, :])

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
