import pandas as pd
import numpy as np 
import torch 
import pytorch_lightning as pl
import src
import pdb


from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from src.data import utils
from src.postprocessing import plot_train_progress
from src.dsap.dsap import DSAP
from src.dsap.coalition_policies.playergenerators import *
from src.integrated_gradients import IntegratedGradients


class DeepSurvExperiment(pl.LightningModule):
    """
    """
    def __init__(self,
                 model,
                 params,
                 log_params,
                 model_hyperparams,
                 run_name,
                 experiment_name):
        super(DeepSurvExperiment, self).__init__()

        self.model = model.float()
        self.model.epoch = self.current_epoch
        self.params = params 
        self.log_params = log_params
        self.model_hyperparams = model_hyperparams
        self.train_history = pd.DataFrame()
        self.val_history = pd.DataFrame()
        self.test_history = pd.DataFrame()
        self.run_name = run_name 
        self.experiment_name = experiment_name
    
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):

        image, tabular_data, event, time = batch
        riskset = utils.make_riskset(time)

        riskscore = self.forward(tabular_data, image.float())
        train_loss = self.model._loss_function(event, riskset, predictions=riskscore)

        train_history = pd.DataFrame([[value.cpu().detach().numpy() for value in train_loss.values()]],
                                    columns=[key for key in train_loss.keys()])

        return train_loss

    def validation_step(self, batch, batch_idx):

        image, tabular_data, event, time = batch
        riskset = utils.make_riskset(time)
        
        riskscore = self.forward(tabular_data, image.float())
        val_loss = self.model._loss_function(event, riskset, predictions=riskscore)

        val_history = pd.DataFrame([[value.cpu().detach().numpy() for value in val_loss.values()]],
                            columns=[key for key in val_loss.keys()])


        return val_loss

    def validation_epoch_end(self, outputs):
        
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().to(torch.double)
        avg_loss = avg_loss.cpu().detach().numpy() + 0 

        return {'avg_loss': avg_loss}#

    def test_step(self, batch, batch_idx):
        image, tabular_data, event, time = batch
        riskset = utils.make_riskset(time)

        riskscore = self.forward(tabular_data, image.float())
        test_loss = self.model._loss_function(event, riskset, predictions=riskscore)

        test_history = pd.DataFrame([[value.cpu().detach().numpy() for value in test_loss.values()]],
                            columns=[key for key in test_loss.keys()])

        return test_loss

    def test_epoch_end(self, outputs):

        torch.set_grad_enabled(True)
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().to(torch.double)
        avg_loss = avg_loss.cpu().detach().numpy() + 0 

        try:
            plot_train_progress(self.train_history, 
                                storage_path=f"logs/{self.run_name}/{self.params['dataset']}/training/")
            plot_train_progress(self.train_history, 
                        storage_path=f"logs/{self.run_name}/{self.params['dataset']}/validation/")
        except:
            pass

        images, tabular_data, events, times = self.model.accumulate_batches(data=self.test_gen)
        #riskset = utils.make_riskset(times)
        riskscores = self.forward(tabular_data, images.float())

        self.model.get_measures(riskscore=riskscores,
                                events=events,
                                times=times)
        
        
        ## derive feature attributions
        images, tabular_data = images[:4, :, :, ], tabular_data[:4, :]
        
        IG = IntegratedGradients()
        integrated_gradients = IG.integrated_gradients(model=self.model,
                                                       images=images,
                                                       tabular_data=tabular_data)
        wasserstein_ig = IG.wasserstein_integrated_gradients(model=self.model,
                                                             images=images,
                                                             img2=images[1, :, :, :],
                                                             length=9, 
                                                             tabular_data=tabular_data,
                                                             storage_path="wasserstein_morph",
                                                             run_name=self.run_name)

        self.model.visualize_attributions(images=images, 
                                          attributions=integrated_gradients,
                                          method='integrated_gradient',
                                          storage_path="attributions",
                                          run_name=self.run_name)

        self.model.visualize_attributions(images=images,
                                          attributions=wasserstein_ig,
                                          method='wasserstein_ig',
                                          storage_path='attributions',
                                          run_name=self.run_name)

        lpdn_model = self.model._build_lpdn_model()

        dsap = DSAP(player_generator=WideDeepPlayerIterator(ground_input=(images, tabular_data)),
                    input_shape=images[0].shape,
                    lpdn_model=lpdn_model)
        shapley_attributions = dsap.run(images=images, tabular_data=tabular_data, steps=50)
        self.model.visualize_attributions(images=images, 
                                          attributions=shapley_attributions,
                                          method='shapley',
                                          storage_path='attributions',
                                          run_name=self.run_name)

        return {'avg_loss': avg_loss}

    def configure_optimizers(self):
        """
        """
        
        optims = []
        scheds = []
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['learning_rate'],
                               weight_decay=self.params['weight_decay'])

        optims.append(optimizer)

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma=self.params['scheduler_gamma'])
                scheds.append(scheduler)
                return optims, scheds
        except:
            return optims



    def train_dataloader(self):
        """
        """

        path = f"{self.params['data_path']}"
        X, df = utils.load_data(path=path, split='train')

        train_data = utils.ImageData(features=X, df=df)

        train_gen = DataLoader(dataset=train_data,
                               batch_size=self.params['batch_size'],
                               shuffle=False)

        return train_gen

    def val_dataloader(self):
        """
        """

        path = f"{self.params['data_path']}"
        X, df = utils.load_data(path=path, split='val')

        val_data = utils.ImageData(features=X, df=df)

        self.val_gen = DataLoader(dataset=val_data,
                               batch_size=self.params['batch_size'],
                               shuffle=False)
        
        return self.val_gen

    def test_dataloader(self):
        """
        """

        path = f"{self.params['data_path']}"
        X, df = utils.load_data(path=path, split='test')

        test_data = utils.ImageData(features=X, df=df)

        self.test_gen = DataLoader(dataset=test_data,
                               batch_size=self.params['batch_size'],
                               shuffle=False)
        
        return self.test_gen