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
from src.data.sim_ped import SimPED
from src.data.sim_coxph import SimCoxPH
from src.postprocessing import plot_train_progress
from src.dsap.dsap import DSAP
from src.dsap.coalition_policies.playergenerators import *
from src.integrated_gradients import IntegratedGradients
from src.baselines.baseline_generator import BaselineGenerator

from src.architectures.network2d import Discriminator2d, Generator2d



class DeepSurvExperiment(pl.LightningModule):
    """
    """
    def __init__(self,
                 model,
                 params,
                 log_params,
                 model_hyperparams,
                 baseline_params,
                 run_name,
                 experiment_name):
        super(DeepSurvExperiment, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model.float().to(self.device)
        self.model.epoch = self.current_epoch
        self.params = params 
        self.log_params = log_params
        self.model_hyperparams = model_hyperparams
        self.baseline_params = baseline_params
        self.train_history = pd.DataFrame()
        self.val_history = pd.DataFrame()
        self.test_history = pd.DataFrame()
        self.run_name = run_name 
        self.experiment_name = experiment_name
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):

        y_pred = self.forward(**batch)
        if self.params['data_type'] == 'coxph':
            train_loss = self.model._loss_function(batch['event'], batch['risket'], predictions=y_pred)
        else:
            train_loss = self.model._loss_function(y_pred, batch['ped_status'])

        # if self.params['data_type'] == 'coxph':
        #     image, tabular_data, event, time = batch
        #     riskset = utils.make_riskset(time)

        #     y_pred = self.forward(tabular_data, image.float())
        #     train_loss = self.model._loss_function(event, riskset, predictions=y_pred)

        # else:
        #     image, tabular_data, offset, ped_status, index, splines = batch
        #     y_pred = self.forward(tabular_data, image.float(), offset, index, splines)
        #     train_loss = self.model._loss_function(y_pred, ped_status)
        
        train_loss = {'loss': train_loss}

        self.train_history = pd.DataFrame([[value.cpu().detach().numpy() for value in train_loss.values()]],
                            columns=[key for key in train_loss.keys()])

        return train_loss

    def validation_step(self, batch, batch_idx):

        y_pred = self.forward(**batch)
        if self.params['data_type'] == 'coxph':
            val_loss = self.model._loss_function(batch['event'], batch['risket'], predictions=y_pred)
        else:
            val_loss = self.model._loss_function(y_pred, batch['ped_status'])
        
        # if self.params['data_type'] == 'coxph':
        #     image, tabular_data, event, time = batch
        #     riskset = utils.make_riskset(time)
            
        #     y_pred = self.forward(tabular_data, image.float())
        #     val_loss = self.model._loss_function(event, riskset, predictions=y_pred)

        # else:
        #     pdb.set_trace()
        #     image, tabular_data, offset, ped_status, index, splines = batch
        #     y_pred = self.forward(tabular_data, image.float(), offset, index, splines)
        #     val_loss = self.model._loss_function(y_pred, ped_status)
        
        val_loss = {'loss': val_loss}

        self.val_history = pd.DataFrame([[value.cpu().detach().numpy() for value in val_loss.values()]],
                        columns=[key for key in val_loss.keys()])

        return val_loss

    def validation_epoch_end(self, outputs):
        
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().to(torch.double)
        avg_loss = avg_loss.cpu().detach().numpy() + 0 

        return {'avg_loss': avg_loss}

    def test_step(self, batch, batch_idx):

        y_pred = self.forward(**batch)
        if self.params['data_type'] == 'coxph':
            test_loss = self.model._loss_function(batch['event'], batch['risket'], predictions=y_pred)
        else:
            test_loss = self.model._loss_function(y_pred, batch['ped_status'])

        # if self.params['data_type'] == 'coxph':
        #     image, tabular_data, event, time = batch
        #     riskset = utils.make_riskset(time)

        #     y_pred = self.forward(tabular_data, image.float())
        #     test_loss = self.model._loss_function(event, riskset, predictions=y_pred)

        # else:
        #     image, tabular_data, offset, ped_status, index, splines = batch
        #     y_pred = self.forward(tabular_data, image.float(), offset, index, splines)
        #     test_loss = self.model._loss_function(y_pred, ped_status)
        
        test_loss = {'loss': test_loss}
        
        self.test_history = pd.DataFrame([[value.cpu().detach().numpy() for value in test_loss.values()]],
                            columns=[key for key in test_loss.keys()])

        return test_loss

    def test_epoch_end(self, outputs):

        # where/when to set grad_enabled = True?
        torch.set_grad_enabled(True)
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().to(torch.double)
        avg_loss = avg_loss.cpu().detach().numpy() + 0 

        try:
            plot_train_progress(self.train_history, 
                                storage_path=f"logs/{self.run_name}/{self.params['dataset']}/training/")
            plot_train_progress(self.val_history, 
                        storage_path=f"logs/{self.run_name}/{self.params['dataset']}/validation/")
        except:
            pass

        accumulated_batch = self.model._accumulate_batches(data=self.test_gen)
        sample_batch = self.model._sample_batch(self.test_gen, num_obs=4)
        eval_data = utils.get_eval_data(batch=accumulated_batch,
                                        model=self.model)
        
        
        evaluation_data = {**accumulated_batch, **self.eval_data, **eval_data}
       
        # get survival cindex and ibs
        self.model.get_metrics(**evaluation_data)
        
        # log metrics
        for key, value in zip(self.model.scores.keys(), self.model.scores.values()):
            self.logger.experiment.log_metric(key=key,
                                              value=value,
                                              run_id=self.logger.run_id)


        # # train baseline generator 
        baseline_generator = BaselineGenerator(discriminator=Discriminator2d,
                                               generator=Generator2d,
                                               survival_model=self.model,
                                               data_type=self.params['data_type'],
                                               c_dim=self.baseline_params['c_dim'],
                                               img_size=self.baseline_params['img_size'],
                                               generator_params=self.baseline_params['generator_params'],
                                               discriminator_params=self.baseline_params['discriminator_params'],
                                               trainer_params=self.baseline_params['trainer_params'])

        # train baseline generator
        baseline_generator.train(train_gen=self.train_gen)
        
        # validate baseline generator
        #baseline_generator.validate(val_gen=self.val_gen)
        
        #baseline_images = baseline_generator.test(batch=accumulated_batch)
        baseline_images = baseline_generator.test(batch=sample_batch)

        ##################################################################################################
        ## derive feature attributions

        #images, tabular_data = images[:4, :, :, ], tabular_data[:4, :]
        # I think, we should determine how many images to be calculated in the attribution method itself!

        #images, tabular_data = accumulated_batch['images'][:4, :, :, ], accumulated_batch['tabular_data'][:4, :]
        
        images, tabular_data = sample_batch['images'], sample_batch['tabular_data']

        # 1.) Calculate Integrated Gradients
        # IG = IntegratedGradients()
        # integrated_gradients = IG.integrated_gradients(model=self.model,
        #                                                images=images,
        #                                                tabular_data=tabular_data,
        #                                                length=9,
        #                                                baseline=baseline_images,
        #                                                storage_path="euclidean_morph",
        #                                                run_name=self.run_name)
        # wasserstein_ig = IG.wasserstein_integrated_gradients(model=self.model,
        #                                                      images=images,
        #                                                      baseline=baseline_images,
        #                                                      length=9, 
        #                                                      tabular_data=tabular_data,
        #                                                      storage_path="wasserstein_morph",
        #                                                      run_name=self.run_name)

        # pdb.set_trace()
        # self.model.visualize_attributions(images=images, 
        #                                   attributions=integrated_gradients,
        #                                   method='integrated_gradient',
        #                                   storage_path="attributions",
        #                                   run_name=self.run_name)

        # self.model.visualize_attributions(images=images,
        #                                   attributions=wasserstein_ig,
        #                                   method='wasserstein_ig',
        #                                   storage_path='attributions',
        #                                   run_name=self.run_name)

        # 2.) Calculate approximate Shapley Values
        lpdn_model = self.model._build_lpdn_model()

        dsap = DSAP(player_generator=WideDeepPlayerIterator(ground_input=(images, tabular_data)),
                    input_shape=images[0].shape,
                    lpdn_model=lpdn_model)
        shapley_attributions = dsap.run(images=images, 
                                        tabular_data=tabular_data,
                                        baselines=baseline_images, 
                                        steps=50)
        self.model.visualize_attributions(images=images, 
                                          attributions=shapley_attributions,
                                          method='shapley',
                                          storage_path='attributions',
                                          run_name=self.run_name)

        # # 3.) Calculate Occlusion
        # occ = Occlusion(player_generator,
        #                 input_shape,
        #                 baseline,
        #                 )
        # occlusion_attributions = 


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
        if self.params['data_type'] == 'coxph':
            train_data = SimCoxPH(root='./data',
                                  part='train',
                                  base_folder=self.params['base_folder'],
                                  data_type=self.params['data_type'])
            self.train_gen = DataLoader(dataset=train_data,
                                   batch_size=self.params['batch_size'],
                                   collate_fn=utils.cox_collate_fn,
                                   shuffle=False)
        else:
            train_data = SimPED(root='./data',
                                part='train',
                                base_folder=self.params['base_folder'],
                                data_type=self.params['data_type'])
            self.train_gen = DataLoader(dataset=train_data, 
                                   batch_size=self.params['batch_size'],
                                   collate_fn=utils.ped_collate_fn,
                                   shuffle=False)

        return self.train_gen

    def val_dataloader(self):
        """
        """
        if self.params['data_type'] == 'coxph':
            val_data = SimCoxPH(root='./data',
                                  part='val',
                                  base_folder=self.params['base_folder'],
                                  data_type=self.params['data_type'])
            self.val_gen = DataLoader(dataset=val_data,
                                   batch_size=self.params['batch_size'],
                                   collate_fn=utils.cox_collate_fn,
                                   shuffle=False)
        else:
            val_data = SimPED(root='./data',
                                part='val',
                                base_folder=self.params['base_folder'],
                                data_type=self.params['data_type'])
            self.val_gen = DataLoader(dataset=val_data, 
                                   batch_size=self.params['batch_size'],
                                   collate_fn=utils.ped_collate_fn,
                                   shuffle=False)
        
        return self.val_gen

    def test_dataloader(self):
        """
        """
        if self.params['data_type'] == 'coxph':
            test_data = SimCoxPH(root='./data',
                                  part='test',
                                  base_folder=self.params['base_folder'],
                                  data_type=self.params['data_type'])
            self.test_gen = DataLoader(dataset=test_data,
                                   batch_size=self.params['batch_size'],
                                   collate_fn=utils.cox_collate_fn,
                                   shuffle=False)

        else:
            test_data = SimPED(root='./data',
                                part='test',
                                base_folder=self.params['base_folder'],
                                data_type=self.params['data_type'])
            self.test_gen = DataLoader(dataset=test_data, 
                                   batch_size=self.params['batch_size'],
                                   collate_fn=utils.ped_collate_fn,
                                   shuffle=False)
        self.eval_data = test_data.eval_data

        return self.test_gen