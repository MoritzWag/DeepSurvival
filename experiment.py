import pandas as pd
import numpy as np 
import torch 
import pytorch_lightning as pl
import src
import pdb
import os
import torch.nn as nn

from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image

from src.data import utils
from src.data.utils import get_dataloader, generated_colored_bs_img
from src.data.adni import ADNI
# from src.data.sim_coxph import SimCoxPH
from src.data.sim_mnist import SimMNIST
from src.data.sim_images import SimImages
from src.postprocessing import plot_train_progress
from src.dsap.dsap import DSAP
from src.dsap.coalition_policies.playergenerators import *
from src.integrated_gradients import IntegratedGradients
from src.baselines.baseline_generator import BaselineGenerator
from src.occlusion import Occlusion
from src.helpers import get_optimizer

from src.architectures.SIM.discriminators import DiscriminatorSIM
from src.architectures.ADNI.discriminators import DiscriminatorADNI
from src.architectures.SIM.map_generator import GeneratorSIM
from src.architectures.ADNI.map_generator import GeneratorADNI

from src.architectures.utils import weight_init


class DeepSurvExperiment(pl.LightningModule):
    """
    """
    def __init__(self,
                 model,
                 params,
                 log_params,
                 model_hyperparams,
                 baseline_params,
                 split,
                 tune,
                 model_name,
                 run_name,
                 experiment_name,
                 trial=None):
        super(DeepSurvExperiment, self).__init__()

        self.new_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model.float()
        self.model.epoch = self.current_epoch
        self.params = params 
        self.log_params = log_params
        self.model_hyperparams = model_hyperparams
        self.baseline_params = baseline_params
        self.train_history = pd.DataFrame()
        self.val_history = pd.DataFrame()
        self.train_cindex = pd.DataFrame()
        self.val_cindex = pd.DataFrame()
        self.test_history = pd.DataFrame()
        self.model_name = model_name
        self.run_name = run_name 
        self.experiment_name = experiment_name
        self.trial = trial

        # riskscore tracking
        self.rs_train = []
        self.rs_val = []

        # initialize parameters for baseline generation
        self.discriminator = DiscriminatorADNI if self.params['dataset'] == 'adni' else DiscriminatorSIM
        self.generator = GeneratorADNI if self.params['dataset'] == 'adni' else GeneratorSIM
        
        # initialize model
        self.split = split
        self.tune = tune
        self.model = self.weight_init(self.model).to(self.new_device).float()
        
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        y_pred = self.forward(**batch)

        train_loss = self.model._loss_function(batch['event'], batch['riskset'], predictions=y_pred)

        train_loss = {'loss': train_loss}

        train_history = pd.DataFrame([[value.cpu().detach().numpy() for value in train_loss.values()]],
                            columns=[key for key in train_loss.keys()])

        self.train_history = self.train_history.append(train_history, ignore_index=True)
        
        return train_loss

    def train_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().to(torch.double)
        avg_loss = avg_loss.cpu().detach().numpy() + 0 

        return {'avg_train_loss': avg_loss}
    
    def validation_step(self, batch, batch_idx):
        y_pred = self.forward(**batch)
        val_loss = self.model._loss_function(batch['event'], batch['riskset'], predictions=y_pred)
        
        val_loss = {'loss': val_loss}

        val_history = pd.DataFrame([[value.cpu().detach().numpy() for value in val_loss.values()]],
                        columns=[key for key in val_loss.keys()])
        
        self.val_history = self.val_history.append(val_history, ignore_index=True)

        return val_loss

    def validation_epoch_end(self, outputs):
        
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().to(torch.double)
        avg_loss = avg_loss.cpu().detach().numpy() + 0 

        try:
            self.logger.experiment.log_metric(key='avg_val_loss',
                                            value=avg_loss,
                                            run_id=self.logger.run_id)
        except:
            pass
        
        if self.tune:
            # calculate c-index for tuning!
            val_gen, eval_data_val1 = get_dataloader(root='./data',
                                                    part='val',
                                                    transform=False,
                                                    base_folder=self.params['base_folder'],
                                                    data_type=self.params['data_type'],
                                                    batch_size=-1,
                                                    split=self.split)

            accumulated_batch_val = self.model._accumulate_batches(data=val_gen)
            eval_data_val = utils.get_eval_data(batch=accumulated_batch_val, 
                                                model=self.model)
            evaluation_data_val = {**accumulated_batch_val, **eval_data_val1, **eval_data_val}

            c_index = self.model.concordance_index(event=evaluation_data_val['event'].cpu().detach().numpy().astype(bool),
                                                   time=evaluation_data_val['time'].cpu().detach().numpy(),
                                                   riskscores=evaluation_data_val['riskscores'])

            val_cindex = pd.DataFrame([c_index[0]], columns=['cindex'])
            
            self.val_cindex = self.val_cindex.append(val_cindex, ignore_index=True)

            self.log('cindex_val', c_index[0])

            del accumulated_batch_val, eval_data_val, evaluation_data_val

            return {'log': {'cindex': c_index[0]}}

        return {'avg_loss': avg_loss}

    def test_step(self, batch, batch_idx):

        y_pred = self.forward(**batch)
        test_loss = self.model._loss_function(batch['event'], batch['riskset'], predictions=y_pred)
        
        test_loss = {'loss': test_loss}

        test_history = pd.DataFrame([[value.cpu().detach().numpy() for value in test_loss.values()]],
                            columns=[key for key in test_loss.keys()])
        
        self.test_history = self.test_history.append(test_history, ignore_index=True)

        return test_loss

    def test_epoch_end(self, outputs):

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().to(torch.double)
        avg_loss = avg_loss.cpu().detach().numpy() + 0 

        try:
            plot_train_progress(self.train_history, 
                                storage_path=f"logs/{self.run_name}/{self.params['dataset']}/training/")
            plot_train_progress(self.val_history, 
                                storage_path=f"logs/{self.run_name}/{self.params['dataset']}/validation/")
            plot_train_progress(self.val_cindex, 
                                storage_path=f"logs/{self.run_name}/{self.params['dataset']}/val_cindex")
        except:
            pass
        
        for part in ['train', 'val', 'test']:
            if self.params['base_folder'] == "adni_slices" and part == 'train':
                continue 

            data_gen, eval_data1 = get_dataloader(root='./data',
                                                 part=part,
                                                 transform=False,
                                                 base_folder=self.params['base_folder'],
                                                 data_type=self.params['data_type'],
                                                 batch_size=-1,
                                                 split=self.split)
            accumulated_batch = self.model._accumulate_batches(data=data_gen)
            eval_data = utils.get_eval_data(batch=accumulated_batch,
                                            model=self.model)
            evaluation_data = {**accumulated_batch, **eval_data1, **eval_data}
            self.model.get_metrics(**evaluation_data, part=part)

            self.model.plot_riskscores(evaluation_data['riskscores'], storage_path=f"./hist/{part}_data", run_name=self.run_name, epoch=self.current_epoch)

            del data_gen, eval_data1, accumulated_batch, eval_data, evaluation_data
        
        # log metrics
        try:
            self.logger.experiment.log_metric(key='avg_test_loss',
                                            value=avg_loss,
                                            run_id=self.logger.run_id)

            for key, value in zip(self.model.scores.keys(), self.model.scores.values()):
                self.log(key, value)

            for key, value in zip(self.model.scores.keys(), self.model.scores.values()):
                self.logger.experiment.log_metric(key=key,
                                                value=value,
                                                run_id=self.logger.run_id)

            for _name, _param in zip(self.params.keys(), self.params.values()):
                self.logger.experiment.log_param(key=_name,
                                                value=_param,
                                                run_id=self.logger.run_id)
            
            self.logger.experiment.log_param(key='run_name',
                                            value=self.run_name,
                                            run_id=self.logger.run_id)
            self.logger.experiment.log_param(key='experiment_name',
                                            value=self.experiment_name,
                                            run_id=self.logger.run_id)
            self.logger.experiment.log_param(key='manual_seed',
                                            value=self.log_params['manual_seed'],
                                            run_id=self.logger.run_id)
        except:
            pass
        
        print(self.model.scores)

        if self.params['dataset'] == 'adni':
            storage_path = os.path.expanduser(f'linear_weights/{self.run_name}')
            if not os.path.exists(storage_path):
                os.makedirs(storage_path)
            coefficients =  self.model.linear.weight.data[:, :self.model.structured_input_dim].cpu().detach().numpy()
            np.save(file=f"{storage_path}/weights_deep_{self.split}.npy", arr=coefficients)    
            return {'avg_loss': avg_loss}

        # store model
        storage_path = os.path.expanduser(f'survival_model')
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)
        torch.save(self.model.state_dict(), f"{storage_path}/sm_{self.run_name}")

        # where/when to set grad_enabled = True?
        torch.set_grad_enabled(True)

        print("BASELINE GENERATOR")
        train_gen, _ = get_dataloader(root='./data',
                                      part='train',
                                      transform=False,
                                      base_folder=self.params['base_folder'],
                                      data_type=self.params['data_type'],
                                      batch_size=self.params['batch_size'],
                                      split=self.split,
                                      cox_collate=False)
        
        val_gen, _ = get_dataloader(root='./data',
                                    part='val',
                                    transform=False,
                                    base_folder=self.params['base_folder'],
                                    data_type=self.params['data_type'],
                                    batch_size=self.params['batch_size'],
                                    split=self.split,
                                    cox_collate=False)

        test_gen, _ = get_dataloader(root='./data',
                                     part='test',
                                     transform=False,
                                     base_folder=self.params['base_folder'],
                                     data_type=self.params['data_type'],
                                     batch_size=self.params['batch_size'],
                                     split=self.split,
                                     cox_collate=False)
        
        # train baseline generator 
        baseline_generator = BaselineGenerator(discriminator=self.discriminator,
                                               generator=self.generator,
                                               survival_model=self.model,
                                               data_type=self.params['data_type'],
                                               c_dim=self.baseline_params['c_dim'],
                                               img_size=self.baseline_params['img_size'],
                                               generator_params=self.baseline_params['generator_params'],
                                               discriminator_params=self.baseline_params['discriminator_params'],
                                               trainer_params=self.baseline_params['trainer_params'],
                                               logging_params=self.baseline_params['logging_params'],
                                               rgb_trained=False if self.params['base_folder'] in ['sim_cont', 'sim_cont_mult'] else True)

        baseline_generator.train(train_gen=train_gen, val_gen=val_gen, test_gen=test_gen)

        return {'avg_test_loss': avg_loss}

    def weight_init(self, model):
        """
        """
        # this is not completely right!
        if self.params['dataset'] == 'adni':
            if self.model_name == "Baseline":
                model.apply(weight_init)
            else:
                model.apply(weight_init)
                os.system(f"python ./sksurv_train.py --download True --seed {self.log_params['manual_seed']} --experiment_name {self.experiment_name} --run_name {self.run_name} --split {self.split}")
                linear_coefficients = np.load(file=f"./linear_weights/weights_{self.split}.npy").astype('float64')
                linear_coefficients = np.expand_dims(linear_coefficients, axis=0)

            try:
                model.linear.weight.data[:, :model.structured_input_dim] = nn.Parameter(torch.FloatTensor(linear_coefficients),
                                                                                        requires_grad=True)

            except:
                pass

        return model

    def configure_optimizers(self):
        """
        """
        
        optims = []
        scheds = []
        print(self.params['learning_rate'])

        if self.trial is None:
            if self.params['optimizer'] == 'AdamW':
                optimizer = get_optimizer(model=self.model,
                                          lr=self.params['learning_rate'],
                                          l2_penalty=self.params['weight_decay'],
                                          optimizer=optim.AdamW)
                optims.append(optimizer)
            if self.params['optimizer'] == 'SGD':
                optimizer = get_optimizer(model=self.model,
                                          lr=self.params['learning_rate'],
                                          l2_penalty=self.params['weight_decay'],
                                          optimizer=optim.SGD)
                optims.append(optimizer)

        else:
            optimizer = get_optimizer(model=self.model,
                                      lr=self.trial.suggest_loguniform("lr", 1e-5, 2e-2),
                                      l2_penalty=self.trial.suggest_uniform("weight_decay", 0.00001, 20.0),
                                      optimizer=optim.AdamW)
            optims.append(optimizer)

        try:
            if self.params['scheduler_gamma'] is not None:
                if self.trial is None:
                    scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                                gamma=self.params['scheduler_gamma'])
                    scheds.append(scheduler)
                    return optims, scheds
                else:
                    scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                                 gamma=self.trial.suggest_uniform('scheduler_gamma', 0.99, 0.999))
                    scheds.append(scheduler)
                    return optims, scheds
        except:
            return optims

    def train_dataloader(self):
        """
        """


        if self.params['dataset'] == 'adni':
            train_data = ADNI(root='./data',
                              part='train',
                              transform=self.params['transforms'],
                              download=True,
                              base_folder=self.params['base_folder'],
                              data_type=self.params['data_type'],
                              simulate=self.params['simulate'],
                              split=self.split,
                              seed=self.log_params['manual_seed'],
                              trial=None)
            train_gen = DataLoader(dataset=train_data,
                                        batch_size=self.params['batch_size'],
                                        collate_fn=utils.cox_collate_fn,
                                        shuffle=True)
        else:
            if self.params['data_type'] == 'coxph':
                if self.params['base_folder'] in ['mnist', 'mnist3d']:
                    train_data = SimMNIST(root='./data',
                                          part='train',
                                          base_folder=self.params['base_folder'],
                                          data_type=self.params['data_type'],
                                          n_dim=self.params['n_dim'])
                else:
                    train_data = SimImages(root='./data',
                                           part='train',
                                           base_folder=self.params['base_folder'],
                                           data_type=self.params['data_type'],
                                           n_dim=self.params['n_dim'])
                train_gen = DataLoader(dataset=train_data,
                                            batch_size=self.params['batch_size'],
                                            collate_fn=utils.cox_collate_fn,
                                            shuffle=False)

        return train_gen


    def val_dataloader(self):
        """
        """
        if self.params['dataset'] == 'adni':
            val_data = ADNI(root='./data',
                             part='val',
                             transform=False,
                             download=True,
                             base_folder=self.params['base_folder'],
                             data_type=self.params['data_type'],
                             simulate=self.params['simulate'],
                             split=self.split,
                             seed=self.log_params['manual_seed'])
            val_gen = DataLoader(dataset=val_data,
                                       batch_size=len(val_data),
                                       collate_fn=utils.cox_collate_fn,
                                       shuffle=False)
        else:
            if self.params['data_type'] == 'coxph':
                if self.params['base_folder'] in ['mnist', 'mnist3d']:
                    val_data = SimMNIST(root='./data',
                                        part='val',
                                        base_folder=self.params['base_folder'],
                                        data_type=self.params['data_type'],
                                        n_dim=self.params['n_dim'])
                else:
                    val_data = SimImages(root='./data',
                                         part='val',
                                         base_folder=self.params['base_folder'],
                                         data_type=self.params['data_type'],
                                         n_dim=self.params['n_dim'])
                val_gen = DataLoader(dataset=val_data,
                                    batch_size=self.params['batch_size'],
                                    collate_fn=utils.cox_collate_fn,
                                    shuffle=False)
        
        return val_gen

    def test_dataloader(self):
        """
        """
        if self.params['dataset'] == 'adni':
            test_data = ADNI(root='./data',
                             part='test',
                             transform=False,
                             download=True,
                             base_folder=self.params['base_folder'],
                             data_type=self.params['data_type'],
                             simulate=self.params['simulate'],
                             split=self.split,
                             seed=self.log_params['manual_seed'])
            test_gen = DataLoader(dataset=test_data,
                                       batch_size=len(test_data),
                                       collate_fn=utils.cox_collate_fn,
                                       shuffle=False)
        else:
            if self.params['data_type'] == 'coxph':
                if self.params['base_folder'] in ['mnist', 'mnist3d']:
                    test_data = SimMNIST(root='./data',
                                          part='test',
                                          base_folder=self.params['base_folder'],
                                          data_type=self.params['data_type'],
                                          n_dim=self.params['n_dim'])
                else:
                    test_data = SimImages(root='./data',
                                           part='test',
                                           base_folder=self.params['base_folder'],
                                           data_type=self.params['data_type'],
                                           n_dim=self.params['n_dim'])
                test_gen = DataLoader(dataset=test_data,
                                    batch_size=self.params['batch_size'],
                                    collate_fn=utils.cox_collate_fn,
                                    shuffle=False)
        
        return test_gen
