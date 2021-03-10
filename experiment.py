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
from src.data.utils import get_dataloader
from src.data.adni import ADNI
from src.data.sim_coxph import SimCoxPH
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
        #pdb.set_trace()
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

        # where/when to set grad_enabled = True?
        torch.set_grad_enabled(True)

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().to(torch.double)
        avg_loss = avg_loss.cpu().detach().numpy() + 0 

        try:
            plot_train_progress(self.train_history, 
                                storage_path=f"logs/{self.run_name}/{self.params['dataset']}/training/")
            plot_train_progress(self.val_history, 
                                storage_path=f"logs/{self.run_name}/{self.params['dataset']}/validation/")
            plot_train_progress(self.val_cindex, 
                                storage_path=f"logs/{self.run_name}/{self.params['dataset']}/val_cindex")
            # plot_train_progress(self.train_cindex, 
            #                     storage_path=f"logs/{self.run_name}/{self.params['dataset']}/train_cindex")
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

            del data_gen, eval_data1, accumulated_batch, eval_data, evaluation_data


        # print("EVALUATION!!!!!!")
        # if self.params['base_folder'] == "adni_slices":
        #     pass
        # else:
        #     train_gen, eval_data_train1 = get_dataloader(root='./data',
        #                                                 part='train',
        #                                                 transform=False,
        #                                                 base_folder=self.params['base_folder'],
        #                                                 data_type=self.params['data_type'],
        #                                                 batch_size=-1,
        #                                                 split=self.split)
        #     accumulated_batch_train = self.model._accumulate_batches(data=train_gen)
        #     eval_data_train = utils.get_eval_data(batch=accumulated_batch_train,
        #                                           model=self.model)
        #     evaluation_data_train = {**accumulated_batch_train, **eval_data_train1, **eval_data_train}
        #     self.model.get_metrics(**evaluation_data_train, part='train')

        #     #del train_gen, eval_data_train1, accumulated_batch_train, eval_data_train, evaluation_data_train


        # val_gen, eval_data_val1 = get_dataloader(root='./data',
        #                                         part='val',
        #                                         transform=False,
        #                                         base_folder=self.params['base_folder'],
        #                                         data_type=self.params['data_type'],
        #                                         batch_size=-1,
        #                                         split=self.split)
        # accumulated_batch_val = self.model._accumulate_batches(data=val_gen)
        # eval_data_val = utils.get_eval_data(batch=accumulated_batch_val,
        #                                     model=self.model)
        # evaluation_data_val = {**accumulated_batch_val, **eval_data_val1, **eval_data_val}
        # self.model.get_metrics(**evaluation_data_val, part='val')

        # #del val_gen, eval_data_val1, accumulated_batch_val, eval_data_val, evaluation_data_val

        # test_gen, eval_data_test1 = get_dataloader(root='./data',
        #                                             part='test',
        #                                             transform=False,
        #                                             base_folder=self.params['base_folder'],
        #                                             data_type=self.params['data_type'],
        #                                             batch_size=-1,
        #                                             split=self.split)        
        # accumulated_batch_test = self.model._accumulate_batches(data=test_gen)
        # eval_data_test = utils.get_eval_data(batch=accumulated_batch_test,
        #                                      model=self.model)
        # evaluation_data_test = {**accumulated_batch_test, **eval_data_test1, **eval_data_test}
        # self.model.get_metrics(**evaluation_data_test, part='test')
        
        #del test_gen, eval_data_test1, accumulated_batch_test, eval_data_test, evaluation_data_test

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
        

        # store model 
        if self.model_name == "Baseline":
            torch.save(self.model.deep.state_dict(), "model_weights")
            return {'avg_loss': avg_loss}

        print("BASELINE GENERATOR")
        train_gen, _ = get_dataloader(root='./data',
                                      part='train',
                                      transform=False,
                                      base_folder=self.params['base_folder'],
                                      data_type=self.params['data_type'],
                                      batch_size=self.params['batch_size'],
                                      split=self.split)
        
        val_gen, _ = get_dataloader(root='./data',
                                    part='val',
                                    transform=False,
                                    base_folder=self.params['base_folder'],
                                    data_type=self.params['data_type'],
                                    batch_size=self.params['batch_size'],
                                    split=self.split)

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
                                               logging_params=self.baseline_params['logging_params'])

        baseline_generator.train(train_gen=train_gen, val_gen=val_gen)
        pdb.set_trace()
        # validate baseline generator
        baseline_generator.validate(val_gen=self.val_gen)
        
        baseline_images = baseline_generator.test(batch=accumulated_batch_test)
        baseline_images = baseline_generator.test(batch=sample_batch)
        
        zero_baseline_images = torch.zeros(sample_batch['images'].shape).to(self.new_device).float()

        # ##################################################################################################
        # ## derive feature attributions
        
        # images, tabular_data = sample_batch['images'], sample_batch['tabular_data']

        # # 1.) Calculate Integrated Gradients
        # IG = IntegratedGradients()
        # integrated_gradients = IG.integrated_gradients(model=self.model,
        #                                                images=images,
        #                                                tabular_data=tabular_data,
        #                                                length=9,
        #                                                baseline=baseline_images,
        #                                                storage_path="euclidean_morph",
        #                                                run_name=self.run_name)

        # self.model.visualize_attributions(images=images, 
        #                                   attributions=integrated_gradients,
        #                                   method='integrated_gradient',
        #                                   storage_path="attributions",
        #                                   run_name=self.run_name)

        # wasserstein_ig = IG.wasserstein_integrated_gradients(model=self.model,
        #                                                      images=images,
        #                                                      baseline=baseline_images,
        #                                                      length=9, 
        #                                                      tabular_data=tabular_data,
        #                                                      storage_path="wasserstein_morph",
        #                                                      run_name=self.run_name)


        # self.model.visualize_attributions(images=images,
        #                                   attributions=wasserstein_ig,
        #                                   method='wasserstein_ig',
        #                                   storage_path='attributions',
        #                                   run_name=self.run_name)

        # # 2.) Calculate approximate Shapley Values
        # lpdn_model = self.model._build_lpdn_model()

        # dsap = DSAP(player_generator=WideDeepPlayerIterator(ground_input=(images, tabular_data), windows=False),
        #             input_shape=images[0].shape,
        #             lpdn_model=lpdn_model)
        # shapley_attributions = dsap.run(images=images, 
        #                                 tabular_data=tabular_data,
        #                                 baselines=baseline_images, 
        #                                 steps=50)
        # self.model.visualize_attributions(images=images, 
        #                                   attributions=shapley_attributions,
        #                                   method='shapley',
        #                                   storage_path='attributions',
        #                                   run_name=self.run_name)

        # # 3.) Calculate Occlusion
        # occlusion = Occlusion(model=self.model,
        #                       player_generator=WideDeepPlayerIterator(ground_input=(images, tabular_data), 
        #                                                               windows=False))
        # occ_attributions = occlusion.run(images=images,
        #                                  tabular_data=tabular_data,
        #                                  baselines=baseline_images)
        # self.model.visualize_attributions(images=images,
        #                                   attributions=occ_attributions,
        #                                   method='occlusion',
        #                                   storage_path='attributions',
        #                                   run_name=self.run_name)


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
                linear_coefficients = np.load(file="./linear_weights/weights.npy").astype('float64')
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
                train_data = SimCoxPH(root='./data',
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
                val_data = SimCoxPH(root='./data',
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
                test_data = SimCoxPH(root='./data',
                                    part='test',
                                    base_folder=self.params['base_folder'],
                                    data_type=self.params['data_type'],
                                    n_dim=self.params['n_dim'])
                test_gen = DataLoader(dataset=test_data,
                                    batch_size=self.params['batch_size'],
                                    collate_fn=utils.cox_collate_fn,
                                    shuffle=False)
        
        return test_gen
