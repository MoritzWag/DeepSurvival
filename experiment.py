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
from src.data.sim_ped import SimPED
from src.data.sim_coxph import SimCoxPH
from src.postprocessing import plot_train_progress
from src.dsap.dsap import DSAP
from src.dsap.coalition_policies.playergenerators import *
from src.integrated_gradients import IntegratedGradients
from src.baselines.baseline_generator import BaselineGenerator
from src.occlusion import Occlusion
from src.helpers import get_optimizer

from src.architectures.SIM.network2d import Discriminator2d, Generator2d
from src.architectures.SIM.network3d import Discriminator3d, Generator3d

from src.architectures.ADNI.map_generator import UNet
from src.architectures.ADNI.discriminators import Discriminator
from src.architectures.ADNI.utils import weight_init


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
                 model_name,
                 run_name,
                 experiment_name,
                 trial=None):
        super(DeepSurvExperiment, self).__init__()

        self.new_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #self.model = model.float().to(self.new_device)
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

        # initialize model
        self.model = self.weight_init(self.model).to(self.new_device).float()
        self.split = split

    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):

        y_pred = self.forward(**batch)
        if self.params['data_type'] == 'coxph':
            train_loss = self.model._loss_function(batch['event'], batch['riskset'], predictions=y_pred)
            #train_loss = self.model._loss_function(y_pred, time=batch['time'], event=batch['event'])
        else:
            train_loss = self.model._loss_function(y_pred, batch['ped_status'])
        
        train_loss = {'loss': train_loss}

        train_history = pd.DataFrame([[value.cpu().detach().numpy() for value in train_loss.values()]],
                            columns=[key for key in train_loss.keys()])

        self.train_history = self.train_history.append(train_history, ignore_index=True)

        # if self.current_epoch % 10 == 0 and batch_idx == 0:
        #     self.model.plot_riskscores(riskscores=y_pred,
        #                                storage_path="train_histograms",
        #                                run_name=self.run_name,
        #                                epoch=self.current_epoch)

        return train_loss

    def train_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().to(torch.double)
        avg_loss = avg_loss.cpu().detach().numpy() + 0 

        #self.log('avg_train_loss', avg_loss)
        # self.logger.experiment.log_metric(key='avg_train_loss',
        #                                   value=avg_loss,
        #                                   run_id=self.logger.run_id)

        # # calculate c-index
        # accumulated_batch_train = self.model._accumulate_batches(data=self.train_gen)
        # eval_data_train = utils.get_eval_data(batch=accumulated_batch_train, 
        #                                       model=self.model)
        # evaluation_data_val = {**accumulated_batch_val, **self.eval_data_train, **eval_data_train}

        
        # c_index = self.model.concordance_index(event=evaluation_data_train['event'].cpu().detach().numpy().astype(bool),
        #                                        time=evaluation_data_train['time'].cpu().detach().numpy(),
        #                                        riskscores=evaluation_data_train['riskscores'])

        # train_cindex = pd.DataFrame([c_index[0]], columns=['cindex'])
        
        # self.train_cindex = self.train_cindex.append(train_cindex, ignore_index=True)

        # self.log('cindex_train', c_index[0])

        return {'avg_train_loss': avg_loss}
    
    def validation_step(self, batch, batch_idx):
        y_pred = self.forward(**batch)
        if self.params['data_type'] == 'coxph':
            val_loss = self.model._loss_function(batch['event'], batch['riskset'], predictions=y_pred)
            #val_loss = self.model._loss_function(y_pred, time=batch['time'], event=batch['event'])
        else:
            val_loss = self.model._loss_function(y_pred, batch['ped_status'])
        
        val_loss = {'loss': val_loss}

        val_history = pd.DataFrame([[value.cpu().detach().numpy() for value in val_loss.values()]],
                        columns=[key for key in val_loss.keys()])
        
        self.val_history = self.val_history.append(val_history, ignore_index=True)

        # if self.current_epoch % 10 == 0 and batch_idx == 0:
        #     self.model.plot_riskscores(riskscores=y_pred,
        #                                storage_path="val_historgrams",
        #                                run_name=self.run_name,
        #                                epoch=self.current_epoch)
        #     try:
        #         img_riskscore = self.model.predict_on_images(**batch)
        #         self.model.plot_riskscores(riskscores=img_riskscore,
        #                                 storage_path="img_val_hist",
        #                                 run_name=self.run_name,
        #                                 epoch=self.current_epoch)
        #     except:
        #         pass

        return val_loss

    def validation_epoch_end(self, outputs):
        
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().to(torch.double)
        avg_loss = avg_loss.cpu().detach().numpy() + 0 

        self.log('avg_val_loss', avg_loss)
        self.logger.experiment.log_metric(key='avg_val_loss',
                                          value=avg_loss,
                                          run_id=self.logger.run_id)

        # # calculate c-index for tuning!
        accumulated_batch_val = self.model._accumulate_batches(data=self.val_gen)
        eval_data_val = utils.get_eval_data(batch=accumulated_batch_val, 
                                            model=self.model)
        evaluation_data_val = {**accumulated_batch_val, **self.eval_data_val, **eval_data_val}

        
        c_index = self.model.concordance_index(event=evaluation_data_val['event'].cpu().detach().numpy().astype(bool),
                                               time=evaluation_data_val['time'].cpu().detach().numpy(),
                                               riskscores=evaluation_data_val['riskscores'])

        val_cindex = pd.DataFrame([c_index[0]], columns=['cindex'])
        
        self.val_cindex = self.val_cindex.append(val_cindex, ignore_index=True)

        self.log('cindex_val', c_index[0])

        try: 
            #return {'log': {'avg_loss': avg_loss}}
            return {'log': {'cindex': c_index[0]}}
        except:
            return {'avg_loss': avg_loss}

    def test_step(self, batch, batch_idx):

        y_pred = self.forward(**batch)
        if self.params['data_type'] == 'coxph':
            test_loss = self.model._loss_function(batch['event'], batch['riskset'], predictions=y_pred)
            #test_loss = self.model._loss_function(y_pred, time=batch['time'], event=batch['event'])
        else:
            test_loss = self.model._loss_function(y_pred, batch['ped_status'])
        
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

        self.log('avg_test_loss', avg_loss)
        self.logger.experiment.log_metric(key='avg_test_loss',
                                          value=avg_loss,
                                          run_id=self.logger.run_id)

        try:
            plot_train_progress(self.train_history, 
                                storage_path=f"logs/{self.run_name}/{self.params['dataset']}/training/")
            plot_train_progress(self.val_history, 
                                storage_path=f"logs/{self.run_name}/{self.params['dataset']}/validation/")
            plot_train_progress(self.val_cindex, 
                                storage_path=f"logs/{self.run_name}/{self.params['dataset']}/val_cindex")
            plot_train_progress(self.train_cindex, 
                                storage_path=f"logs/{self.run_name}/{self.params['dataset']}/train_cindex")
        except:
            pass
        
        accumulated_batch = self.model._accumulate_batches(data=self.test_gen)
        accumulated_batch_train = self.model._accumulate_batches(data=self.train_gen)
        sample_batch = self.model._sample_batch(self.test_gen, num_obs=4)
        eval_data = utils.get_eval_data(batch=accumulated_batch,
                                        model=self.model)
        eval_data_train = utils.get_eval_data(batch=accumulated_batch_train,
                                              model=self.model)

        # self.model.plot_riskscores(riskscores=eval_data['riskscores'],
        #                            storage_path='histograms',
        #                            run_name=self.run_name,
        #                            epoch=self.params['max_epochs'])
        # try:
        #     self.model.plot_riskscores(riskscores=eval_data['riskscore_img'],
        #                             storage_path='histo_img',
        #                             run_name=self.run_name,
        #                             epoch=self.params['max_epochs'])
        # except:
        #     pass
        
        evaluation_data = {**accumulated_batch, **self.eval_data, **eval_data}
        evaluation_data_train = {**accumulated_batch_train, **self.eval_data_train, **eval_data_train}
       
        # get survival cindex and ibs
        # ibs is very slow on MNIST
        self.model.get_metrics(**evaluation_data, part='test')
        self.model.get_metrics(**evaluation_data_train, part='train')
        
        # # log metrics
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
        

        # store model 
        if self.model_name == "Baseline":
            torch.save(self.model.deep.state_dict(), "model_weights")
            return {'avg_loss': avg_loss}

        #self.log('cindex_test', self.model.scores['cindex_test'][0])
        
        # # train baseline generator 
        # baseline_generator = BaselineGenerator(discriminator=Discriminator,
        #                                        generator=UNet,
        #                                        survival_model=self.model,
        #                                        data_type=self.params['data_type'],
        #                                        c_dim=self.baseline_params['c_dim'],
        #                                        img_size=self.baseline_params['img_size'],
        #                                        generator_params=self.baseline_params['generator_params'],
        #                                        discriminator_params=self.baseline_params['discriminator_params'],
        #                                        trainer_params=self.baseline_params['trainer_params'],
        #                                        logging_params=self.baseline_params['logging_params'])

        # # train baseline generator
        # train_gen = get_dataloader(root='./data',
        #                            part='train',
        #                            transform=False,
        #                            base_folder=self.params['base_folder'],
        #                            data_type=self.params['data_type'],
        #                            batch_size=self.params['batch_size'])
        
        # val_gen = get_dataloader(root='./data',
        #                          part='val',
        #                          transform=False,
        #                          base_folder=self.params['base_folder'],
        #                          data_type=self.params['data_type'],
        #                          batch_size=self.params['batch_size'])

        # baseline_generator.train(train_gen=train_gen, val_gen=val_gen)
        
        # # validate baseline generator
        # baseline_generator.validate(val_gen=self.val_gen)
        
        # baseline_images = baseline_generator.test(batch=accumulated_batch)
        # baseline_images = baseline_generator.test(batch=sample_batch)
        
        # baseline_images = torch.zeros(sample_batch['images'].shape).to(self.new_device).float()

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


        return {'avg_loss': avg_loss}

    def weight_init(self, model):
        """
        """
        if self.model_name == "Baseline":
            model.apply(weight_init)
        else:
            model.apply(weight_init)
            if os.path.isfile("model_weights"):
                weights = torch.load("model_weights")
                model.deep.load_state_dict(weights, strict=False)
                print("load weights!")
            else:
                os.system(f"python run.py --config configs/ADNI/baseline.yaml --experiment_name {self.experiment_name} --run_name {self.run_name}")
            os.system(f"python /home/moritz/DeepSurvival/sksurv_trainer.py --download True --seed {self.log_params['manual_seed']} --experiment_name {self.experiment_name} --run_name {self.run_name} --split {self.split}")
            linear_coefficients = np.load(file="/home/moritz/DeepSurvival/linear_weights/weights.npy").astype('float64')
            linear_coefficients = np.expand_dims(linear_coefficients, axis=0)

        try:
            model.linear.weight.data[:, :model.structured_input_dim] = nn.Parameter(torch.FloatTensor(linear_coefficients),
                                                                                    requires_grad=True)
        except:
            pass
  
        for param in model.parameters():
            param.requires_grad = True
        model.train()

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
                                      l2_penalty=self.trial.suggest_uniform("weight_decay", 0.0001, 20.0),
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
            train_data = ADNI(root='/home/moritz/DeepSurvival/data',
                              part='train',
                              transform=self.params['transforms'],
                              download=True,
                              base_folder=self.params['base_folder'],
                              data_type=self.params['data_type'],
                              simulate=self.params['simulate'],
                              split=self.split,
                              seed=self.log_params['manual_seed'],
                              trial=None)
            self.train_gen = DataLoader(dataset=train_data,
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
        
        self.eval_data_train = train_data.eval_data

        return self.train_gen

    def val_dataloader(self):
        """
        """
        if self.params['dataset'] == 'adni':
            val_data = ADNI(root='/home/moritz/DeepSurvival/data',
                             part='val',
                             transform=False,
                             download=True,
                             base_folder=self.params['base_folder'],
                             data_type=self.params['data_type'],
                             simulate=self.params['simulate'],
                             split=self.split,
                             seed=self.log_params['manual_seed'])
            self.val_gen = DataLoader(dataset=val_data,
                                       batch_size=127,
                                       #batch_size=self.params['batch_size'],
                                       collate_fn=utils.cox_collate_fn,
                                       shuffle=False)
        else:
            if self.params['data_type'] == 'coxph':
                val_data = SimCoxPH(root='./data',
                                    part='val',
                                    base_folder=self.params['base_folder'],
                                    data_type=self.params['data_type'],
                                    n_dim=self.params['n_dim'])
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
                                    äcollate_fn=utils.ped_collate_fn,
                                    shuffle=False)
        
        self.eval_data_val = val_data.eval_data
        
        return self.val_gen

    def test_dataloader(self):
        """
        """
        if self.params['dataset'] == 'adni':
            test_data = ADNI(root='/home/moritz/DeepSurvival/data',
                             part='test',
                             transform=False,
                             download=True,
                             base_folder=self.params['base_folder'],
                             data_type=self.params['data_type'],
                             simulate=self.params['simulate'],
                             split=self.split,
                             seed=self.log_params['manual_seed'])
            self.test_gen = DataLoader(dataset=test_data,
                                       batch_size=self.params['batch_size'],
                                       collate_fn=utils.cox_collate_fn,
                                       shuffle=False)
        else:
            if self.params['data_type'] == 'coxph':
                test_data = SimCoxPH(root='./data',
                                    part='test',
                                    base_folder=self.params['base_folder'],
                                    data_type=self.params['data_type'],
                                    n_dim=self.params['n_dim'])
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