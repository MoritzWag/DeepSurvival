import pandas as pd
import numpy as np 
import torch 
import src 
import pdb 
import os 
import torch.nn as nn
import yaml 
import argparse 

from src.data import utils 
from src.data.utils import get_dataloader, generated_colored_bs_img
from src.data.adni import ADNI 
from src.data.sim_mnist import SimMNIST
from src.data.sim_images import SimImages
from src.helpers import *

from src.dsap.dsap import DSAP
from src.dsap.coalition_policies.playergenerators import *
from src.integrated_gradients import IntegratedGradients
from src.baselines.baseline_generator import BaselineGenerator
from src.shapley_sampling import ShapleySampling

from src.architectures.SIM.discriminators import DiscriminatorSIM
from src.architectures.ADNI.discriminators import DiscriminatorADNI
from src.architectures.SIM.map_generator import GeneratorSIM
from src.architectures.ADNI.map_generator import GeneratorADNI


parser = argparse.ArgumentParser(description="generic runner for Explainer")
parser.add_argument('--config', '-c',
                    dest='filename',
                    metavar='FILE',
                    help='path to config file',
                    default='configs/SIM/deepcoxph.yaml')
parser.add_argument('--run_name', type=str, default='deepsurv')
parser.add_argument('--step', type=int, default=None,
                    help='step where baseline generator results were optimal')
parser.add_argument('--DSAP', type=bool, default=False,
                    help='if True, calculate and visualize DSAP values')

#baseline params
parser.add_argument('--alpha', type=float, default=None,
                    help="penalty for domain loss")
parser.add_argument('--lambda_cls', type=float, default=None,
                    help='loss weight for domain loss')
parser.add_argument('--lambda_gp', type=float, default=None,
                    help="loss weight for  gradient penalty")
parser.add_argument('--lambda_rec', type=float, default=None,
                    help='loss weight for reconstruction')
parser.add_argument('--n_critic', type=int, default=None,
                    help='number of generator steps per discriminator')
parser.add_argument('--storage_path', type=str, default=None,
                    help="path to stored outputs of baseline generator")
parser.add_argument('--PATH', type=str, default=None,
                    help="path to stored survival model")

args = parser.parse_args()

with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# update config 
config = update_config(config=config, args=args)
params = config['exp_params']
logging_params = config['baseline_params']['logging_params']

# load survival model 
model = parse_model_config(config)
PATH = f"survival_model/sm_{args.run_name}"
model.load_state_dict(torch.load(PATH))
model = model.to(device).float()


def main(params, logging_params, args, survival_model):
    
    # load data!
    test_gen, _, data = get_dataloader(root='./data',
                                       part='test',
                                       transform=False,
                                       base_folder=params['base_folder'],
                                       data_type=params['data_type'],
                                       batch_size=-1,
                                       split=0,
                                       cox_collate=False,
                                       return_data=True)
   
    sample_batches = survival_model._sample_batch(data=test_gen, num_obs=4, num_batches=3)

    # storage path of baseline generated images
    storage_path = os.path.expanduser(logging_params['storage_path'])
    storage_path = f"{storage_path}/{logging_params['run_name']}/test_results"
    bg_dict = np.load(f"{storage_path}/data_dict_{args.step}.npy", allow_pickle=True)

    
    for sample_key in sample_batches:
        sample_batch = sample_batches[sample_key]
        images, tabular_data, indeces = sample_batch['images'].to(device), sample_batch['tabular_data'].to(device), sample_batch['indeces']

        bg_imgs = torch.tensor(bg_dict.item()['bs_img'][indeces]).to(device)

        # generate colored baseline images
        colord_bs_images = generated_colored_bs_img(data, sample_batch, indeces).float()

        # generate zero baseline images
        zero_bs_images = torch.zeros(sample_batch['images'].shape).to(device).float()

        baseline_dict = {'zero_bs': zero_bs_images, 'colored_bs': colord_bs_images,
                        'bg_bs': bg_imgs}

        # Initialize Integrated Gradients
        IG = IntegratedGradients()

        if args.DSAP: 
            # Initialize DSAP
            lpdn_model = survival_model._build_lpdn_model()
            dsap = DSAP(player_generator=WideDeepPlayerIterator(ground_input=(images, tabular_data), windows=False),
                        input_shape=images[0].shape, 
                        lpdn_model=lpdn_model)

        # Initialize Shapley sampling
        sv_sampler = ShapleySampling()

        # Loop through different baselines
        for key in baseline_dict:

            # Calculate Integrated Gradients
            print(f"Calculate Integrated Gradients for: {key}")
            integrated_gradients = IG.integrated_gradients(model=survival_model,
                                                        images=images, 
                                                        tabular_data=tabular_data,
                                                        length=100,
                                                        baseline=baseline_dict[key],
                                                        storage_path="euclidean_morph",
                                                        run_name=logging_params['run_name'])

            # Visualize IG attributions 
            survival_model.visualize_attributions(images=images,
                                                attributions=integrated_gradients,
                                                rgb_trained=False,
                                                method='integrated_gradeints',
                                                storage_path=f"attributions/{key}/{sample_key}",
                                                run_name=logging_params['run_name'])
        
        for key in baseline_dict:
            # Calculate sampled shapley attributions
            shapley_attributions = sv_sampler.sv_sampling(model=survival_model,
                                                        images=images, 
                                                        tabular_data=tabular_data,
                                                        baseline=baseline_dict[key],
                                                        n_steps=10)

            # Visualize sample shapley attributions
            survival_model.visualize_attributions(images=images,
                                                attributions=shapley_attributions,
                                                rgb_trained=False,
                                                method="shapley",
                                                storage_path=f"attributions/{key}/{sample_key}",
                                                run_name=logging_params['run_name'])
        if args.DSAP:
            for key in baseline_dict:
                # Calculate approximate Shapley Values
                print('Calculate Deep Approximate Shapley Values')
                lpdn_model = survival_model._build_lpdn_model()
                dsap = DSAP(player_generator=WideDeepPlayerIterator(ground_input=(images, tabular_data), windows=False),
                            input_shape=images[0].shape,
                            lpdn_model=lpdn_model)
                shapley_attributions = dsap.run(images=images, 
                                                tabular_data=tabular_data,
                                                baselines=zero_baseline_images, 
                                                steps=50)

                # Visualize DSAP attributions
                survival_model.visualize_attributions(images=images, 
                                                attributions=shapley_attributions,
                                                rgb_trained=False,
                                                method='dsap',
                                                storage_path=f'attributions/{key}/{sample_key}',
                                                run_name=args.run_name)

if __name__ == "__main__":
    main(params=params,
         logging_params=logging_params,
         args=args, 
         survival_model=model)