import yaml 
import argparse
import numpy as np 
import pdb 

from experiment import DeepSurvExperiment
from pytorch_lightning import Trainer 
from pytorch_lightning.loggers import MLFlowLogger
from src.helpers import *

import torch.backends.cudnn as cudnn 


torch.set_default_dtype(torch.float64)

parser = argparse.ArgumentParser(description='generic runner for Deep Survival Analysis')
parser.add_argument('--config', '-c',
                    dest='filename',
                    metavar='FILE',
                    help='path to config file',
                    default='configs/MNIST/deepcoxph.yaml')
parser.add_argument('--experiment_name', type=str, default='deepsurv')
parser.add_argument('--run_name', type=str, default='deepsurv')
parser.add_argument('--manual_seed', type=int, default=None,
                    help="seed for reproducibility (default: config file)")
parser.add_argument('--max_epochs', type=int, default=None,
                    help="number of epochs (default: config file")


args = parser.parse_args()

with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


# update config 
config = update_config(config=config, args=args)

# compile model
model = parse_model_config(config)

# instantiate logger
mlflow_logger = MLFlowLogger(experiment_name=args.experiment_name)

# for reproducibility
torch.manual_seed(config['logging_params']['manual_seed'])
np.random.seed(config['logging_params']['manual_seed'])
cudnn.deterministic = True
cudnn.benchmark = True 

# build experiment
experiment = DeepSurvExperiment(model,
                                params=config['exp_params'],
                                log_params=config['logging_params'],
                                model_hyperparams=config['model_hyperparams'],
                                baseline_params=config['baseline_params'],
                                run_name=args.run_name,
                                experiment_name=args.experiment_name)

# build trainer 
runner = Trainer(min_epochs=1,
                logger=mlflow_logger,
                check_val_every_n_epoch=1,
                train_percent_check=1.0,
                val_percent_check=1.0,
                num_sanity_val_steps=5,
                early_stop_callback=False,
                fast_dev_run=False,
                **config['trainer_params'])

# run trainer 
print(f"======= Training {config['model_params']['model_name']} ==========")
runner.fit(experiment)
runner.test()

