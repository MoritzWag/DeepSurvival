import yaml 
import argparse
import numpy as np 
import pdb 

from experiment import DeepSurvExperiment
from pytorch_lightning import Trainer 
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from src.helpers import *

from ray import tune 
from ray.tune import CLIReporter
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.integration.pytorch_lightning import TuneReportCallback

import torch.backends.cudnn as cudnn 


torch.set_default_dtype(torch.float64)

parser = argparse.ArgumentParser(description='generic runner for Deep Survival Analysis')
parser.add_argument('--config', '-c',
                    dest='filename',
                    metavar='FILE',
                    help='path to config file',
                    default='configs/ADNI/deepcoxph.yaml')
parser.add_argument('--experiment_name', type=str, default='deepsurv')
parser.add_argument('--run_name', type=str, default='deepsurv')
parser.add_argument('--manual_seed', type=int, default=None,
                    help="seed for reproducibility (default: config file)")
parser.add_argument('--max_epochs', type=int, default=None,
                    help="number of epochs (default: config file")


parser.add_argument('--learning_rate', type=float, default=None,
                    help='learning rate for survival model')

args = parser.parse_args()


with open(args.filename, 'r') as file:
    try:
        configuration = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


def objective(config, configuration, args):

    configuration = update_config(configuration, args=args)

    # configuration['model_params']['deep_params']['out_dim'] = config['out_dim']

    model = parse_model_config(configuration)
    
    configuration['exp_params']['learning_rate'] = config['learning_rate']
    configuration['exp_params']['scheduler_gamma'] = config['scheduler_gamma']
    configuration['exp_params']['weight_decay'] = config['weight_decay']

    # build experiment 
    experiment = DeepSurvExperiment(model,
                                params=configuration['exp_params'],
                                log_params=configuration['logging_params'],
                                model_hyperparams=configuration['model_hyperparams'],
                                baseline_params=configuration['baseline_params'],
                                run_name=args.run_name,
                                experiment_name=args.experiment_name)
    
    # build trainer 
    runner = Trainer(min_epochs=1,
                    checkpoint_callback=True,
                    check_val_every_n_epoch=1,
                    num_sanity_val_steps=5,
                    callbacks=[TuneReportCallback(
                        metrics={
                            "cindex": "cindex_val"
                        },
                        on="validation_end"
                    )],
                    fast_dev_run=False,
                    **configuration['trainer_params'])
    #pdb.set_trace()
    runner.fit(experiment)
    runner.test()


config = {
    'learning_rate': tune.loguniform(1e-5, 2e-2),
    'scheduler_gamma': tune.uniform(0.99, 0.999),
    'weight_decay': tune.loguniform(0.000001, 0.001)
    # 'out_dim': tune.choice([10, 20, 30, 40])
}

bayesopt = BayesOptSearch(metric='cindex', mode="max")

report = CLIReporter(
    parameter_columns=['learning_rate', 'scheduler_gamma', 'weight_decay'],
    metric_columns=['cindex']
)

analysis = tune.run(
    tune.with_parameters(
        objective,
        configuration=configuration,
        args=args
    ),
    metric='cindex',
    mode='max',
    config=config,
    search_alg=bayesopt,
    progress_reporter=report,
    name="tune_bayesopt")


print("Best hyperparameters found were: ", analysis.best_config)

df = analysis.results_df
df.to_csv(f"bo_{args.run_name}.csv")