import yaml 
import argparse
import numpy as np 
import pdb 
import os
import optuna
import pytorch_lightning as pl

from experiment import DeepSurvExperiment
from pytorch_lightning import Trainer 
from pytorch_lightning.loggers import MLFlowLogger
from src.helpers import *

from pytorch_lightning import Callback 
from optuna.integration import PyTorchLightningPruningCallback
from datetime import datetime

import plotly.io as pio 
pio.orca.config.use_xvfb = True

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


#tuner params
parser.add_argument('--n_trials', type=int, default=None, metavar='N',
                    help='specifies the number of trials for tuning')
parser.add_argument('--timeout', type=int, default=54000, metavar='N',
                    help="specifies the total seconds used for tuning")
parser.add_argument('--min_resource', type=int, default=5, metavar='N',
                    help='minimum resource use for each configuration during tuning')
parser.add_argument('--reduction_factor', type=int, default=3, metavar='N',
                    help='factor by which number of configurations are reduced')


args = parser.parse_args()

with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


# update config 
config = update_config(config=config, args=args)


class MetricsCallback(Callback):
    """
    """
    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)

DIR = os.getcwd()
MODEL_DIR = os.path.join(DIR, "result")

start = datetime.now()

def objective(trial):
    """
    """
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, "trial_{}".format(trial.number), "{epoch}"), monitor="cindex"
    )

    metrics_callback = MetricsCallback()
    
    model = parse_model_config(config)
    mlflow_logger = MLFlowLogger(experiment_name=args.experiment_name)

    # build experiment
    experiment = DeepSurvExperiment(model,
                                    params=config['exp_params'],
                                    log_params=config['logging_params'],
                                    model_hyperparams=config['model_hyperparams'],
                                    baseline_params=config['baseline_params'],
                                    run_name=args.run_name,
                                    experiment_name=args.experiment_name,
                                    trial=trial)

    # build trainer 
    runner = Trainer(min_epochs=1,
                    logger=mlflow_logger,
                    check_val_every_n_epoch=1,
                    train_percent_check=1.0,
                    val_percent_check=1.0,
                    num_sanity_val_steps=5,
                    callbacks=[metrics_callback],
                    early_stop_callback=PyTorchLightningPruningCallback(trial, monitor='cindex'),
                    fast_dev_run=False,
                    **config['trainer_params'])
    
    runner.fit(experiment)

    return metrics_callback.metrics[-1]['cindex'].item()

n_train_iter = config['trainer_params']['max_epochs']
study = optuna.create_study(
    direction='maximize',
    pruner=optuna.pruners.HyperbandPruner(
        min_resource=args.min_resource,
        max_resource=n_train_iter,
        reduction_factor=args.reduction_factor
    )
)


if args.n_trials is not None:
    print(f"tune with n_trials: {args.n_trials}")
    study.optimize(objective, n_trials=args.n_trials, catch=(TypeError, RuntimeError, ))
if args.timeout is not None:
    study.optimize(objective, timeout=args.timeout, catch=(TypeError, RuntimeError, ))

end = datetime.now()
diff = end - start 
print(f"total tuning time was: {diff}")

# retrieve/print best trial
best_trial = study.best_trial 
print(best_trial)

# retrieve/print best params
best_params = study.best_params
print(best_params)

# save results in dataframe
df = study.trials_dataframe()
df.to_csv(f"hb_{args.run_name}.csv")


fig_intermediate_values = optuna.visualization.plot_intermediate_values(study)
fig_intermediate_values.write_image(f'hb_tune_interm_{args.run_name}.png')

fig_opt_history = optuna.visualization.plot_optimization_history(study)
fig_opt_history.write_image(f'hb_tune_opt_hist_{args.run_name}.png')


fig_hyp_importance = optuna.visualization.plot_param_importances(study)
fig_hyp_importance.write_image(f"hb_tune_hyp_imp_{args.run_name}.png")