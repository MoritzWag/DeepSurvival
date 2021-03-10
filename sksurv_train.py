import pdb
import sksurv
import numpy as np
import pandas as pd 
import os
import argparse
import mlflow
import torch
from torch import nn
from torch.utils import data

from typing import Any, List, Optional

from src.data.adni import ADNI
#from src.data.linear import LinearData
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

from torch.optim import Adam
from torch.utils.data import DataLoader
#from src.models.baseline import Linear
from torch.utils.data.dataloader import default_collate
#from src.data.utils import cox_collate_fn



def parse_args():
    parser = argparse.ArgumentParser(description="generic runner for sksurv CoxPH model")

    parser.add_argument('--download', type=bool, default=False, metavar='N',
                        help='if trained in conjunction with DeepSurvival then download = False')
    parser.add_argument('--seed', type=int, default=1328, metavar='N',
                        help="seed for training")
    parser.add_argument('--experiment_name', type=str, default="sksurv", metavar='N')
    parser.add_argument('--run_name', type=str, default="sksurv", metavar='N')
    parser.add_argument('--split', type=int, default=0, help="which cv split to take")
    
    args = parser.parse_args()
    return args


def main(args):

    # start mlflow run
    mlflow.set_experiment(args.experiment_name)

    features = ['ABETA', 'APOE4', 'AV45',
                'C(PTGENDER)[T.Male]',
                'FDG', 'PTAU', 
                'TAU', 
                'real_age', 
                'age_male',
                'bs_1', 'bs_2', 'bs_3', 'bs_4',
                '.Linear', '.Quadratic', '.Cubic', 
                'C(ABETA_MISSING)[T.1]',
                'C(TAU_MISSING)[T.1]',
                'C(PTAU_MISSING)[T.1]',
                'C(FDG_MISSING)[T.1]',
                'C(AV45_MISSING)[T.1]'
                ]


    train_data = ADNI(root='./data',
                    part='train',
                    transform=False,
                    download=True,
                    base_folder='adni2d',
                    data_type='coxph',
                    simulate=False,
                    split=args.split,
                    seed=args.seed)

    df_train = train_data.df
    X_train = df_train[features].to_numpy()
    y_train = train_data.eval_data['y']

    
    val_data = ADNI(root='./data',
                part='val',
                transform=False,
                download=args.download,
                base_folder='adni2d',
                data_type='coxph',
                simulate=False,
                split=args.split,
                seed=args.seed)

    df_val = val_data.df
    X_val = df_val[features].to_numpy()

    y_val = val_data.eval_data['y']


    test_data = ADNI(root='./data',
                part='test',
                transform=False,
                download=args.download,
                base_folder='adni2d',
                data_type='coxph',
                simulate=False,
                split=args.split,
                seed=args.seed)

    df_test = test_data.df
    X_test = df_test[features].to_numpy()
    y_test = test_data.eval_data['y']


    # fit the model
    estimator = CoxPHSurvivalAnalysis(alpha=0.00001).fit(X_train, y_train)


    # make predictions
    preds = estimator.predict(X_test)
    preds_val = estimator.predict(X_val)
    preds_train = estimator.predict(X_train)
    
    # evaluate 
    cindex_val = concordance_index_censored(df_val['event'].astype(np.bool), df_val['time'], preds_val)
    cindex = concordance_index_censored(df_test['event'].astype(np.bool), df_test['time'], preds)
    cindex_train = concordance_index_censored(df_train['event'].astype(bool), df_train['time'], preds_train)

    print("test", cindex)
    print("val", cindex_val)
    print("train", cindex_train)

    # retrieve coefficients
    coefficients = estimator.coef_

    # store coefficients for initalization
    storage_path = os.path.expanduser("linear_weights")
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)

    np.save(file=f"{storage_path}/weights.npy", arr=coefficients)    

    mlflow.log_metric("cindex_tabular", cindex[0])
    mlflow.log_metric("cindex_tabular_val", cindex_val[0])
    mlflow.log_metric("cindex_tabular_train", cindex_train[0])
    mlflow.log_param('experiment_name', args.experiment_name)
    mlflow.log_param('run_name', args.run_name)
    mlflow.log_param('split', args.split)


if __name__ == "__main__":
    args = parse_args()
    main(args=args)