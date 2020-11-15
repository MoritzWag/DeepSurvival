import numpy as np 
import pandas as pd 
import argparse
import torch 
import os 
import pdb 
import shutil

import torchvision.transforms as transforms
import torchvision.datasets as datasets 
import matplotlib.pyplot as plt

from sksurv.nonparametric import kaplan_meier_estimator
from sklearn.model_selection import train_test_split
from src.data.simulations import simulate_mm_coxph_riskscores, generate_survival_time, simulate_um_coxph_riskscores



def parse_args():
    parser = argparse.ArgumentParser(description='mnist download')
    parser.add_argument('--path', type=str, default='../data',
                        help='path to store data')
    parser.add_argument('--val_size', type=int, default=3000,
                        help='size of validation data')
    parser.add_argument('--multimodal', type=bool, default=False,
                        help='specify whether structured part shall be also simulated')
    args = parser.parse_args()
    return args 


def get_data(args):
    
    data_path = os.path.expanduser(args.path)
    storage_path = f'{data_path}/mnist/'
    if not os.path.exists(storage_path):
        os.mkdir(storage_path)

    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    testset = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)

    X_train = trainset.data.unsqueeze(1) / 255.
    Y_train = trainset.targets.numpy()

    X_test = testset.data.unsqueeze(1) / 255.
    Y_test = testset.targets.numpy()

    Y = np.concatenate((Y_train, Y_test))
    
    if args.multimodal:
        df = simulate_mm_coxph_riskscores(Y, n_groups=4, seed=89)
        risk_scores = df['risk_scores']
    else:
        risk_score_assignment, risk_scores = simulate_um_coxph_riskscores(Y, n_groups=4, seed=89)
        df = pd.DataFrame(data={'labels': Y, 'risk_scores': risk_scores})

    time, event = generate_survival_time(num_samples=Y.shape,
                                         mean_survival_time=365.0,
                                         prob_censored=0.45,
                                         risk_score=risk_scores,
                                         seed=89)
    df['time'] = time 
    df['event'] = event

    df_train = df.iloc[:Y_train.shape[0], :]
    df_test = df.iloc[Y_train.shape[0]: , :]

    # plot data
    if args.multimodal == False:
        styles = ('-', '--', '-.', ':')

        plt.figure(figsize=(6, 4.5))
        for row in risk_score_assignment.itertuples():
            mask = Y_train == row.Index
            coord_x, coord_y = kaplan_meier_estimator(event_train[mask], time_train[mask])
            ls = styles[row.risk_group]
            plt.step(coord_x, coord_y, where="post", label=f"Class {row.Index}", linestyle=ls)
        plt.ylim(0, 1)
        plt.ylabel("Probability of survival $P(T > t)$")
        plt.xlabel("Time $t$")
        plt.grid()
        plt.legend()
        plt.show()

    X_train, X_val, df_train, df_val = train_test_split(X_train, 
                                                        df_train,
                                                        test_size=args.val_size,
                                                        stratify=df_train['labels'], 
                                                        random_state=1337)

    # save data 
    if not os.path.exists(f"{args.path}/coxph/"):
        os.mkdir(f"{args.path}/coxph/")
    df_train.to_csv(f"{args.path}/coxph/df_train.csv")
    df_val.to_csv(f"{args.path}/coxph/df_val.csv")
    df_test.to_csv(f"{args.path}/coxph/df_test.csv")

    np.save(file=f"{args.path}/coxph/X_train.npy", arr=X_train)
    np.save(file=f"{args.path}/coxph/X_val.npy", arr=X_val)
    np.save(file=f"{args.path}/coxph/X_test.npy", arr=X_test)

    shutil.rmtree(f"{args.path}/mnist")



if __name__ == "__main__":
    args = parse_args()
    get_data(args=args)