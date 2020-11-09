import numpy as np 
import pandas as pd 
import argparse
import torch 
import os 
import pdb 

import torchvision.transforms as transforms
import torchvision.datasets as datasets 

from src.data.simulations import make_risk_score_for_groups, generate_survival_time



def parse_args():
    parser = argparse.ArgumentParser(description='mnist download')
    parser.add_argument('--path', type=str, default='.',
                        help='path to store data')
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
    Y_test = trainset.targets.numpy()

    risk_score_assignment, risk_scores = make_risk_score_for_groups(Y_train, n_groups=4, seed=89)

    time, event = generate_survival_time(num_samples=Y_train.shape,
                                         mean_survival_time=365.0,
                                         prob_censored=0.45,
                                         risk_score=risk_scores,
                                         seed=89)
    
    pdb.set_trace()

    # store data !!

if __name__ == "__main__":
    args = parse_args()
    get_data(args=args)