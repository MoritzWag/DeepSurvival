import numpy as np 
import pandas as pd 
import argparse
import torch 
import os 
import pdb 

import torchvision.transforms as transforms
import torchvision.datasets as datasets 
import matplotlib.pyplot as plt

from sksurv.nonparametric import kaplan_meier_estimator
from sklearn.model_selection import train_test_split
from src.data.simulations import simulate_mm_coxph_riskscores, generate_survival_time, simulate_um_coxph_riskscores



def parse_args():
    parser = argparse.ArgumentParser(description='mnist download')
    parser.add_argument('--path', type=str, default='.',
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


if __name__ == "__main__":
    args = parse_args()
    get_data(args=args)