
import os 
import pdb 
import torch 
import numpy as np 
import pandas as pd 
import shutil 
import zipfile
import h5py
import cv2
import threading
import multiprocessing
import math
import colorsys

import torchvision.transforms as transforms
import torchvision.datasets as datasets 
import matplotlib.pyplot as plt

from torch.utils import data
from sklearn.model_selection import train_test_split
from torchvision.utils import save_image
#from src.data.utils import rectangle_mask, triangle_mask, circle_mask
#from src.data.utils import hsl2rgb, rgb2hsl
#from .utils import *

class SimCoxPH(data.Dataset):
    """
    """
    seed = 1328
    num_obs = 1000
    val_size = 0.2 * num_obs 
    num_groups = 7
    mean_survival_time = 20.0
    prob_censored = 0.45
    mnist3d_path = './data/3dmnist/'

    def __init__(self, 
                 root, 
                 part='train',
                 download=True,
                 base_folder="mnist2",
                 data_type='coxph',
                 n_dim=1):


        self.root = root
        self.part = part 
        self.base_folder = base_folder 
        self.data_type = data_type
        self.n_dim = n_dim
        self.final_path = os.path.join(self.root, self.base_folder)

        if download: 
            self.download()
        self.x, self.df = self.load_dataset(path=self.final_path, part=self.part)
        self.features_list = [col for col in self.df if col.startswith('x')]
        self.eval_data = self.prepare_for_eval()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        images = torch.tensor(self.x[index, :, :, ]).to(self.device)
        tabular_data = torch.tensor(self.df[self.features_list].to_numpy()[index, :]).to(self.device)
        event = torch.tensor(self.df['event'].to_numpy()[index]).to(self.device)
        time = torch.tensor(self.df['time'].to_numpy()[index]).to(self.device)

        return images, tabular_data, event, time
    
    def generate_survival_times(self, num_samples, riskscores):
        """
        """
        random = np.random.RandomState(self.seed)
        baseline_hazard = 1. / self.mean_survival_time
        scale = baseline_hazard * np.exp(riskscores)
        u = random.uniform(low=0, high=1, size=riskscores.shape[0])
        t = - np.log(u) / scale

        # generate time of censoring
        qt = np.quantile(t, 1.0 - self.prob_censored)
        c = random.uniform(low=t.min(), high=qt)

        observed_event = t <= c
        observed_time = np.where(observed_event, t, c)
        return observed_time, observed_event

    def prepare_for_eval(self):
        """
        """
        y = []
        times = self.df['time']
        times_unique = np.unique(self.df['time'])
        times_unique[-1] -= 0.01
        events = self.df['event']
        for time, status in zip(times, events):
            instance = (bool(status), time) 
            y.append(instance)
        
        dt = np.dtype('bool, float')
        y = np.array(y, dtype=dt)

        return {'y': y, 'times_unique': times_unique}

    def load_dataset(self, path, part):
        """
        """
        X = np.load(file=f"{path}/{self.data_type}/X_{part}.npy").astype('float64')
        df = pd.read_csv(f"{path}/{self.data_type}/df_{part}.csv")

        return X, df

    def _extract_images(self, file_path):
        """
        """
        images = []
        labels = []

        with h5py.File(file_path, 'r') as hf:
            for key in tqdm(hf.keys(), leave=False):
                sample = hf[key]
                img = get_voxel_grid(sample['points'][:], resolution=28)
                img = self.minmax_normalize(img)
                label = sample.attrs['label']

                images.append(img)
                labels.append(label)

        return np.stack(images), np.asarray(labels)

    def minmax_normalize(x, lower_bound=None, upper_bound=None):
        """
        Normalize a provided array to 0-1 range.

        Normalize a provided array to 0-1 range.
        :param x: Input array
        :param lower_bound: Optional lower bound. Takes minimum of x if not provided.
        :param upper_bound: Optional upper bound. Takes maximum of x if not provided.
        :return: Normalized array
        """
        if lower_bound is None:
            lower_bound = np.min(x)

        if upper_bound is None:
            upper_bound = np.max(x)

        return (x - lower_bound) / (upper_bound - lower_bound)