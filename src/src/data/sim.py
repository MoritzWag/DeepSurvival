import json
import os
import pdb
import torch
import numpy as np
import pandas as pd
import shutil

import torchvision.transforms as transforms
import torchvision.datasets as datasets 
import matplotlib.pyplot as plt

from torch.utils import data
from sklearn.model_selection import train_test_split
from src.data.utils import rectangle_mask, triangle_mask, circle_mask


class SimulationData2d(data.Dataset):
    """
    """

    seed = 1328
    num_obs = 10000
    val_size = 0.2 * num_obs
    num_groups = 4
    mean_survival_time = 365.0
    prob_censored = 0.45

    def __init__(self, 
                 root,
                 part='train',
                 download=True,
                 base_folder='mnist2',
                 data_type='coxph'):
        self.root = root
        self.part = part
        self.base_folder = base_folder
        self.data_type = data_type
        self.final_path = os.path.join(self.root, self.base_folder)

    def size(self):
        pass

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def download_mnist(self):
        transform = transforms.Compose([transforms.ToTensor()])
        trainset = datasets.MNIST(root=self.final_path, train=True, download=True, transform=transform)
        testset = datasets.MNIST(root=self.final_path, train=False, download=True, transform=transform)

        X_train = trainset.data.unsqueeze(1) / 255.
        Y_train = trainset.targets.numpy() 

        X_test = testset.data.unsqueeze(1) / 255.
        Y_test = testset.targets.numpy()

        Y = np.concatenate((Y_train, Y_test))

        df = pd.DataFrame(data={'labels': Y})

        random = np.random.RandomState(self.seed)
        classes = np.unique(Y)
        group_assignment = {}
        group_members = {}
        groups = random.randint(self.num_groups, size=classes.shape)
        for label, group in zip(classes, groups):
            group_assignment[label] = group
            group_members.setdefault(group, []).append(label)

        df['riskgroup'] = df['labels'].map(group_assignment)

        riskgroup_train = df.iloc[:Y_train.shape[0], :]['riskgroup']
        riskgroup_test = df.iloc[Y_train.shape[0]:, :]['riskgroup']

        shutil.rmtree(f"{self.final_path}/MNIST")

        return X_train, riskgroup_train, X_test, riskgroup_test


    def simulate_images(self,
                        img_size,
                        num_obs,
                        n_groups,
                        pos_random,
                        num_shapes):

        random = np.random.RandomState(self.seed)
        # assign each image a riskgroup
        groups = random.randint(n_groups, size=num_obs)
        # assign each image a shape
        shapes = random.randint(num_shapes, size=num_obs)

        # assign grayscale to each group: [0, 1]
        # this will solely determine the riskscore
        grayscales = random.uniform(0, 1, n_groups)
        grayscales_assignment = []
        for i in groups:
            grayscales_assignment.append(grayscales[i])
        
        
        images_final = []
        for i in range(num_obs):
            group = groups[i]
            shape = shapes[i]
            grayscale = grayscales_assignment[i]

            if group == 0:
                img = rectangle_mask(img_size=img_size,
                                     length=5,
                                     gray_scale=grayscale,
                                     pos_random=pos_random)
            elif group == 1:
                img = triangle_mask(img_size=img_size,
                                    length=5,
                                    gray_scale=grayscale,
                                    pos_random=pos_random)
            else:
                img = circle_mask(img_size=img_size,
                                  center=(12, 12),
                                  radius=5,
                                  gray_scale=grayscale,
                                  pos_random=pos_random)
            
            #images_final = images_final.append(img)
            images_final.append(img)

        images_final = np.stack(images_final)

        # split data into train and test 
        X_train, X_test, riskgroup_train, riskgroup_test = train_test_split(images_final,
                                                                            groups,
                                                                            test_size=int(0.1*num_obs),
                                                                            stratify=groups,
                                                                            random_state=self.seed)
        

        return X_train, riskgroup_train, X_test, riskgroup_test


    def load_dataset(self, path, part):
        """
        """
        X = np.load(file=f"{path}/{self.data_type}/X_{part}.npy").astype('float64')
        df = pd.read_csv(f"{path}/{self.data_type}/df_{part}.csv")

        return X, df


