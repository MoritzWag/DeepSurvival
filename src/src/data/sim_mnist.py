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


from src.data.sim import SimCoxPH


class SimMNIST(SimCoxPH):
    """
    """
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

    def download(self):
        if not os.path.exists(self.final_path):
            os.mkdir(self.final_path)

        storage_path = os.path.join(self.final_path, self.data_type)
        if not os.path.exists(storage_path):
            os.mkdir(storage_path)
        else:
            return

        np.random.seed(self.seed)

        if self.base_folder == 'mnist':
            X_train, riskgroup_train, X_test, riskgroup_test = self.download_mnist(sample=True)
        else:
            X_train, riskgroup_train, X_test, riskgroup_test = self.download_mnist3d()
        
        X_train, X_val, df_train, df_val, df_test = self.gen_surv_times_groups(X_train,
                                                                               X_test,
                                                                               riskgroup_train,
                                                                               riskgroup_test)

        # save data      
        df_train.to_csv(f"{self.final_path}/{self.data_type}/df_train.csv")
        df_val.to_csv(f"{self.final_path}/{self.data_type}/df_val.csv")
        df_test.to_csv(f"{self.final_path}/{self.data_type}/df_test.csv")

        np.save(file=f"{self.final_path}/{self.data_type}/X_train.npy", arr=X_train)
        np.save(file=f"{self.final_path}/{self.data_type}/X_val.npy", arr=X_val)
        np.save(file=f"{self.final_path}/{self.data_type}/X_test.npy", arr=X_test)

    def download_mnist(self, sample):
        """
        """
        transform = transforms.Compose([transforms.ToTensor()])
        trainset = datasets.MNIST(root=self.final_path, train=True, download=True, transform=transform)
        testset = datasets.MNIST(root=self.final_path, train=False, download=True, transform=transform)

        X_train = trainset.data.unsqueeze(1) / 255.
        Y_train = trainset.targets.numpy() 

        X_test = testset.data.unsqueeze(1) / 255.
        Y_test = testset.targets.numpy()

        if sample:
            X_train, _, Y_train, _ = train_test_split(X_train,
                                                  Y_train, 
                                                  test_size=int(X_train.shape[0] - 10000),
                                                  stratify=Y_train,
                                                  random_state=self.seed)
            X_test, _, Y_test, _ = train_test_split(X_test,
                                                    Y_test,
                                                    test_size=int(X_test.shape[0] - 1000),
                                                    stratify=Y_test,
                                                    random_state=self.seed)

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

    def download_mnist3d(self):
        """3d point clouds generated from MNIST dataset
        """
        # download zip file
        os.system("kaggle datasets download -d daavoo/3d-mnist")
        wd = os.getcwd()
        current_path = os.path.join(wd, '3d-mnist.zip')

        with zipfile.ZipFile(current_path, 'r') as zp:
            zp.extractall(self.final_path)

        train_file = 'train_point_clouds.h5'
        test_file = 'test_point_clouds.h5'
        file_name = 'full_dataset_vectors.h5'
        train_path = os.path.join(self.final_path, train_file)
        test_path = os.path.join(self.final_path, test_file)

        X_train, Y_train = self._extract_images(train_path)
        X_test, Y_test = self._extract_images(test_path)

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

        return X_train, riskgroup_train, X_test, riskgroup_test

    def simulate_coxph_riskscores(self, riskgroups):
        """
        """
        random = np.random.RandomState(self.seed)
        num_obs = riskgroups.shape[0]
        
        x1 = np.repeat(1.0, num_obs)
        x2 = np.repeat(2.0, num_obs)
        # x1 = random.uniform(-2, 3, size=num_obs)
        # x2 = random.uniform(0, 5, size=num_obs)

        df = pd.DataFrame(data={'riskgroup': riskgroups, 'x1': x1, 'x2': x2})

        df['risk_scores'] = 0.25*df['x1'] - 0.3*df['x2'] \
                            + 1.0*(df['riskgroup'] == 0) \
                            + .5*(df['riskgroup'] == 1) \
                            + .25*(df['riskgroup'] == 2) \
                            + 0.0*(df['riskgroup'] == 3) \
                            - .25*(df['riskgroup'] == 4) \
                            - .5 *(df['riskgroup'] == 5) \
                            - 1.0*(df['riskgroup'] == 6)
        
        return df
    
    def gen_surv_times_groups(self, X_train, X_test, riskgroup_train, riskgroup_test):
        """
        """
        riskgroups = np.concatenate((riskgroup_train, riskgroup_test))
        df = self.simulate_coxph_riskscores(riskgroups)


        riskscores = np.array(df['risk_scores'])
        time, event = self.generate_survival_times(num_samples=len(riskscores),
                                                   riskscores=riskscores)

        df['time'] = time
        df['event'] = event

        df_train = df.iloc[:X_train.shape[0], :]
        df_test = df.iloc[X_train.shape[0]:, :]
        
        X_train, X_val, df_train, df_val = train_test_split(X_train,
                                                            df_train,
                                                            test_size=int(self.val_size),
                                                            stratify=df_train['riskgroup'],
                                                            random_state=self.seed)
    
        return X_train, X_val, df_train, df_val, df_test


def get_voxel_grid(points: np.ndarray, resolution: int = 28):
    """
    Take an array with point indices and transform it to an 3D image.

    This code has been inspired by the procedure in:
    https://www.kaggle.com/daavoo/3d-mnist
    The `point` input is an array of shape (N x 3), where N is the number of points
    in the point cloud and 3 the coordinates of the axis [X, Y , Z]. The output is
    an 3D image representation of the provided coordinates. The resolution parameter
    describes the output pixel dimensions of the image.

    :param points: Pixel coordinates in shape (N x 3)
    :param resolution: Number of pixels along each axis in the output.
    :return: 3D image in shape (C x D x H x W)
    """
    assert points.shape[1] == 3, 'Array is not fitting shape: (N x 3).'

    # Make margin
    points_min = np.min(points, axis=0) - 0.001
    points_max = np.max(points, axis=0) + 0.001

    # Adjust so that all sides are of equal length
    diff = max(points_max - points_min) - (points_max - points_min)
    points_min = points_min - diff / 2
    points_max = points_max + diff / 2

    # Create voxel space
    segments = []
    shape = []
    for i in range(3):
        s, step = np.linspace(
            start=points_min[i], stop=points_max[i], num=(resolution + 1), retstep=True
        )
        segments.append(s)
        shape.append(step)

    structure = np.zeros((len(points), 4), dtype=int)
    structure[:, 0] = np.searchsorted(segments[0], points[:, 0]) - 1
    structure[:, 1] = np.searchsorted(segments[1], points[:, 1]) - 1
    structure[:, 2] = np.searchsorted(segments[2], points[:, 2]) - 1

    # i = ((y * n_x) + x) + (z * (n_x * n_y))
    structure[:, 3] = ((structure[:, 1] * resolution) + structure[:, 0]) + (
        structure[:, 2] * (resolution ** 2)
    )

    # Fill grid with voxels
    vector = np.zeros(resolution ** 3)
    count = np.bincount(structure[:, 3])
    vector[: len(count)] = count

    vector = vector.reshape((1, resolution, resolution, resolution))

    # Rotate and return
    return vector.transpose(0, 3, 1, 2)[..., ::-1, :]