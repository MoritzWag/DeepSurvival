import json
import os
import pdb
import torch
import numpy as np
import pandas as pd
import shutil
import zipfile
import h5py

import torchvision.transforms as transforms
import torchvision.datasets as datasets 
import matplotlib.pyplot as plt

from tqdm import tqdm
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
    mnist3d_path = './data/3dmnist/'

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

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def download_mnist(self, sample=False):
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
                                                  test_size=int(X_train.shape[0] - self.num_obs),
                                                  stratify=Y_train,
                                                  random_state=self.seed)
            X_test, _, Y_test, _ = train_test_split(X_test,
                                                    Y_test,
                                                    test_size=int(X_test.shape[0] - (0.1*self.num_obs)),
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
        pdb.set_trace()
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

    def simulate_images(self,
                        img_size,
                        num_obs,
                        n_groups,
                        n_dim,
                        pos_random,
                        num_shapes):

        random = np.random.RandomState(self.seed)
        # assign each image a riskgroup
        groups = random.randint(n_groups, size=num_obs)
        # assign each image a shape
        shapes = random.randint(num_shapes, size=num_obs)

        # assign grayscale to each group: [0, 1]
        # this will solely determine the riskscore
        if n_dim == 1:
            grayscales = random.uniform(0, 1, n_groups)
            grayscales = np.sort(grayscales)
            color_assignment = []
            for i in groups:
                color_assignment.append(grayscales[i])
        else:
            colors_r = random.randint(0, 255, n_groups)
            colors_g = random.randint(0, 255, n_groups)
            colors_b = random.randint(0, 255, n_groups)
            color_assignment = []
            for i in groups:
                color_assignment.append((int(colors_r[i]), int(colors_g[i]), int(colors_b[i])))

        images_final = []
        for i in range(num_obs):
            #group = groups[i]
            shape = shapes[i]
            color = color_assignment[i]
            if shape == 0:
                img = rectangle_mask(img_size=img_size,
                                     length=5,
                                     color=color,
                                     n_dim=n_dim,
                                     pos_random=pos_random)
            elif shape == 1:
                img = triangle_mask(img_size=img_size,
                                    length=5,
                                    color=color,
                                    n_dim=n_dim,
                                    pos_random=pos_random)
            else:
                img = circle_mask(img_size=img_size,
                                  center=(12, 12),
                                  radius=5,
                                  color=color,
                                  n_dim=n_dim,
                                  pos_random=pos_random)
            
            #images_final = images_final.append(img)
            img = minmax_normalize(img)
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


    def _extract_images(self, file_path):
        """
        """
        images = []
        labels = []

        with h5py.File(file_path, 'r') as hf:
            for key in tqdm(hf.keys(), leave=False):
                sample = hf[key]
                img = get_voxel_grid(sample['points'][:], resolution=28)
                img = minmax_normalize(img)
                label = sample.attrs['label']

                images.append(img)
                labels.append(label)

        return np.stack(images), np.asarray(labels)



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
