
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
    # mean_survival_time = 365.0
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

        if self.base_folder == "sim_cont":
            self.continuous = True
        else:
            self.continuous = False

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
    

    def download(self):
        if not os.path.exists(self.final_path):
            os.mkdir(self.final_path)

        storage_path = os.path.join(self.final_path, self.data_type)
        if not os.path.exists(storage_path):
            os.mkdir(storage_path)
        else:
            return

        np.random.seed(self.seed)

        if not self.continuous:
            if self.base_folder == 'mnist':
                X_train, riskgroup_train, X_test, riskgroup_test = self.download_mnist(sample=True)
            elif self.base_folder == 'mnist3d':
                X_train, riskgroup_train, X_test, riskgroup_test = self.download_mnist3d()
                self.val_size = 500
            else:
                X_train, riskgroup_train, X_test, riskgroup_test = self.simulate_images(img_size=28,
                                                                                        num_obs=self.num_obs,
                                                                                        n_groups=self.num_groups,
                                                                                        n_dim=self.n_dim,
                                                                                        pos_random=True,
                                                                                        num_shapes=3)
            
            X_train, X_val, df_train, df_val, df_test = self.gen_surv_times_groups(X_train,
                                                                                   X_test, 
                                                                                   riskgroup_train,
                                                                                   riskgroup_test)

        else:
            X_train, X_test, h_channel_train, h_channel_test = self.simulate_continuous_images(img_size=28,
                                                                                               num_obs=self.num_obs,
                                                                                               n_dim=self.n_dim,
                                                                                               pos_random=True,
                                                                                               num_shapes=1,
                                                                                               figure_colored=self.figure_colored)

            X_train, X_val, df_train, df_val, df_test = self.gen_surv_times_cont(X_train,
                                                                                 X_test,
                                                                                 h_channel_train,
                                                                                 h_channel_test)

        # save data      
        df_train.to_csv(f"{self.final_path}/{self.data_type}/df_train.csv")
        df_val.to_csv(f"{self.final_path}/{self.data_type}/df_val.csv")
        df_test.to_csv(f"{self.final_path}/{self.data_type}/df_test.csv")

        np.save(file=f"{self.final_path}/{self.data_type}/X_train.npy", arr=X_train)
        np.save(file=f"{self.final_path}/{self.data_type}/X_val.npy", arr=X_val)
        np.save(file=f"{self.final_path}/{self.data_type}/X_test.npy", arr=X_test)

    def gen_surv_times_cont(self, X_train, X_test, h_channel_train, h_channel_test):
        """
        """
        h_channels = np.concatenate((h_channel_train, h_channel_test))
        df = self.simulate_coxph_riskscores_cont(h_channels)

        riskscores = np.array(df['risk_scores'])

        time, event = self.generate_survival_times(num_samples=len(riskscores),
                                                   riskscores=riskscores)
        
        df['time'] = time 
        df['event'] = event 

        df_train = df.iloc[:X_train.shape[0], :]
        df_test = df.iloc[X_train.shape[0]: , :]

        X_train, X_val, df_train, df_val = train_test_split(X_train,
                                                            df_train, 
                                                            test_size=int(self.val_size),
                                                            random_state=self.seed)
        
        return X_train, X_val, df_train, df_val, df_test
        
    
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

    def simulate_coxph_riskscores_cont(self, h_channels):
        """
        """
        random = np.random.RandomState(self.seed)
        num_obs = h_channels.shape[0]
        x1 = np.repeat(0.00001, num_obs)
        x2 = np.repeat(0.00001, num_obs)

        h_channels_norm = (h_channels - 127.5) / 127.5

        df = pd.DataFrame(data={'h_channels': h_channels, 'h_channels_norm': h_channels_norm, 'x1': x1, 'x2': x2})

        df['risk_scores'] = 0.25*df['x1'] - 0.3*df['x2'] \
                            + 1.0*df['h_channels_norm']

        return df


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

    def simulate_continuous_images(self, 
                                   img_size,
                                   num_obs,
                                   n_dim,
                                   pos_random,
                                   num_shapes,
                                   figure_colored):
        
        random = np.random.RandomState(self.seed)
        shapes = random.randint(num_shapes, size=num_obs)
        color_h = random.randint(1, 255, size=num_obs)
        
        color_hsl = []
        color_rgb = []
        color_hls = []
        for idx in range(color_h.shape[0]):
            h_channel = color_h[idx]
            color_hsl.append((h_channel, 100, 50))
            color_hls.append((h_channel, 50, 100))
            color_rgb_norm = colorsys.hls_to_rgb(h_channel/360, 50/100, 100/100)
            color_rgb.append((int(round(color_rgb_norm[0]*255)), int(round(color_rgb_norm[1]*255)), int(round(color_rgb_norm[2]*255))))
        
        images_final = []
        for i in range(num_obs):
            shape = shapes[i]
            color = color_rgb[i]
            img = self.get_geometric_images(img_size=img_size,
                                            shape=shape,
                                            color=color,
                                            figure_colored=figure_colored, 
                                            n_dim=n_dim,
                                            pos_random=pos_random)
            
            img = img.reshape(28, 28, 3)
            img = rgb2hsl(img)
            img = img.reshape(3, 28, 28)
            images_final.append(img)
        
        images_final = np.stack(images_final)

        X_train, X_test, h_channel_train, h_channel_test = train_test_split(images_final,
                                                                            color_h,
                                                                            test_size=int(0.1*num_obs),
                                                                            # stratify=color_h,
                                                                            random_state=self.seed)

        return X_train, X_test, h_channel_train, h_channel_test

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

        colors_rgb = [(255, 4, 0), (255, 132, 0), (255, 247, 0), (144, 255, 0), (34, 255, 0), (0, 183, 255), (8, 0, 255)]
        colors_hsl = [(1, 100, 50), (31, 100, 50), (58, 100, 50), (86, 100, 50), (112, 100, 50), (197, 100, 50), (242, 100, 50)]
        color_assignment = []
        color_assignment_hsl = []
        for i in groups:
            color_assignment.append(colors_rgb[i])
        for i in groups:
            color_assignment_hsl.append(colors_hsl[i])

        images_final = []
        for i in range(num_obs):
            shape = shapes[i]
            color = color_assignment[i]
            img = self.get_geometric_images(img_size=img_size,
                                            shape=shape,
                                            color=color,
                                            n_dim=n_dim,
                                            pos_random=pos_random)

            img = img.reshape(28, 28, 3)
            img = rgb2hsl(img)
            img = img.reshape(3, 28, 28)
            images_final.append(img)
            
        images_final = np.stack(images_final)

        # split data into train and test 
        X_train, X_test, riskgroup_train, riskgroup_test = train_test_split(images_final,
                                                                            groups,
                                                                            test_size=int(0.1*num_obs),
                                                                            stratify=groups,
                                                                            random_state=self.seed)
        
        return X_train, riskgroup_train, X_test, riskgroup_test

    def simulate_circles(self, 
                         img_size,
                         num_obs,
                         n_dim,
                         pos_random):
        """
        """
        rand = np.random.RandomState(self.seed)
        colors_rgb = [(255, 4, 0), (255, 132, 0), (255, 247, 0), (144, 255, 0), (34, 255, 0), (0, 183, 255), (8, 0, 255)]
        lengths = rand.randint(3, 20, size=num_obs)
        n_groups = len(colors_rgb)
        groups = rand.randint(n_groups, size=num_obs)
        color_assignment = []
        for i in groups:
            color_assignment.append(colors_rgb[i])

        for i in range(num_obs):
            color = color_assignment[i]
            img = np.zeros((img_size, img_size, 3), dtype='float32')
            img[:, :] = color
            ang = random.randrange(0, 360, 10)
            length = lenghts[i]
            (W, H) = (length, length)
            P0 = 


    def get_geometric_images(self, 
                            img_size, 
                            shape,
                            color,
                            n_dim,
                            pos_random):
        """
        """
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
                                #center=(12, 12),
                                center=14,
                                radius=5,
                                color=color,
                                n_dim=n_dim,
                                pos_random=pos_random)

        return img

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


def rectangle_mask(img_size, 
                   length,
                   color,
                   n_dim,
                   pos_random=True,
                   seed=1328):

    random = np.random.RandomState(seed)
    if n_dim == 1:
        img = np.zeros((n_dim, img_size, img_size), dtype='float32')
    else:
        img = np.zeros((img_size, img_size, n_dim), dtype='float32')

    if pos_random is False:
        upper_corner = (14, 14)
        lower_corner = (10, 10)
    else:
        height = random.randint(img_size, size=1)[0]
        width = random.randint(img_size-length, size=1)[0]
        upper_corner = (height, width)
    if n_dim == 1:
        img[:, upper_corner[0]- length: upper_corner[0], upper_corner[1]:upper_corner[1]+length] = color
    else:
        left_corner = [14, 14]
        points = np.array([[left_corner[0], left_corner[1]],
                        [left_corner[0], left_corner[1] + length],
                        [left_corner[0] + length, left_corner[1] + length],
                        [left_corner[0] + length, left_corner[1]]], np.int32)

        
        image = cv2.fillPoly(img, [points], color=color)
        img = image.reshape(n_dim, img_size, img_size)

    return img 

def triangle_mask(img_size,
                  length,
                  color,
                  n_dim,
                  pos_random=True,
                  seed=1328):

    random = np.random.RandomState(seed)
    img = np.zeros((img_size, img_size, n_dim), dtype='float32')
    if pos_random is False:
        left_corner = [14, 14]
    else:
        left_corner = [random.randint(img_size-5, size=1)[0], random.randint(img_size-5, size=1)[0]]
    
    points = np.array([[left_corner[0], left_corner[1]],
                       [left_corner[0] + 5, left_corner[1] + 5],
                       [left_corner[0], left_corner[1] + 10]], np.int32)
    
    image = cv2.fillPoly(img, [points], color=color)

    image = image.reshape(n_dim, img_size, img_size)

    print('worked triangle')

    return image

def circle_mask(img_size,
                center,
                radius,
                color,
                n_dim,
                pos_random=True,
                seed=1328):

    random = np.random.RandomState(seed)
    img = np.zeros((img_size, img_size, n_dim), dtype='float32')
    if pos_random is False:
        center = center
    else:
        center = (random.randint(img_size - radius, size=1)[0], random.randint(img_size - radius, size=1)[0])
    
    img = cv2.circle(img, center, radius, color, thickness=-1)

    image = img.reshape(n_dim, img_size, img_size)

    print("worked circle")

    return image



def rgb2hsl(rgb):

    def core(_rgb, _hsl):

        irgb = _rgb.astype(np.uint16)
        ir, ig, ib = irgb[:, :, 0], irgb[:, :, 1], irgb[:, :, 2]
        h, s, l = _hsl[:, :, 0], _hsl[:, :, 1], _hsl[:, :, 2]

        imin, imax = irgb.min(2), irgb.max(2)
        iadd, isub = imax + imin, imax - imin

        ltop = (iadd != 510) * (iadd > 255)
        lbot = (iadd != 0) * (ltop == False)

        l[:] = iadd.astype(np.float) / 510

        fsub = isub.astype(np.float)
        s[ltop] = fsub[ltop] / (510 - iadd[ltop])
        s[lbot] = fsub[lbot] / iadd[lbot]

        not_same = imax != imin
        is_b_max = not_same * (imax == ib)
        not_same_not_b_max = not_same * (is_b_max == False)
        is_g_max = not_same_not_b_max * (imax == ig)
        is_r_max = not_same_not_b_max * (is_g_max == False) * (imax == ir)

        h[is_r_max] = ((0. + ig[is_r_max] - ib[is_r_max]) / isub[is_r_max])
        h[is_g_max] = ((0. + ib[is_g_max] - ir[is_g_max]) / isub[is_g_max]) + 2
        h[is_b_max] = ((0. + ir[is_b_max] - ig[is_b_max]) / isub[is_b_max]) + 4
        h[h < 0] += 6
        h[:] /= 6

    hsl = np.zeros(rgb.shape, dtype=np.float)
    cpus = multiprocessing.cpu_count()
    length = int(math.ceil(float(hsl.shape[0]) / cpus))
    line = 0
    threads = []
    while line < hsl.shape[0]:
        line_next = line + length
        thread = threading.Thread(target=core, args=(rgb[line:line_next], hsl[line:line_next]))
        thread.start()
        threads.append(thread)
        line = line_next

    for thread in threads:
        thread.join()

    return hsl


def hsl2rgb(hsl):

    def core(_hsl, _frgb):

        h, s, l = _hsl[:, :, 0], _hsl[:, :, 1], _hsl[:, :, 2]
        fr, fg, fb = _frgb[:, :, 0], _frgb[:, :, 1], _frgb[:, :, 2]

        q = np.zeros(l.shape, dtype=np.float)

        lbot = l < 0.5
        q[lbot] = l[lbot] * (1 + s[lbot])

        ltop = lbot == False
        l_ltop, s_ltop = l[ltop], s[ltop]
        q[ltop] = (l_ltop + s_ltop) - (l_ltop * s_ltop)

        p = 2 * l - q
        q_sub_p = q - p

        is_s_zero = s == 0
        l_is_s_zero = l[is_s_zero]
        per_3 = 1./3
        per_6 = 1./6
        two_per_3 = 2./3

        def calc_channel(channel, t):

            t[t < 0] += 1
            t[t > 1] -= 1
            t_lt_per_6 = t < per_6
            t_lt_half = (t_lt_per_6 == False) * (t < 0.5)
            t_lt_two_per_3 = (t_lt_half == False) * (t < two_per_3)
            t_mul_6 = t * 6

            channel[:] = p.copy()
            channel[t_lt_two_per_3] = p[t_lt_two_per_3] + q_sub_p[t_lt_two_per_3] * (4 - t_mul_6[t_lt_two_per_3])
            channel[t_lt_half] = q[t_lt_half].copy()
            channel[t_lt_per_6] = p[t_lt_per_6] + q_sub_p[t_lt_per_6] * t_mul_6[t_lt_per_6]
            channel[is_s_zero] = l_is_s_zero.copy()

        calc_channel(fr, h + per_3)
        calc_channel(fg, h.copy())
        calc_channel(fb, h - per_3)

    frgb = np.zeros(hsl.shape, dtype=np.float)
    cpus = multiprocessing.cpu_count()
    length = int(math.ceil(float(hsl.shape[0]) / cpus))
    line = 0
    threads = []
    while line < hsl.shape[0]:
        line_next = line + length
        thread = threading.Thread(target=core, args=(hsl[line:line_next], frgb[line:line_next]))
        thread.start()
        threads.append(thread)
        line = line_next

    for thread in threads:
        thread.join()

    return (frgb*255).round().astype(np.uint8)



def hsl_to_rgb(h, s, l):
    def hue_to_rgb(p, q, t):
        t += 1 if t < 0 else 0
        t -= 1 if t > 1 else 0
        if t < 1/6: return p + (q - p) * 6 * t
        if t < 1/2: return q
        if t < 2/3: p + (q - p) * (2/3 - t) * 6
        return p

    if s == 0:
        r, g, b = l, l, l
    else:
        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q
        r = hue_to_rgb(p, q, h + 1/3)
        g = hue_to_rgb(p, q, h)
        b = hue_to_rgb(p, q, h - 1/3)

    return r, g, b


class RectangleRotated:
    def __init__(self, p0, s, ang):
        (self.W, self.H) = s
        self.d = math.sqrt(self.W**2 + self.H**2) / 2.0 
        self.c = (int(p0[0] + self.W/2.0), int(p0[1] + self.H / 2.0))
        self.ang = ang 
        self.alpha = math.radians(self.ang)
        self.beta = math.atan2(self.H, self.W)

        self.P0 = (int(self.c[0] - self.d * math.cos(self.beta - self.alpha)), int(self.c[1] - self.d * math.sin(self.beta-self.alpha))) 
        self.P1 = (int(self.c[0] - self.d * math.cos(self.beta + self.alpha)), int(self.c[1] + self.d * math.sin(self.beta+self.alpha))) 
        self.P2 = (int(self.c[0] + self.d * math.cos(self.beta - self.alpha)), int(self.c[1] + self.d * math.sin(self.beta-self.alpha))) 
        self.P3 = (int(self.c[0] + self.d * math.cos(self.beta + self.alpha)), int(self.c[1] - self.d * math.sin(self.beta+self.alpha))) 

        self.verts = [self.P0, self.P1, self.P2, self.P3]

    def draw(self, image):
        points = np.array([self.P0, self.P1, self.P2, self.P3])
        image = cv2.fillPoly(image, [points], color=(0, 0, 0))