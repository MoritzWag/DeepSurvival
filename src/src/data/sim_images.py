
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
import random

import torchvision.transforms as transforms
import torchvision.datasets as datasets 
import matplotlib.pyplot as plt

from torch.utils import data
from sklearn.model_selection import train_test_split
from torchvision.utils import save_image

from src.data.sim import SimCoxPH


class SimImages(SimCoxPH):
    """
    """
    seed = 1328
    num_obs = 1000
    val_size = 0.2 * num_obs 
    num_groups = 7
    # mean_survival_time = 365.0
    mean_survival_time = 20.0
    prob_censored = 0.2
    
    def __init__(self, 
                 root, 
                 part='train',
                 download=True,
                 base_folder='sim_cont',
                 data_type='coxph',
                 n_dim=3):
        
        self.root = root
        self.part = part 
        self.base_folder = base_folder 
        self.data_type = data_type
        self.n_dim = n_dim
        self.final_path = os.path.join(self.root, self.base_folder)

        if self.base_folder in ['sim_cont', 'sim_rr', 'sim_cont_mult']:
            self.continuous = True
        else:
            self.continuous = False

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

        if not self.continuous:
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
            if self.base_folder == "sim_cont":
                X_train, X_test, labels_train, labels_test = self.simulate_continuous_images(img_size=28,
                                                                                            num_obs=self.num_obs,
                                                                                            n_dim=self.n_dim,
                                                                                            pos_random=True,
                                                                                            num_shapes=1)
            
            if self.base_folder == "sim_cont_mult":
                X_train, X_test, labels_train, labels_test = self.simulate_continuous_images(img_size=28,
                                                                                             num_obs=self.num_obs,
                                                                                             n_dim=self.n_dim,
                                                                                             pos_random=True,
                                                                                             num_shapes=3)
            elif self.base_folder == "sim_rr":
                X_train, X_test, labels_train, labels_test = self.simulate_circle(img_size=28,
                                                                                  num_obs=self.num_obs,
                                                                                  n_dim=self.n_dim,
                                                                                  pos_random=True)

            X_train, X_val, df_train, df_val, df_test = self.gen_surv_times_cont(X_train,
                                                                                 X_test,
                                                                                 labels_train,
                                                                                 labels_test)
        
        # save data      
        df_train.to_csv(f"{self.final_path}/{self.data_type}/df_train.csv")
        df_val.to_csv(f"{self.final_path}/{self.data_type}/df_val.csv")
        df_test.to_csv(f"{self.final_path}/{self.data_type}/df_test.csv")

        np.save(file=f"{self.final_path}/{self.data_type}/X_train.npy", arr=X_train)
        np.save(file=f"{self.final_path}/{self.data_type}/X_val.npy", arr=X_val)
        np.save(file=f"{self.final_path}/{self.data_type}/X_test.npy", arr=X_test)

    def simulate_coxph_riskscores_cont(self, labels):
        """
        """
        random = np.random.RandomState(self.seed)
        num_obs = labels.shape[0]
        x1 = np.repeat(0.00001, num_obs)
        x2 = np.repeat(0.00001, num_obs)

        min_l = np.min(labels)
        max_l = np.max(labels)
        mean_l = (max_l - min_l) / 2
        labels_norm = (labels - mean_l) / mean_l

        df = pd.DataFrame(data={'labels': labels, 'labels_norm': labels_norm, 'x1': x1, 'x2': x2})

        df['risk_scores'] = 0.25*df['x1'] - 0.3*df['x2'] \
                            + 1.0*df['labels_norm']

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
    
    def gen_surv_times_cont(self, X_train, X_test, label_train, label_test):
        """
        """
        labels = np.concatenate((label_train, label_test))
        #df = self.hue_to_riskscore(hue=labels)
        df = self.simulate_coxph_riskscores_cont(labels)

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

    def simulate_continuous_images(self, 
                                   img_size,
                                   num_obs,
                                   n_dim,
                                   pos_random,
                                   num_shapes):

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
                                            n_dim=n_dim,
                                            pos_random=pos_random)
            
            img = rgb2hsl(img)
            img = np.transpose(img, axes=(2, 0, 1))
            images_final.append(img)
        
        images_final = np.stack(images_final)

        X_train, X_test, labels_train, labels_test = train_test_split(images_final,
                                                                      color_h,
                                                                      test_size=int(0.1*num_obs),
                                                                      random_state=self.seed)

        return X_train, X_test, labels_train, labels_test

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
            
            img = np.transpose(img, axes=(1, 2, 0))
            img = rgb2hsl(img)
            img = np.transpose(img, axes=(2, 0, 1))
            images_final.append(img)
            
        images_final = np.stack(images_final)

        # split data into train and test 
        X_train, X_test, riskgroup_train, riskgroup_test = train_test_split(images_final,
                                                                            groups,
                                                                            test_size=int(0.1*num_obs),
                                                                            stratify=groups,
                                                                            random_state=self.seed)
        
        return X_train, riskgroup_train, X_test, riskgroup_test

    def simulate_circle(self,
                        img_size,
                        num_obs,
                        n_dim,
                        pos_random):
        """
        """
        rand = np.random.RandomState(self.seed)
        gray_scale = 0.5
        radiuses = rand.randint(2, 12, size=num_obs)

        images_final = []
        for i in range(num_obs):
            img = np.zeros((img_size, img_size, 1), dtype='float32')
            img[:, :] = gray_scale
            radius = radiuses[i]
            center = (14, 14)
            img = cv2.circle(img, center, radius, (0, 0, 0), thickness=-1)
            img = np.transpose(img, axes=(2, 0, 1))
            images_final.append(img)
        
        images_final = np.stack(images_final)

        X_train, X_test, radius_train, radius_test = train_test_split(images_final,
                                                                radiuses,
                                                                test_size=int(0.1*num_obs),
                                                                stratify=radiuses,
                                                                random_state=self.seed)

        return X_train, X_test, radius_train, radius_test

    def simulate_rotated_rectangles(self,
                                    img_size,
                                    num_obs,
                                    n_dim,
                                    pos_random):
        """
        """
        rand = np.random.RandomState(self.seed)
        lengths = rand.randint(3, 20, size=num_obs)
        gray_scale = 190
        color_assignment = []

        images_final = []
        for i in range(num_obs):
            img = np.zeros((img_size, img_size, 1), dtype='float32')
            img[:, :] = gray_scale
            ang = random.randrange(0, 360, 10)
            length = lengths[i]
            (W, H) = (length, length)
            P0 = (rand.randint(low=4, high=40, size=1), rand.randint(low=4, high=40, size=1))
            rr = RectangleRotated(P0, (W, H), ang)
            rr.draw(img)
            img = np.transpose(img, axes=(1, 2, 0))
            images_final.append(img)
        
        images_final = np.stack(images_final)

        # split data intro train and test 
        X_train, X_test, length_train, length_test = train_test_split(images_final,
                                                                      lengths,
                                                                      test_size=int(0.1*num_obs),
                                                                      stratify=lengths,
                                                                      random_state=self.seed)

        return X_train, X_test, length_train, length_test

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
                              center=14,
                              radius=5,
                              color=color,
                              n_dim=n_dim,
                              pos_random=pos_random)

        return img

def rectangle_mask(img_size, 
                   length, 
                   color, 
                   n_dim,
                   pos_random=True,
                   seed=1328):
    
    random = np.random.RandomState(seed)
    x = random.randint(length, img_size)
    y = random.randint(img_size - length)

    img = np.zeros((img_size, img_size) + (3,), dtype=np.float32)
    img[x - length : x, y : y + length, :] = color 
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

    return img



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
        # image = cv2.fillPoly(image, [points], color=(0, 0, 0))
        image = cv2.fillPoly(image, [points], color=(0))