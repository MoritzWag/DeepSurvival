
import os 
import pdb 
import torch 
import numpy as np 
import pandas as pd
import h5py
import torchvision
import patsy

from torch.utils import data 
from torchvision.utils import save_image
from sklearn.impute import SimpleImputer

from src.data.augment import (spatial_transform, 
                              intensity_transform,
                              RicianNoise,
                              ElasticDeformationsBspline)


class ADNI(data.Dataset):
    """
    """
    seed = 1328
    #features_list = ['TAU', 'real_age']
    features_list = ['ABETA', 'APOE4', 'AV45',
                     'C(PTGENDER)[T.Male]',
                     'FDG', 'PTAU', 'PTEDUCAT', 
                     'TAU', 'real_age', 'age_male',
                     'bs_1', 'bs_2', 'bs_3', 'bs_4']

    mean_survival_time = 20.0
    prob_censored = 0.45

    def __init__(self,
                 root, 
                 part='train',
                 transform=True,
                 download=True,
                 base_folder='adni',
                 data_type='coxph',
                 simulate=False,
                 trial=None):

        self.root = root
        self.part = part 
        self.base_folder = base_folder
        self.data_type = data_type
        self.final_path = os.path.join(self.root, self.base_folder)
        self.simulate_survival = simulate
        self.trial = trial

        self.do_transform = transform
        self.spatial_transform = spatial_transform
        self.intensity_transform = intensity_transform

        if self.do_transform:
            if self.trial is None:
                transforms = []
                transforms.append(RicianNoise(noise_level=[0, 1]))
                transforms.append(ElasticDeformationsBspline(num_controlpoints=[1], sigma=[0, 1]))
            else:
                transforms = []
                transforms.append(RicianNoise(noise_level=[0, 
                                                          self.trial.suggest_int('nl2', low=5, high=20, step=2)]))
                transforms.append(ElasticDeformationsBspline(
                    num_controlpoints=[self.trial.suggest_int("num_ct", low=0, high=10, step=2)],
                    sigma=[0, self.trial.suggest_int('sigma2', low=10, high=20, step=2)]
                ))
            # transforms.append(RicianNoise(noise_level=[9, 10]))
            # transforms.append(ElasticDeformationsBspline(num_controlpoints=[15], sigma=[2, 10]))

            self.transforms = torchvision.transforms.Compose(transforms)
        
        if self.part == 'val':
            self.file = f"/home/moritz/adni/0-valid.h5"
        else:
            self.file = f"/home/moritz/adni/0-{self.part}.h5"

        if download:
            self.download()

        if self.simulate_survival:
            if 'real_age' in self.features_list:
                self.features_list.remove('real_age')

        self.df = self.load_dataset(path=self.final_path, part=self.part)
        self.eval_data = self.prepare_for_eval()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
            
        images = np.load(file=f"{self.final_path}/{self.part}/image_{index}.npy").astype('float64')

        if self.do_transform:
            images = self.transforms(images)
            # images = self.spatial_transform(images)
            # images = self.intensity_transform(images)
        images = torch.tensor(images).to(self.device)

        # if simulate survival => self.simulate_survival(tabular_data) -> time, event
        event = torch.tensor(self.df['event'].to_numpy()[index]).to(self.device)
        time = torch.tensor(self.df['time'].to_numpy()[index]).to(self.device)
        # if self.simulate_survival:
        #     self.features_list.remove('real_age')
        tabular_data = torch.tensor(self.df[self.features_list].to_numpy()[index, :]).to(self.device)
        
        return images, tabular_data, event, time

    def download(self):
        if not os.path.exists(self.final_path):
            os.mkdir(self.final_path)

        storage_path = os.path.join(self.final_path, self.part)
        if not os.path.exists(storage_path):
            os.mkdir(storage_path)
        else:
            return 

        df = self.handle_rawdata(file=self.file)
        df.to_csv(f"{self.final_path}/df_{self.part}.csv")

    def handle_rawdata(self, file):
        """
        """
        np.random.seed(self.seed)

        df = pd.DataFrame()
        
        with h5py.File(file, mode="r") as fin:
            columns = [x for x in fin['stats']['tabular']['columns']]
            idx = 0
            for i, (image_id, grp) in enumerate(fin.items()):
                try:
                    img = grp["norm_wimt_converted"][:]
                except:
                    continue
                
                if self.base_folder == 'adni2d' or self.base_folder == 'adni_sim':
                    img = img[:, 80, :]
                
                img = np.expand_dims(img, axis=(0))
                img = minmax_normalize(img)
                img = torch.from_numpy(img)
                img = torchvision.transforms.functional.rotate(img, 90).numpy()

                is_event, observed_time = grp.attrs['event'], grp.attrs['time']

                surv_info = pd.Series({'event': is_event, 'time': observed_time})
                features = pd.Series(grp['tabular'][:], index=columns)
                tabular_data = pd.concat([surv_info, features])
                df = df.append(tabular_data, ignore_index=True)
                np.save(file=f"{self.final_path}/{self.part}/image_{idx}.npy", arr=img)
                idx += 1


        if self.simulate_survival:
            observed_time, is_event = self.generate_survival_times(age=df['real_age'].to_numpy())
            df['event'] = is_event.astype('int')
            df['time'] = observed_time
        else:
            pass
            # one-hot encode event "no" = 0, "yes" = 1
            df['event'] = df['event'].replace({'no': 0, 'yes': 1})

            # include b-splines expansion of degree 4
            real_age = {'real_age': df['real_age']}
            bs = patsy.dmatrix("bs(real_age, df=4)", real_age)

            b_splines = {}
            for i in range(bs.shape[1]):
                if i == 0:
                    continue
                df[f"bs_{i}"] = bs[:, i]

            # generate interactione effect between age and gender
            df['age_male'] = df['C(PTGENDER)[T.Male]'] * df['real_age']

            df = self.impute_missing_data(df)

        return df
    
    def load_dataset(self, path, part):
        """
        """
        df = pd.read_csv(f"{path}/df_{part}.csv")

        return df 

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


    def generate_survival_times(self, age):
        """
        """
        # maybe try different normalization !!!! => (-1, 1)
        # and then only one factor to multiply with !!!
        # age_norm = (age - 50.0) / 50.0

        age_norm = 2 * (age - np.min(age)) / (np.max(age) - np.min(age)) - 1
        
        
        # age_norm = age

        random = np.random.RandomState(self.seed)
        epsilon = random.uniform(0, 2, size=age.shape[0])

        riskscores = 4.0 * age_norm + epsilon
        # riskscores = 25 * age_norm * (age >= 90) \
        #             + 15 * age_norm * ((age >= 85) & (age < 90)) \
        #             + 5 * age_norm * ((age >= 75) & (age < 85)) \
        #             - 15 * age_norm * ((age >= 70) & (age < 75)) \
        #             - 25 * age_norm * ((age >= 60) & (age < 70)) + epsilon
        # pdb.set_trace()
        # riskscores = 1.5 * age 
        baseline_hazard = 1. / self.mean_survival_time
        scale = baseline_hazard * np.exp(riskscores)
        u = random.uniform(low=0, high=1, size=age.shape[0])
        t = - np.log(u) / scale

        # generate time of censoring
        qt = np.quantile(t, 1.0 - self.prob_censored)
        c = random.uniform(low=t.min(), high=qt)

        observed_event = t <= c
        observed_time = np.where(observed_event, t, c)

        return observed_time, observed_event

    def impute_missing_data(self, df):
        """
        """

        df.loc[df['C(ABETA_MISSING)[T.1]'] == 1.0, 'ABETA'] = np.nan
        df.loc[df['C(TAU_MISSING)[T.1]'] == 1.0, 'TAU'] = np.nan
        df.loc[df['C(PTAU_MISSING)[T.1]'] == 1.0, 'PTAU'] = np.nan
        df.loc[df['C(FDG_MISSING)[T.1]'] == 1.0, 'FDG'] = np.nan
        df.loc[df['C(AV45_MISSING)[T.1]'] == 1.0, 'AV45'] = np.nan

        df = df.apply(lambda x: x.fillna(x.mean()),axis=0)

        return df 

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

