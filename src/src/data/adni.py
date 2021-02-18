
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from patsy.contrasts import Poly
from torchvision import transforms


from src.data.augment import (spatial_transform, 
                              intensity_transform,
                              RicianNoise,
                              ElasticDeformationsBspline,
                              ImgTransforms, 
                              data_augmentation)


to_pil = transforms.ToPILImage()
to_tensor = transforms.ToTensor()

class ADNI(data.Dataset):
    """
    """
    num_cvsplits = 10
    features_list = ['ABETA', 'APOE4', 'AV45',
                     'C(PTGENDER)[T.Male]',
                     'FDG', 'PTAU', 
                     'TAU', 'real_age', 'age_male',
                     'bs_1', 'bs_2', 'bs_3', 'bs_4',
                     '.Linear', '.Quadratic', '.Cubic',
                     'C(ABETA_MISSING)[T.1]',
                     'C(TAU_MISSING)[T.1]',
                     'C(PTAU_MISSING)[T.1]',
                     'C(FDG_MISSING)[T.1]',
                     'C(AV45_MISSING)[T.1]'
                     ]

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
                 split=0,
                 seed=1328,
                 trial=None):

        self.root = root
        self.part = part 
        self.base_folder = base_folder
        self.data_type = data_type
        self.final_path = os.path.join(self.root, self.base_folder)
        self.simulate_survival = simulate
        self.seed = seed
        self.trial = trial
        self.split = split

        self.do_transform = transform
        #self.img_transform = data_augmentation
        self.img_transform = ImgTransforms()
        self.p_aug = 0.5
        
        if self.part == 'val':
            self.file = f"/home/moritz/adni/0-valid.h5"
        else:
            self.file = f"/home/moritz/adni/0-{self.part}.h5"

        if download:
            self.download()

        if self.simulate_survival:
            if 'real_age' in self.features_list:
                self.features_list.remove('real_age')
        self.df = self.load_dataset(path=self.final_path, part=self.part, split=self.split)
        self.eval_data = self.prepare_for_eval()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        if self.part != 'test':
            images = np.load(file=f"{self.final_path}/{self.part}/split_{self.split}/image_{index}.npy").astype('float64')
        else:
            images = np.load(file=f"{self.final_path}/{self.part}/image_{index}.npy").astype('float64')
        if self.do_transform:
            if np.random.rand() < self.p_aug:
                images = self.img_transform(images)
                images = torch.from_numpy(images).to(self.device)
            else:
                images = torch.from_numpy(images).to(self.device)
        else:
            images = torch.from_numpy(images).to(self.device)

        # if simulate survival => self.simulate_survival(tabular_data) -> time, event
        event = torch.tensor(self.df['event'].to_numpy()[index]).to(self.device)
        time = torch.tensor(self.df['time'].to_numpy()[index]).to(self.device)
        tabular_data = torch.tensor(self.df[self.features_list].to_numpy()[index, :]).to(self.device)
        
        return images, tabular_data, event, time

    def download(self):
        if not os.path.exists(self.final_path):
            os.mkdir(self.final_path)
        else:
            return
        
        self.handle_rawdata()

    def handle_rawdata(self):
        """
        """
        
        print("preprocess rawdata")
        np.random.seed(self.seed)

        storage_path_val = os.path.join(self.final_path, 'val')
        if not os.path.exists(storage_path_val):
            os.mkdir(storage_path_val)

        df_val = pd.DataFrame()
        file_val = f"/home/moritz/adni/0-valid.h5"

        images = []

        with h5py.File(file_val, mode='r') as fin:
            columns = [x for x in fin['stats']['tabular']['columns']]
            idx_val = 0
            for i, (image_id, grp) in enumerate(fin.items()):
                try:
                    img = grp["norm_wimt_converted"][:]
                except:
                    continue

                if self.base_folder == 'adni2d' or self.base_folder == 'adni_sim':
                    # img = img[64, :, :]
                    img = img[87, :, :]

                
                img = np.expand_dims(img, axis=(0))
                img = minmax_normalize(img, lower_bound=0.0, upper_bound=255.0)
                img = torch.from_numpy(img)
                img = to_tensor(torchvision.transforms.functional.rotate(to_pil(img), 90)).numpy()

                is_event, observed_time = grp.attrs['event'], grp.attrs['time']

                surv_info = pd.Series({'event': is_event, 'time': observed_time, 'split': 0})
                features = pd.Series(grp['tabular'][:], index=columns)
                tabular_data = pd.concat([surv_info, features])
                df_val = df_val.append(tabular_data, ignore_index=True)
                images.append(img)
                #np.save(file=f"{storage_path_val}/image_{idx_val}.npy", arr=img)

                idx_val += 1
        
        storage_path_train = os.path.join(self.final_path, 'train')
        if not os.path.exists(storage_path_train):
            os.mkdir(storage_path_train)

        df_train = pd.DataFrame()
        file_train = f"/home/moritz/adni/0-train.h5"
        idx_train = 0
        with h5py.File(file_train, mode='r') as fin:
            columns = [x for x in fin['stats']['tabular']['columns']]
            idx = 0
            for i, (image_id, grp) in enumerate(fin.items()):

                try: 
                    img = grp["norm_wimt_converted"]
                except:
                    continue 

                if self.base_folder == 'adni2d' or self.base_folder == 'adni_sim':
                    # img = img[64, :, :]
                    img = img[87, :, :]


                img = np.expand_dims(img, axis=(0))
                img = minmax_normalize(img, lower_bound=0.0, upper_bound=255.0)
                img = torch.from_numpy(img)
                #to_pil = transforms.ToPILImage()
                img = to_tensor(torchvision.transforms.functional.rotate(to_pil(img), 90)).numpy()

                is_event, observed_time = grp.attrs['event'], grp.attrs['time']

                surv_info = pd.Series({'event': is_event, 'time': observed_time, 'split': 1})
                features = pd.Series(grp['tabular'][:], index=columns)
                tabular_data = pd.concat([surv_info, features])
                df_train = df_train.append(tabular_data, ignore_index=True)
                images.append(img)
                #np.save(file=f"{storage_path_train}/image_{idx_train}.npy", arr=img)
                idx_train += 1


        storage_path_test = os.path.join(self.final_path, 'test')
        if not os.path.exists(storage_path_test):
            os.mkdir(storage_path_test)

        df_test = pd.DataFrame()
        file_test = f"/home/moritz/adni/0-test.h5"
        with h5py.File(file_test, mode='r') as fin:
            columns = [x for x in fin['stats']['tabular']['columns']]
            idx_test = 0
            for i, (image_id, grp) in enumerate(fin.items()):

                try: 
                    img = grp["norm_wimt_converted"]
                except:
                    continue 

                if self.base_folder == 'adni2d' or self.base_folder == 'adni_sim':
                    # img = img[64, :, :]
                    img = img[87, :, :]
                
                img = np.expand_dims(img, axis=(0))
                img = minmax_normalize(img, lower_bound=0.0, upper_bound=255.0)
                img = torch.from_numpy(img)
                img = to_tensor(torchvision.transforms.functional.rotate(to_pil(img), 90)).numpy()

                is_event, observed_time = grp.attrs['event'], grp.attrs['time']

                surv_info = pd.Series({'event': is_event, 'time': observed_time, 'split': 2})
                features = pd.Series(grp['tabular'][:], index=columns)
                tabular_data = pd.concat([surv_info, features])
                df_test = df_test.append(tabular_data, ignore_index=True)
                images.append(img)
                #np.save(file=f"{storage_path_test}/image_{idx_test}.npy", arr=img)
                idx_test += 1


        images = np.stack(images)
        # merge train, val and test df 
        df = pd.concat([df_val, df_train, df_test]).reset_index(drop=True)

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

        # apply orthogonal polynomial coding for education
        _, bins = np.histogram(df['PTEDUCAT'].to_numpy(), 15)
        try: 
            educcat = np.digitize(df['PTEDUCAT'].to_numpy(), bins, True)
        except:
            educcat = np.digitize(df['PTEDUCAT'].to_numpy(), bins)
        
        df['educcat'] = educcat
        levels = df['educcat'].unique().tolist()
        levels.sort()
        contrast = Poly(scores=levels).code_without_intercept(levels)

        suffices = contrast.column_suffixes
        sff_idx = 0
        for var in suffices:
            df[var] = ""
            # contrast.matrix.shape = (15, 14)
            for idx in range(contrast.matrix.shape[0]):
                cat = levels[idx]
                df.loc[df['educcat'] == cat, var] = contrast.matrix[idx, sff_idx]
            sff_idx += 1

        # feature normalization
        scaler = MinMaxScaler().fit(df.to_numpy())
        X_scaled = scaler.transform(df.to_numpy())

        df_scaled = pd.DataFrame(X_scaled, columns=df.columns)
        df_scaled['event'] = df['event']
        df_scaled['time'] = df['time']


        X_train, X_test, df_train, df_test = train_test_split(images, df_scaled,
                                                              train_size=int(0.9*df_scaled.shape[0]),
                                                              random_state=self.seed,
                                                              stratify=df_scaled['event'])
        
        df_test.to_csv(f"{self.final_path}/df_test.csv")

        for idx in range(X_test.shape[0]):
            img = X_test[idx]
            np.save(f"{storage_path_test}/image_{idx}.npy", arr=img)


        indeces = np.arange(X_train.shape[0])
        kf = KFold(n_splits=self.num_cvsplits)
        kf.get_n_splits(indeces)

        split = 0
        for train_index, val_index in kf.split(indeces):

            x_train, df_t = X_train[train_index], df_train.iloc[train_index, :]
            x_val, df_v = X_train[val_index], df_train.iloc[val_index, :]

            storage_path_train_split = os.path.join(storage_path_train, f"split_{split}")
            os.mkdir(storage_path_train_split)
            for idx in range(x_train.shape[0]):
                img = x_train[idx]
                np.save(file=f"{storage_path_train_split}/image_{idx}.npy", arr=img)
            df_t.to_csv(f"{self.final_path}/df_train_{split}.csv")
            
            storage_path_val_split = os.path.join(storage_path_val, f"split_{split}")
            os.mkdir(storage_path_val_split)
            for idx in range(x_val.shape[0]):
                img = x_val[idx]
                np.save(file=f"{storage_path_val_split}/image_{idx}.npy", arr=img)
            df_v.to_csv(f"{self.final_path}/df_val_{split}.csv")
            split += 1

        

        # for split in range(self.num_cvsplits):
        #     print(f"generate split number: {split+1}/{self.num_cvsplits}")
        #     pdb.set_trace()
        #     seed = np.random.seed(self.seed)
        #     print(seed)

        #     #split dataframes in val, train, and test
        #     X_train, X_val, df_train, df_val = train_test_split(images, df_scaled,
        #                                                         train_size=int(0.8*df_scaled.shape[0]),
        #                                                         random_state=seed)

        #     X_val, X_test, df_val, df_test = train_test_split(X_val, df_val,
        #                                                     train_size=int(0.5*df_val.shape[0]),
        #                                                     random_state=seed)


        #     storage_path_train_split = os.path.join(storage_path_train, f"split_{split}")
        #     os.mkdir(storage_path_train_split)

        #     for idx in range(X_train.shape[0]):
        #         img = X_train[idx]
        #         np.save(file=f"{storage_path_train_split}/image_{idx}.npy", arr=img)

        #     storage_path_val_split = os.path.join(storage_path_val, f"split_{split}")
        #     os.mkdir(storage_path_val_split)

        #     for idx in range(X_val.shape[0]):
        #         img = X_val[idx]
        #         np.save(file=f"{storage_path_val_split}/image_{idx}.npy", arr=img)

        #     storage_path_test_split = os.path.join(storage_path_test, f"split_{split}")
        #     os.mkdir(storage_path_test_split)

        #     for idx in range(X_test.shape[0]):
        #         img = X_test[idx]
        #         np.save(file=f"{storage_path_test_split}/image_{idx}.npy", arr=img)

        #     # save dataframes
        #     df_val.to_csv(f"{self.final_path}/df_val_{split}.csv")
        #     df_train.to_csv(f"{self.final_path}/df_train_{split}.csv")
        #     df_test.to_csv(f"{self.final_path}/df_test_{split}.csv")

    
    def load_dataset(self, path, part, split):
        """
        """
        if part != 'test':
            df = pd.read_csv(f"{path}/df_{part}_{split}.csv")
        else:
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

