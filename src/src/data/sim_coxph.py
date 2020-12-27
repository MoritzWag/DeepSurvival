
import os 
import pdb 
import torch 
import numpy as np 
import pandas as pd 
import shutil 

from src.data.sim import SimulationData2d
from sklearn.model_selection import train_test_split


class SimCoxPH(SimulationData2d):
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
                 base_folder="mnist2",
                 data_type='coxph'):
        self.root = root
        self.part = part 
        self.base_folder = base_folder 
        self.data_type = data_type
        self.final_path = os.path.join(self.root, self.base_folder)

        if download: 
            self.download()
        self.x, self.df = self.load_dataset(path=self.final_path, part=self.part)
        self.features_list = [col for col in self.df if col.startswith('x')]
        if self.part == 'test':
            self.eval_data = self.prepare_for_eval()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        images = self.x[index, :, :, ]
        tabular_data = self.df[self.features_list].to_numpy()[index, :]
        event = self.df['event'].to_numpy()[index]
        time = self.df['time'].to_numpy()[index]

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

        if self.base_folder == 'mnist2':
            X_train, riskgroup_train, X_test, riskgroup_test = self.download_mnist()
        else:
            X_train, riskgroup_train, X_test, riskgroup_test = self.simulate_images(img_size=28,
                                                                                    num_obs=self.num_obs,
                                                                                    n_groups=self.num_groups,
                                                                                    pos_random=True,
                                                                                    num_shapes=3)

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
                                                            test_size=self.val_size,
                                                            stratify=df_train['riskgroup'],
                                                            random_state=self.seed)

        # save data      
        df_train.to_csv(f"{self.final_path}/{self.data_type}/df_train.csv")
        df_val.to_csv(f"{self.final_path}/{self.data_type}/df_val.csv")
        df_test.to_csv(f"{self.final_path}/{self.data_type}/df_test.csv")

        np.save(file=f"{self.final_path}/{self.data_type}/X_train.npy", arr=X_train)
        np.save(file=f"{self.final_path}/{self.data_type}/X_val.npy", arr=X_val)
        np.save(file=f"{self.final_path}/{self.data_type}/X_test.npy", arr=X_test)


    def simulate_coxph_riskscores(self, riskgroups):
        """
        """
        random = np.random.RandomState(self.seed)
        num_obs = riskgroups.shape[0]
        x1 = random.uniform(-2, 3, size=num_obs)
        x2 = random.uniform(0, 5, size=num_obs)

        df = pd.DataFrame(data={'riskgroup': riskgroups, 'x1': x1, 'x2': x2})

        df['risk_scores'] = -0.5 + 0.25*df['x1'] - 0.3*df['x2'] \
                                    + 0.5*(df['riskgroup'] == 0) \
                                    - 1*(df['riskgroup'] == 1) \
                                    + 0.3*(df['riskgroup'] == 2) \
                                    - 0.8*(df['riskgroup'] == 3)
        
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