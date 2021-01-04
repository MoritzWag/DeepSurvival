
import os 
import pdb 
import torch 
import numpy as np 
import pandas as pd 
import shutil 
import pygam

from src.data.sim import SimulationData2d


class SimPED(SimulationData2d):
    """
    """
    seed = 1328
    num_obs = 10000
    val_size = 0.2 * num_obs 
    num_groups = 4

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
        self.splines_list = [col for col in self.df if col.startswith('splines')]
        if self.part == 'test':
            self.eval_data = self.prepare_for_eval()


    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        df = self.df[self.df['index'] == index]
        images = self.x[index, :, :, ]
        tabular_data = df[self.features_list].to_numpy()

        offset = df['offset'].to_numpy()
        ped_status = df['ped_status'].to_numpy()
        index = df['index'].to_numpy()
        splines = df[self.splines_list].to_numpy()
        
        # return {'images': images, 'tabular_data': tabular_data,
        #         'offset': offset, 'ped_status': ped_status, 
        #         'index': index, 'splines': splines}

        return images, tabular_data, offset, ped_status, index, splines

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
        
        # in case that I want to simulate ped_data, I have to store the data first!
        cache = f"{self.final_path}/temp_storage"
        #os.path.join(self.final_path, "temp_storage")
        if not os.path.exists(cache):
            os.mkdir(cache)
        
        np.save(file=f"{cache}/riskgroups_train.npy", arr=riskgroup_train.astype(np.float64))
        np.save(file=f"{cache}/riskgroups_test.npy", arr=riskgroup_test.astype(np.float64))

        # run simulate_ped()
        self.simulate_ped(path=cache)
        df_train = pd.read_csv(f"{cache}/ped_train.csv")
        df_val = pd.read_csv(f"{cache}/ped_val.csv")
        df_test = pd.read_csv(f"{cache}/ped_test.csv")
        
        idx_train = df_train['id'].unique() - 1
        idx_val = df_val['id'].unique() - 1

        X_val = X_train[idx_val, :, :, :]
        X_train = X_train[idx_train, :, :, :]
        
        # reindex df_train 
        df_train['index'] = 0
        idx_unique_train = df_train['id'].unique()
        for idx in range(len(idx_unique_train)):
            df_train.loc[df_train['id'] == idx_unique_train[idx], 'index'] = idx

        # reindex df_val
        df_val['index'] = 0
        idx_unique_val = df_val['id'].unique()
        for idx in range(len(idx_unique_val)):
            df_val.loc[df_val['id'] == idx_unique_val[idx], 'index'] = idx

        # reindex df_test
        df_test['index'] = 0
        idx_unique_test = df_test['id'].unique()
        for idx in range(len(idx_unique_test)):
            df_test.loc[df_test['id'] == idx_unique_test[idx], 'index'] = idx
        

        shutil.rmtree(cache)    

        # save data
        df_train.to_csv(f"{self.final_path}/{self.data_type}/df_train.csv")
        df_val.to_csv(f"{self.final_path}/{self.data_type}/df_val.csv")
        df_test.to_csv(f"{self.final_path}/{self.data_type}/df_test.csv")

        np.save(file=f"{self.final_path}/{self.data_type}/X_train.npy", arr=X_train)
        np.save(file=f"{self.final_path}/{self.data_type}/X_val.npy", arr=X_val)
        np.save(file=f"{self.final_path}/{self.data_type}/X_test.npy", arr=X_test)


    def simulate_ped(self, path):
        os.system(f"Rscript ./src/src/data/sim_ped.R --path {path} --val_size {self.val_size}")

    def prepare_for_eval(self):
        """
        """
        y = []
        statuses = []
        times = []
        times_unique = np.unique(self.df['tend'])
        times_unique[-1] -= 0.01
        num_obs = self.df['index'].unique()
        for obs in range(len(num_obs)):
            max_time = max(self.df.loc[self.df['index'] == obs, ]['tend'])
            times.append(max_time)
            status = self.df.loc[self.df['index'] == obs, ]['ped_status'][-1:]
            statuses.append(status)
            status = bool(status.values)
            instance = (status, max_time)
            y.append(instance)
        
        dt = np.dtype('bool, float')
        y = np.array(y, dtype=dt)
        statuses = np.stack(statuses).squeeze(1)
        times = np.stack(times)

        return {'y': y, 'event': statuses, 'time': times, 'times_unique': times_unique}


    def true_hazard(self):
        pass

