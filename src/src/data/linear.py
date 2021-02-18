import os 
import pdb 
import torch 
import numpy as np 
import pandas as pd
import h5py
import torchvision

from torch.utils import data 

class LinearData(data.Dataset):
    """
    """
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
    
    def __init__(self, root,
                part='train',
                base_folder='adni',
                split=0,
                seed=1328):
    
        self.root = root
        self.part = part 
        self.split = split
        self.base_folder = base_folder
        self.seed = seed
        self.final_path = os.path.join(self.root, self.base_folder)

        self.df = self.load_dataset(path=self.final_path, part=self.part, split=self.split)
        self.eval_data = self.prepare_for_eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        event = torch.tensor(self.df['event'].to_numpy()[index]).to(self.device)
        time = torch.tensor(self.df['time'].to_numpy()[index]).to(self.device)
        tabular_data = torch.tensor(self.df[self.features_list].to_numpy()[index, :]).to(self.device)
        
        return tabular_data, event, time

    def load_dataset(self, path, part, split):
        """
        """
        df = pd.read_csv(f"{path}/df_{part}_{split}.csv")

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

