
import numpy as np 
import pandas as pd 
import pdb 

from torch.utils import data 


class ImageData(data.Dataset):
    """
    Args:
        rawdata: {ndarray tuple} as provided by different data loader
    Returns:
        initiates the dataset object with instance attributes needed later
    """

    def __init__(self, features, df):
        self.features = features
        self.df = df
        self.feature_list = [col for col in df if col.startswith('x')]

    def __len__(self):
        """Denotes the total number of samples
        """
        return self.features.shape[0]
    
    def __getitem__(self, idx):
        images = self.features[idx, :, :, ]
        tabular_data = self.df[self.feature_list].to_numpy()[idx, :]
        event = self.df['event'].to_numpy()[idx]
        time = self.df['time'].to_numpy()[idx]
        
        return images, tabular_data, event, time



def make_riskset(time):
    """Compute mask that represents each sample's risk set.
    Args:
        time: {np.ndarray} Observed event time sorted in descending order
    
    Returns:
        riskset {np.ndarray} Boolean matrix where the i-th row denotes the
        risk set of the i-th  instance, i.e. the indices j for which the observer time
        y_j >= y_i
    """

    assert time.ndim == 1
    #sort in descending order
    o = np.argsort(-time, kind="mergesort")
    n_samples = time.shape[0]
    risk_set = np.zeros((n_samples, n_samples), dtype=np.bool_)
    for i_org, i_sort in enumerate(o):
        ti  = time[i_sort]
        k = i_org
        while k < n_samples and ti == time[o[k]]:
            k += 1
        risk_set[i_sort, o[:k]] = True

    return risk_set



def load_data(path, split):
    """ loads data that is stored
    Args:
        path: {str} path to where data is stored
        split: {str} determines which data split to load (e.g. 'train', 'val', 'test')
    
    Returns:
        X: {np.array} images
        df: {pd.DataFrame}
    """
    X = np.load(file=f"{path}/X_{split}.npy").astype('float64')
    df = pd.read_csv(f"{path}/df_{split}.csv")

    return X, df

    

