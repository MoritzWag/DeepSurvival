
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

    def __init__(self, rawdata):
        self.rawdata = rawdata

    def __len__(self):
        """Denotes the total number of samples
        """
        return(len(self.rawdata[0]))
    
    def __getitem__(self, idx):
        images = self.rawdata[0][idx, :, :, ]
        event = self.rawdata[1][idx]
        time = self.rawdata[2][idx]
        riskset =  make_riskset(time)

        return images, event, time, riskset



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
    n_samples = len(time)
    risk_set = np.zeros((n_samples, n_samples), dtype=np.bool_)
    for i_org, i_sort in enumerate(o):
        ti  = time[i_sort]
        k = i_org
        while k < n_samples and ti == time[o[k]]:
            k += 1
        risk_set[i_sort, o[:k]] = True
    return risk_set