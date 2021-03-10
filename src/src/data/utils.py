
import numpy as np 
import pandas as pd 
import pdb 
import cv2
import torch

from torch import nn
from torch.utils import data 
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from src.data.adni import ADNI
from src.data.sim_coxph import SimCoxPH


def get_dataloader(root, part, transform,
                   base_folder, data_type, split, batch_size):
    """
    """

    if base_folder == "adni2d":
        data = ADNI(root=root, 
                    part=part,
                    transform=transform,
                    base_folder=base_folder,
                    data_type=data_type,
                    split=split)
    else:
        data = SimCoxPH(root=root,
                        part=part,
                        base_folder=base_folder,
                        data_type=data_type,
                        n_dim=3 if base_folder == "simulation" else 1)

    data_gen = DataLoader(dataset=data,
                          batch_size=len(data) if batch_size == -1 else batch_size,
                          collate_fn=cox_collate_fn,
                          shuffle=False)

    eval_data = data.eval_data

    return data_gen, eval_data


def get_eval_data(batch, model):
    """
    """
    # linear_coefficients = np.load(file="/home/moritz/DeepSurv/DeepSurvival/linear_weights/weights.npy").astype('float64')
    # linear_coefficients = np.expand_dims(linear_coefficients, axis=0)

    # model.linear.weight.data[:, :model.structured_input_dim] = nn.Parameter(torch.FloatTensor(linear_coefficients))
    prediction = model(**batch).cpu().detach().numpy()
    prediction = prediction.squeeze(1)
    
    try:
        image_prediction = model.predict_on_images(**batch)
        image_prediction = image_prediction.squeeze(1)
        return {'riskscores': prediction, 'riskscore_img': image_prediction}
    except:
        return {'riskscores': prediction}


def cox_collate_fn(batch, time_index=-1, data_collate=default_collate):
    """Create risk set from batch
    """
    transposed_data = list(zip(*batch))
    y_time = np.array(transposed_data[time_index])
    
    data = []
    for b in transposed_data:
        bt = data_collate(b)
        data.append(bt)
    
    data.append(torch.from_numpy(make_riskset(y_time)))

    return {'images': data[0].float(), 'tabular_data': data[1], 
            'event': data[2], 'time': data[3], 'riskset': data[4]}
    
def safe_normalize(x):
    """Normalize risk scores to avoid exp underflowing.

    Note that only risk scores relative to each other matter.
    If minimum risk score is negative, we shift scores so minimum
    is at zero.
    """

    x_min, _ = torch.min(x, dim=0)
    c = torch.zeros(x_min.shape, device=x.device)
    norm = torch.where(x_min < 0, -x_min, c)
    return x + norm

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
