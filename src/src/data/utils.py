
import numpy as np 
import pandas as pd 
import pdb 
import cv2
import torch

from torch.utils import data 
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from src.data.adni import ADNI 


def get_dataloader(root, part, transform,
                   base_folder, data_type, batch_size):
    """
    """
    data = ADNI(root=root, 
                part=part,
                transform=transform,
                base_folder=base_folder,
                data_type=data_type)
    data_gen = DataLoader(dataset=data,
                          batch_size=batch_size,
                          shuffle=False)

    return data_gen


def get_eval_data(batch, model):
    """
    """
    if len(batch.keys()) == 6:
        # make predictions for PAM with offset = 0 and splines = 0
        zero_offset = torch.zeros(batch['offset'].shape)
        batch['offset'] = zero_offset
        zero_splines = torch.zeros(batch['splines'].shape)
        batch['splines'] = zero_splines

        predictions = model(**batch).cpu().detach().numpy()

        prediction = []
        num_of_intervals = np.unique(batch['index'].cpu().detach().numpy(), return_counts=True)[1]
        for idx in range(num_of_intervals.shape[0]):
            num_obs = np.sum(num_of_intervals[0:idx+1])
            predictions_obs = predictions[num_obs -1]
            prediction.append(predictions_obs)
        
        prediction = np.stack(prediction)
    else:
        prediction = model(**batch).cpu().detach().numpy()
        prediction = prediction.squeeze(1)
    
    try:
        image_prediction = model.predict_on_images(**batch)
        image_prediction = image_prediction.squeeze(1)
        return {'riskscores': prediction, 'riskscore_img': image_prediction}
    except:
        return {'riskscores': prediction}


def ped_collate_fn(batch, data_collate=default_collate):
    # images
    #pdb.set_trace()
    batch = list(zip(*batch))
    data = []
    images = data_collate(batch[0])
    data.append(images)

    # tabular data
    td = torch.vstack(batch[1])
    data.append(td)
    offset = torch.hstack(batch[2])
    data.append(offset)
    ped_status = torch.hstack(batch[3])
    data.append(ped_status)
    index = torch.hstack(batch[4])
    data.append(index)
    splines = torch.vstack(batch[5])
    data.append(splines)

    return {'images': data[0].float(), 'tabular_data': data[1],
            'offset': data[2], 'ped_status': data[3], 
            'index': data[4], 'splines': data[5]}


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