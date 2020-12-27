
import numpy as np 
import pandas as pd 
import pdb 
import cv2
import torch
from torch.utils import data 
from torch.utils.data.dataloader import default_collate


def get_eval_data(batch, model):
    """
    """
    if len(batch.keys()) == 6:
        # make predictions for PAM with offset = 0 and splines = 0
        zero_offset = torch.zeros(batch['offset'].shape)
        batch['offset'] = zero_offset
        zero_splines = torch.zeros(batch['splines'].shape)
        batch['splines'] = zero_splines

        predictions = model(**batch).detach().numpy()

        prediction = []
        num_of_intervals = np.unique(batch['index'], return_counts=True)[1]
        for idx in range(num_of_intervals.shape[0]):
            num_obs = np.sum(num_of_intervals[0:idx+1])
            predictions_obs = predictions[num_obs -1]
            prediction.append(predictions_obs)
        
        prediction = np.stack(prediction)
    else:
        prediction = model(**batch).detach().numpy()


    return {'riskscores': prediction}




def ped_collate_fn(batch, data_collate=default_collate):
    # images
    batch = list(zip(*batch))
    data = []
    images = data_collate(batch[0])
    data.append(images)

    # tabular data
    td = np.vstack(batch[1])
    data.append(torch.from_numpy(td))
    offset = np.hstack(batch[2])
    data.append(torch.from_numpy(offset))
    ped_status = np.hstack(batch[3])
    data.append(torch.from_numpy(ped_status))
    index = np.hstack(batch[4])
    data.append(torch.from_numpy(index))
    splines = np.vstack(batch[5])
    data.append(torch.from_numpy(splines))

    return data



# def cox_collate_fn(
#     batch: List[Any], time_index: Optional[int] = -1, data_collate=default_collate
# ) -> List[torch.Tensor]:
#     """Create risk set from batch."""
#     transposed_data = list(zip(*batch))
#     y_time = np.array(transposed_data[time_index])
# ​
#     data = []
#     for b in transposed_data:
#         bt = data_collate(b)
#         data.append(bt)
# ​
#     data.append(torch.from_numpy(make_riskset(y_time)))
# ​
#     return data


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
                   gray_scale,
                   pos_random=True,
                   seed=1328):

    random = np.random.RandomState(seed)
    img = np.zeros((1, img_size, img_size), dtype='float32')

    if pos_random is False:
        upper_corner = [14, 14]
    else:
        height = random.randint(img_size, size=1)[0]
        width = random.randint(img_size-length, size=1)[0]
        upper_corner = [height, width]
    
    img[:, upper_corner[0]- length: upper_corner[0], upper_corner[1]:upper_corner[1]+length] = gray_scale

    return img 


def triangle_mask(img_size,
                  length,
                  gray_scale,
                  pos_random=True,
                  seed=1328):

    random = np.random.RandomState(seed)
    img = np.zeros((img_size, img_size, 1), dtype='float32')
    if pos_random is False:
        left_corner = [14, 14]
    else:
        left_corner = [random.randint(img_size-5, size=1)[0], random.randint(img_size-5, size=1)[0]]
    
    points = np.array([[left_corner[0], left_corner[1]],
                       [left_corner[0] + 5, left_corner[1] + 5],
                       [left_corner[0], left_corner[1] + 10]], np.int32)
    
    image = cv2.fillPoly(img, [points], color=gray_scale)

    image = image.reshape(1, img_size, img_size)

    return image


def circle_mask(img_size,
                center,
                radius,
                gray_scale,
                pos_random=True,
                seed=1328):

    random = np.random.RandomState(seed)
    img = np.zeros((img_size, img_size, 1), dtype='float32')
    if pos_random is False:
        center = center
    else:
        center = (random.randint(img_size - radius, size=1)[0], random.randint(img_size - radius, size=1)[0])
    
    img = cv2.circle(img, center, radius, gray_scale, thickness=-1)

    image = img.reshape(1, img_size, img_size)

    return image