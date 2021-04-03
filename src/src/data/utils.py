
import numpy as np 
import pandas as pd 
import pdb 
import cv2
import torch
import threading
import multiprocessing
import math
import colorsys

from torch import nn
from torch.utils import data 
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from src.data.adni import ADNI
from src.data.sim_images import SimImages
from src.data.sim_mnist import SimMNIST


def get_dataloader(root, part, transform,
                   base_folder, data_type,
                   split, batch_size, 
                   cox_collate=True, 
                   return_data=False):
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
        if base_folder in ['mnist', 'mnist3d']:
            data = SimMNIST(root=root,
                            part=part,
                            base_folder=base_folder,
                            data_type=data_type,
                            n_dim=1)
        else:
            data = SimImages(root=root,
                             part=part,
                             base_folder=base_folder,
                             data_type=data_type,
                             n_dim=3)

    data_gen = DataLoader(dataset=data,
                          batch_size=len(data) if batch_size == -1 else batch_size,
                          collate_fn=cox_collate_fn if cox_collate else standard_collate_fn,
                          shuffle=False)

    eval_data = data.eval_data

    if return_data:
        return data_gen, eval_data, data
    else:
        return data_gen, eval_data


def get_eval_data(batch, model):
    """
    """
    prediction = model(**batch).cpu().detach().numpy()
    prediction = prediction.squeeze(1)

    try:
        image_prediction = model.predict_on_images(**batch)
        image_prediction = image_prediction.squeeze(1)
        return {'riskscores': prediction, 'riskscore_img': image_prediction}
    except:
        return {'riskscores': prediction}

def standard_collate_fn(batch, data_collate=default_collate):
    """
    """
    transposed_data = list(zip(*batch))
    data = []
    for b in transposed_data:
        bt = data_collate(b)
        data.append(bt)
    
    return {'images': data[0].float(), 'tabular_data': data[1],
            'event': data[2], 'time': data[3]}

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

def generated_colored_bs_img(data, sample_batch, indeces):
    """
    """
    df = data.df 
    h_channels =  np.asarray(df.iloc[indeces, 1])
    sample_images = sample_batch['images']
    shapes = sample_images.shape
    images = []
    for i in range(h_channels.shape[0]):
        img = np.zeros(shape=[shapes[3], shapes[2], shapes[1]], dtype='float32')
        color_rgb_norm = colorsys.hls_to_rgb(h_channels[i]/360, 50/100, 100/100)
        color_rgb = (color_rgb_norm[0]*255, color_rgb_norm[1]*255, color_rgb_norm[2]*255)
        img[:, :] = color_rgb
        img = rgb2hsl(img)
        img = np.transpose(img, axes=(2, 0, 1))
        images.append(img)

    images = np.stack(images)
    images = torch.tensor(images).to(sample_images.device)

    return images
    
def rgb2hsl(rgb):

    def core(_rgb, _hsl):

        irgb = _rgb.astype(np.uint16)
        ir, ig, ib = irgb[:, :, 0], irgb[:, :, 1], irgb[:, :, 2]
        h, s, l = _hsl[:, :, 0], _hsl[:, :, 1], _hsl[:, :, 2]

        imin, imax = irgb.min(2), irgb.max(2)
        iadd, isub = imax + imin, imax - imin

        ltop = (iadd != 510) * (iadd > 255)
        lbot = (iadd != 0) * (ltop == False)

        l[:] = iadd.astype(np.float) / 510

        fsub = isub.astype(np.float)
        s[ltop] = fsub[ltop] / (510 - iadd[ltop])
        s[lbot] = fsub[lbot] / iadd[lbot]

        not_same = imax != imin
        is_b_max = not_same * (imax == ib)
        not_same_not_b_max = not_same * (is_b_max == False)
        is_g_max = not_same_not_b_max * (imax == ig)
        is_r_max = not_same_not_b_max * (is_g_max == False) * (imax == ir)

        h[is_r_max] = ((0. + ig[is_r_max] - ib[is_r_max]) / isub[is_r_max])
        h[is_g_max] = ((0. + ib[is_g_max] - ir[is_g_max]) / isub[is_g_max]) + 2
        h[is_b_max] = ((0. + ir[is_b_max] - ig[is_b_max]) / isub[is_b_max]) + 4
        h[h < 0] += 6
        h[:] /= 6

    hsl = np.zeros(rgb.shape, dtype=np.float)
    cpus = multiprocessing.cpu_count()
    length = int(math.ceil(float(hsl.shape[0]) / cpus))
    line = 0
    threads = []
    while line < hsl.shape[0]:
        line_next = line + length
        thread = threading.Thread(target=core, args=(rgb[line:line_next], hsl[line:line_next]))
        thread.start()
        threads.append(thread)
        line = line_next

    for thread in threads:
        thread.join()

    return hsl


def hsl2rgb(hsl):

    def core(_hsl, _frgb):

        h, s, l = _hsl[:, :, 0], _hsl[:, :, 1], _hsl[:, :, 2]
        fr, fg, fb = _frgb[:, :, 0], _frgb[:, :, 1], _frgb[:, :, 2]

        q = np.zeros(l.shape, dtype=np.float)

        lbot = l < 0.5
        q[lbot] = l[lbot] * (1 + s[lbot])

        ltop = lbot == False
        l_ltop, s_ltop = l[ltop], s[ltop]
        q[ltop] = (l_ltop + s_ltop) - (l_ltop * s_ltop)

        p = 2 * l - q
        q_sub_p = q - p

        is_s_zero = s == 0
        l_is_s_zero = l[is_s_zero]
        per_3 = 1./3
        per_6 = 1./6
        two_per_3 = 2./3

        def calc_channel(channel, t):

            t[t < 0] += 1
            t[t > 1] -= 1
            t_lt_per_6 = t < per_6
            t_lt_half = (t_lt_per_6 == False) * (t < 0.5)
            t_lt_two_per_3 = (t_lt_half == False) * (t < two_per_3)
            t_mul_6 = t * 6

            channel[:] = p.copy()
            channel[t_lt_two_per_3] = p[t_lt_two_per_3] + q_sub_p[t_lt_two_per_3] * (4 - t_mul_6[t_lt_two_per_3])
            channel[t_lt_half] = q[t_lt_half].copy()
            channel[t_lt_per_6] = p[t_lt_per_6] + q_sub_p[t_lt_per_6] * t_mul_6[t_lt_per_6]
            channel[is_s_zero] = l_is_s_zero.copy()

        calc_channel(fr, h + per_3)
        calc_channel(fg, h.copy())
        calc_channel(fb, h - per_3)

    frgb = np.zeros(hsl.shape, dtype=np.float)
    cpus = multiprocessing.cpu_count()
    length = int(math.ceil(float(hsl.shape[0]) / cpus))
    line = 0
    threads = []
    while line < hsl.shape[0]:
        line_next = line + length
        thread = threading.Thread(target=core, args=(hsl[line:line_next], frgb[line:line_next]))
        thread.start()
        threads.append(thread)
        line = line_next

    for thread in threads:
        thread.join()

    return (frgb*255).round().astype(np.uint8)