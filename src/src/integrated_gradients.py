import os
import torch
import pdb
import numpy as np 
from tqdm import tqdm

from torch import nn
from torchvision.utils import save_image
from typing import List, Tuple

class IntegratedGradients(nn.Module):
    """
    """
    def __init__(self, **kwargs):
        super(IntegratedGradients, self).__init__(**kwargs)

    def calculate_gradients(self, model, images, tabular_data):
        """
        """
        gradients = []
        for scaled_input in enumerate(images):
            if isinstance(scaled_input, tuple):
                scaled_input = scaled_input[1]

            if scaled_input.ndim == 3:
                scaled_input = scaled_input.unsqueeze(0)
            scaled_input.requires_grad = True
            tabular_data.requires_grad = True
            output = model.forward(tabular_data, scaled_input.float())
            model.zero_grad()
            output_mean = torch.mean(output)
            output_mean.backward()
            gradient = scaled_input.grad.detach().cpu().numpy()[0]
            gradients.append(gradient)
        
        gradients = np.array(gradients)
        return gradients

    def integrated_gradients(self, 
                             model, 
                             images, 
                             tabular_data,
                             storage_path, 
                             run_name,
                             baseline=None, 
                             length=9):
        """
        """
        if baseline is None:
            baseline = 0 * images
        scaled_inputs = [baseline + (float(i) / length) * (images - baseline) for i in range(0, length + 1)]
        gradients = self.calculate_gradients(model=model, images=scaled_inputs, tabular_data=tabular_data)
        avg_gradients = np.average(gradients[:-1], axis=0)
        integrated_gradient = (images - baseline) * avg_gradients
        # scaled_inputs = [torch.Tensor(np.stack(im_channels.detach().numpy(), axis=0)) for im_channels in scaled_inputs]
        # pdb.set_trace()
        # storage_path = os.path.expanduser(storage_path)
        # storage_path = f"{storage_path}/{run_name}"
        # if not os.path.exists(storage_path):
        #     os.makedirs(storage_path)
        # save_image(scaled_inputs, f"{storage_path}/euclidean_morph.png", nrow=length, normalize=False, range=(0,1))

        return integrated_gradient

    def wasserstein_integrated_gradients(self, 
                                         model, 
                                         images, 
                                         img2,
                                         length, 
                                         tabular_data,
                                         storage_path,
                                         run_name,
                                         baseline=None, 
                                         steps=50,
                                         epsilon=0.5):
        """
        """
        if img2 is None:
            baseline_image = images[0] * 0
        baseline_image = img2.unsqueeze(0)
        list_outs_OT = []
        for i in range(images.shape[0]):
            img1 = images[i, :, :, :]
            img_size = img1.shape[1:]
            img1, img2 = (I.reshape(1, -1, 1) for I in (img1, img2))
            
            C = generate_metric(img_size)
            Q = np.concatenate([img1, img2], axis=-1)
            Q, max_val, Q_counts = preprocess_Q(Q)
            out_OT = []

            print('Computing transportation plan...')
            for dim in range(1):
                out_OT.append([])
                P = sinkhorn(Q[dim, :, 0], Q[dim, :, 1], C, img_size[0], img_size[1], epsilon)
                for t in tqdm(np.linspace(0,1, length)):
                    out_OT[-1].append(max_val - generate_interpolation(img_size[0],img_size[1],P,t)*((1-t)*Q_counts[dim,0,0] + t*Q_counts[dim,0,1]))

            list_out_OT = [torch.Tensor(np.stack(im_channels, axis=0)) for im_channels in zip(*out_OT)]
            list_outs_OT.append(list_out_OT)
            out_OT = torch.stack([torch.Tensor(im).reshape(1,*img_size) for im in list_out_OT])

        scaled_inputs = []
        for l in range(length):
            scaled_input = torch.stack([torch.Tensor(im[l]).reshape(1, *img_size) for im in list_outs_OT])
            scaled_inputs.append(scaled_input)


        gradients = self.calculate_gradients(model=model, images=scaled_inputs, tabular_data=tabular_data)
        avg_gradients = np.average(gradients[:-1], axis=0)
        images_basline = torch.cat(images.shape[0]*[baseline_image])
        integrated_gradient = (images - images_basline) * avg_gradients

        storage_path = os.path.expanduser(storage_path)
        storage_path = f"{storage_path}/{run_name}"
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)


        save_image(list_out_OT, f"{storage_path}/wasserstein_morph.png", nrow=length, normalize=False, range=(0,1))

        return integrated_gradient

        
def _generate_metric(height, width, grid):
    """
    """
    C = np.zeros((height*width, height*width))
    i = 0
    j = 0
    for y1 in range(width):
        for x1 in range(height):
            for y2 in range(width):
                for x2 in range(height):
                    C[i, j] = np.square(grid[x1, y1, :] - grid[x2, y2, :]).sum()
                    j += 1
            j = 0
            i +=1
    
    return C


def generate_metric(im_size: Tuple[int]) -> np.ndarray:
    """
    Computes the Euclidean distances matrix
    
    Arguments:
        im_size {Tuple[int]} -- Size of the input image (height, width)
    
    Returns:
        np.ndarray -- distances matrix
    """
    grid = np.meshgrid(*[range(x) for x in im_size])
    grid = np.stack(grid,-1)
    return _generate_metric(im_size[0], im_size[1], grid)


def sinkhorn(a: np.ndarray, b: np.ndarray, C: np.ndarray, height: int, width: int, 
             epsilon: float, threshold: float=1e-7) -> np.ndarray:
    """Computes the sinkhorn algorithm naively, using the CPU.
    
    Arguments:
        a {np.ndarray} -- the first distribution (image), normalized, and shaped to a vector of size height*width.
        b {np.ndarray} -- the second distribution (image), normalized, and shaped to a vector of size height*width.
        C {np.ndarray} -- the distances matrix
        height {int} -- image height
        width {int} -- image width
        epsilon {float} -- entropic regularization parameter
    
    Keyword Arguments:
        threshold {float} -- convergence threshold  (default: {1e-7})
    
    Returns:
        np.ndarray -- the entropic regularized transportation plan, pushing distribution a to b.
    """
    K = np.exp(-C/epsilon)
    v = np.random.randn(*a.shape)
    i = 0
    while True:
        u = a/(K.dot(v))
        v = b/(K.T.dot(u))
        i += 1
        if i % 50 == 0:
            convergence = np.square(np.sum(u.reshape(-1, 1) * K * v.reshape(1,-1), axis=1) - a).sum()
            if convergence < threshold:
                print(f"Iteration {i}. Sinkhorn convergence: {convergence:.2E} (Converged!)")
                break
            else:
                print(f"Iteration {i}. Sinkhorn convergence: {convergence:.2E} ( > {threshold})")

    P = u.reshape(-1, 1) * K * v.reshape(1,-1)
    P = P.reshape(height, width, height, width)
    return P


def generate_interpolation(height, width, plan, t):
    c = np.zeros((height+1, width+1))
    for y1 in range(width):
        for x1 in range(height):
            for y2 in range(width):
                for x2 in range(height):
                    new_loc_x = (1-t)*x1 + t*x2
                    new_loc_y = (1-t)*y1 + t*y2
                    p = new_loc_x - int(new_loc_x)
                    q = new_loc_y - int(new_loc_y)
                    c[int(new_loc_x),int(new_loc_y)] += (1-p)*(1-q)*plan[x1,y1,x2,y2]
                    c[int(new_loc_x)+1,int(new_loc_y)] += p*(1-q)*plan[x1,y1,x2,y2]
                    c[int(new_loc_x),int(new_loc_y)+1] += (1-p)*q*plan[x1,y1,x2,y2]
                    c[int(new_loc_x)+1,int(new_loc_y)+1] += p*q*plan[x1,y1,x2,y2]
    c = c[:height,:width] #* (I1_count*(1-t) + I2_count*t)
    return c


def preprocess_Q(Q: np.ndarray, max_val: float=None, Q_counts: np.ndarray=None) -> Tuple[np.ndarray, float, np.ndarray]:
    """ Preprocess (normalize) input images before computing their barycenters
    
    Arguments:
        Q {np.ndarray} -- Input images. Every image should reshaped to a column in Q.
    
    Keyword Arguments:
        max_val {float} -- The maximum value. Should be changed from None when using the iterative algorithm (more than 1 iteration in the Algorithm) (default: {None})
        Q_counts {np.ndarray} -- The sum of all the pixel values in each image. Should be changed from None when using the iterative algorithm (more than 1 iteration in the Algorithm) (default: {None})
    
    Returns:
        Tuple[np.ndarray, float, np.ndarray] -- The normalized images the total maximum value and sum of pixels in each image
    """
    if max_val is None:
        max_val = Q.max()
    Q = max_val - Q
    if Q_counts is None:
        Q_counts = np.sum(Q, axis=1, keepdims=True)
    Q = Q / Q_counts
    return Q, max_val, Q_counts