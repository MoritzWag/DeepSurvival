import torch
import pdb
import numpy as np 

from torch import nn



class Evaluator(nn.Module):
    """
    """
    def __init__(self, **kwargs):
        super(Evaluator, self).__init__(**kwargs)
        self.scores = {}

    def cindex_metric(self):
        pass

    def calculate_gradients(self, images, tabular_data):
        """
        """
        gradients = []
        for scaled_input in enumerate(images):
            if isinstance(scaled_input, tuple):
                scaled_input = scaled_input[1]

            scaled_input.requires_grad = True
            tabular_data.requires_grad = True
            output = self.forward(tabular_data, scaled_input.float())
            self.zero_grad()
            output_mean = torch.mean(output)
            output_mean.backward()
            gradient = scaled_input.grad.detach().cpu().numpy()[0]
            gradients.append(gradient)
        
        gradients = np.array(gradients)
        return gradients


    def integrated_gradients(self, images, tabular_data, baseline=None, steps=50):
        """
        """
        
        if baseline is None:
            baseline = 0 * images
        scaled_inputs = [baseline + (float(i) / steps) * (images - baseline) for i in range(0, steps + 1)]
        gradients = self.calculate_gradients(images=scaled_inputs, tabular_data=tabular_data)
        avg_gradients = np.average(gradients[:-1], axis=0)
        #avg_gradients = np.transpose(avg_gradients, (1, 2, 0))
        integrated_gradient = (images - baseline) * avg_gradients

        return integrated_gradient

        


