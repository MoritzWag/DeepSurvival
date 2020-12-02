import torch 
import torch.nn as nn 
import pdb

from abc import ABC, abstractmethod
from src.evaluator import Evaluator
from src.visualizer import Visualizer
from src.dsap.layers.convolution import ProbConv2dInput
from src.dsap.layers.linear import ProbLinearInput

from lpdn import convert_to_lpdn, convert_layer

class BaseModel(ABC, Evaluator, Visualizer):
    """
    """
    def __init__(self, **kwargs):
        super(BaseModel, self).__init__(**kwargs)

    @abstractmethod
    def _loss_function(self):
        pass
    
    @abstractmethod
    def _orthogonalize(self):
        pass

    def _build_lpdn_model(self):
        """
        """
        
        model_parts = tuple(self._modules.keys())
        if len(model_parts) == 1:
            print('only uni-modal model with images')
            lpdn_model = convert_to_lpdn(self.forward())
        else:
            print('multi-modal with images and tabular data')
            deep_model = self._modules[model_parts[0]]
            concat_layer = self._modules[model_parts[1]]

            # convert deep part
            lpdn_deep = convert_to_lpdn(deep_model)
            first_layer = lpdn_deep.modules_list[0]

            # remove first layer of deep part 
            lpdn_deep = lpdn_deep.modules_list[1:]
            
            # convert linear layer after concatentation
            lpdn_concat = convert_layer(concat_layer)

            # define probabilistic inputs for unstructured part
            unstructured_input_layer = ProbConv2dInput(in_channels=first_layer.in_channels,
                                                       out_channels=first_layer.out_channels,
                                                       kernel_size=first_layer.kernel_size)

            # define probabilistic inputs for structured part
            structured_input_layer = ProbLinearInput(in_features=2, out_features=2)

            # setup LP model
            lpdn_model = LPModel(deep=lpdn_deep, 
                                 concat_layer=lpdn_concat,
                                 unstructured_input_layer=unstructured_input_layer,
                                 structured_input_layer=structured_input_layer,
                                 orthogonalize=self.orthogonalize,
                                 _orthogonalize=self._orthogonalize)

            # load in weights from original model
            state_dict = self.state_dict()
            lpdn_model.load_state_dict(state_dict, strict=False)
            
        return lpdn_model 


    def accumulate_batches(self, data, cuda=False):
        """
        """
        images = []
        tabular_data = []
        events = []
        times = []
        for batch, data in enumerate(data):
            image = data[0]
            tabular_date = data[1]
            event = data[2]
            time = data[3]
            if cuda: 
                image = image.cuda()
                tabular_date = tabular_date.cuda()
            images.append(image)
            tabular_data.append(tabular_date)
            events.append(event)
            times.append(time)

        images = torch.cat(images)
        tabular_data = torch.cat(tabular_data)
        events = torch.cat(events)
        times = torch.cat(times)

        return images, tabular_data, events, times


    def save_model(self):
        pass 

    def load_model(self):
        pass

    def save_weights(self):
        pass 

    def load_weights(self):
        pass



class LPModel(nn.Module):
    """
    """
    def __init__(self, 
                 deep, 
                 concat_layer, 
                 unstructured_input_layer,
                 structured_input_layer,
                 orthogonalize,
                 _orthogonalize):
        super(LPModel, self).__init__() 
        self.deep = nn.Sequential(*deep)
        self.concat_layer = concat_layer
        self.unstructured_input_layer = unstructured_input_layer
        self.structured_input_layer = structured_input_layer
        self.orthogonalize = orthogonalize
        self._orthogonalize = orthogonalize

    def forward(self, structured, unstructured): 
        unstructured_input = self.unstructured_input_layer(unstructured)
        structured_input = self.structured_input_layer(structured)

        #  loop through inputs with mask and without mask
        (m1u, v1u), (m2u, v2u) = unstructured_input
        (m1s, v1s), (m2s, v2s) = structured_input

        # run through observations with feature i 
        m1u, v1u = self.deep((m1u, v1u))
        # concatenate with structured input 
        m1 = torch.cat((m1u, m1s), axis=1)
        v1 = torch.cat((v1u, v1s), axis=1)

        m1, v1 = self.concat_layer((m1, v1))
        mv1 = (m1, v1)
        # run through observations without feature i
        m2u, v2u = self.deep((m2u, v2u))
        # concatenate with structured input 
        m2 = torch.cat((m2u, m2s), axis=1)
        v2 = torch.cat((v2u, v2s), axis=1)

        m2, v2 = self.concat_layer((m2, v2))

        mv2 = (m2, v2)

        return mv1, mv2

        
