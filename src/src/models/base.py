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
    
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @abstractmethod
    def _loss_function(self):
        pass
    
    @abstractmethod
    def _orthogonalize(self):
        pass
    
    @abstractmethod
    def _accumulate_batches(self):
        pass
    
    @abstractmethod
    def predict_on_images(self):
        pass

    def _orthogonalize(self, structured, unstructured):
        """orthogonalize unstructured latent representation of unstructured data
        on structured data
        """
        projection_matrix = self.calculate_projection_matrix(structured)
        unstructured_orthogonalized =  self.orthogonalization(projection_matrix, unstructured)
        
        return unstructured_orthogonalized
    
    def calculate_projection_matrix(self, matrix):
        """To calculate the projection matrix of X, the following needs to be 
        calculated: P = x*(xTx)^-1*xT. This can be achieved by applying the gram schmidt algorithm

        Args:
            matrix: {torch.Tensor} matrix for which the projection matrix needs to be identified. 
        
        Returns:
            projection_matrix: {torch.Tensor} projection matrix
        """
        try:
            Q, R = torch.qr(matrix)
            xTx_xT = torch.matmul(torch.inverse(R), Q.t())
            projection_matrix = torch.matmul(matrix, xTx_xT)
        except:
            pdb.set_trace()

        return projection_matrix.to(self.device)

    def orthogonalization(self, projection_matrix, feature_matrix):
        """
        """
        num_obs = projection_matrix.shape[0]
        identity = torch.eye(num_obs).to(self.device)
        orthogonalized_matrix = identity - projection_matrix
        orthogonalized_features = torch.matmul(orthogonalized_matrix.float(), feature_matrix.to(self.device))
        return orthogonalized_features

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

    def forward(self, structured, unstructured, baselines): 
        pdb.set_trace()
        unstructured_input = self.unstructured_input_layer(unstructured, baselines)
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

        
