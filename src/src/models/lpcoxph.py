import torch 
import torch.nn as nn 
import pdb 

from lpdn import convert_to_lpdn, convert_layer


class LPCoxPH(nn.Module):
    """
    """
    def __init__(self, 
                 deep, 
                 concat_layer, 
                 unstructured_input_layer,
                 structured_input_layer,
                 orthogonalize,
                 _orthogonalize):
        super(LPCoxPH, self).__init__() 
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