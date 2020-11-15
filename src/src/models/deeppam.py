
import torch 

from torch import nn
from src.models.base import BaseModel



class DeepPAM(BaseModel):
    """
    """
    def __init__(self,
                 deep: nn.Module, 
                 output_dim: int,
                 orthogonalize: bool,
                 **params):
        super(DeepPAM, self).__init__()

        self.wide = wide
        self.deep = deep 
        self.output_dim = output_dim
        self.orthogonalize = orthogonalize
    
    def _orthogonalize(self, structured, unstructured):
        pass 

    def forward(self, structured, unstructured):
        pass

    def _loss_function(self):
        pass


    # def forward(self, 
    #             structured, 
    #             unstructured):
    #     structured_out = self.wide(structured)
    #     unstructured_out = self.deep(unstructured)

    #     if self.orthogonalize:
    #         projection_matrix = gram_schmidt(structured_out)
    #         unstructured_orthogonalized = orthogonalize(projection_matrix, unstructured_out)
    #         features_concatenated = torch.concatenate((structured_out, unstructured_orthogonalized))
    #     else:
    #         features_concatenated = torch.concatenate((structured_out, unstructured_out))
        
    #     # linear layer
    #     out = self.linear(features_concatenated)
    #     return out




def gram_schmidt(vv):
    """
    """
    def projection(u, v):
        return (v * u).sum() / (u * u).sum() * u

    nk = vv.size(0)
    uu = torch.zeros_like(vv, device=vv.device)
    uu[:, 0] = vv[:, 0].clone()
    for k in range(1, nk):
        vk = vv[k].clone()
        uk = 0
        for j in range(0, k):
            uj = uu[:, j].clone()
            uk = uk + projection(uj, vk)
        uu[:, k] = vk - uk
    for k in range(nk):
        uk = uu[:, k].clone()
        uu[:, k] = uk / uk.norm()
    return uu


def orthogonalize(projection_matrix, feature_matrix):
    """
    """
    n = projection_matrix.dim()[0]
    identity = torch.eye(n)

    orthogonalized_matrix = identity - projection_matrix
    orthogonalized_features = orthogonalized_matrix * feature_matrix

    return orthogonalized_features