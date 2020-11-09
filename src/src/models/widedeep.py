import torch 
import torch.nn as nn 


class Wide(nn.Module):
    """
    """
    def __init__(self, 
                 wide_dim: int,
                 output_dim: int):
        super(Wide, self).__init_()
        
        self.wlinear = nn.Linear(wide_dim, output_dim)

    def forward(self, x):
        out = self.wlinear(x)
        return out



class Deep(nn.Module):
    """
    """
    def __init__(self, 
                 ouput_dim:int=1,
                 pretrained:bool=True):
        super(Deep, self).__init__()
    
    def forward(self, x):
        pass 



class WideDeep(nn.Module):
    """
    """
    def __init__(self,
                 output_dim:int, 
                 **params:Any):
        super(WideDeep, self).__init_()

        self.output_dim = output_dim

        for k, v in params['wide'].items():
            setattr(self, k, v)
        self.wide = Wide(self.wide_dim, self.output_dim)

        for k, v in params['deep'].items():
            setattr(self, k, v)
        self.deep = Deep(self.output_dim,
                         self.pretrained)

    def pass_one_batch(self, batch, loss_function):
        pass 



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
