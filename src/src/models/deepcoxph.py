
import torch 
import pdb

from torch import nn
from src.models.base import BaseModel



class DeepCoxPH(BaseModel):
    """
    """
    def __init__(self,
                 deep: nn.Module, 
                 structured_input_dim: int, 
                 output_dim: int,
                 orthogonalize: bool,
                 **kwargs):
        super(DeepCoxPH, self).__init__(**kwargs)

        self.deep = deep 
        self.output_dim = output_dim
        self.structured_input_dim = structured_input_dim
        self.orthogonalize = orthogonalize
        self.overall_dim = self.structured_input_dim + self.deep.fc3.out_features
        self.linear = nn.Linear(in_features=self.overall_dim, out_features=self.output_dim)

    def _orthogonalize(self, structured, unstructured):
        """
        """
        projection_matrix = calculate_projection_matrix(structured)
        unstructured_orthogonalized = orthogonalize(projection_matrix, unstructured)
        
        return unstructured_orthogonalized

    def forward(self, structured, unstructured):

        unstructured = self.deep(unstructured)
        if self._orthogonalize:
            unstructured_orth = self._orthogonalize(structured, unstructured)
            features_concatenated = torch.cat((structured, unstructured_orth), axis=1)
        else:
            features_concatenated = torch.cat((structured, unstructured), axis=1)
        
        out = self.linear(features_concatenated.float())
        return out

    def _loss_function(self, event, riskset, predictions):
        """
        """

        pred_t = predictions.t()

        rr = self.logsumexp_masked(pred_t, riskset, axis=1, keepdim=True)

        losses = torch.multiply(event, rr - predictions)
        loss = torch.mean(losses)

        return {'loss': loss}


    def logsumexp_masked(self, risk_scores, mask, axis=0, keepdim=None):
        """
        """

        mask = torch.from_numpy(mask)
        risk_scores_masked = torch.multiply(risk_scores, mask)
        amax = torch.max(risk_scores_masked, dim=axis, keepdim=True)
        risk_scores_shift = risk_scores_masked - amax[0]

        exp_masked = torch.multiply(torch.exp(risk_scores_shift), mask)
        exp_sum = torch.sum(exp_masked, axis=axis, keepdim=True)
        output = amax[0] + torch.log(exp_sum)
        if not keepdim:
            output = torch.squeeze(output, axis=axis)
        return output



def calculate_projection_matrix(matrix):
    """To calculate the projection matrix of X, the following needs to be 
    calculated: P = x*(xTx)^-1*xT. This can be achieved by applying the gram schmidt algorithm

    Args:
        matrix: {torch.Tensor} matrix for which the projection matrix needs to be identified. 
    
    Returns:
        projection_matrix: {torch.Tensor} projection matrix
    """

    Q, R = torch.qr(matrix)
    xTx_xT = torch.matmul(torch.inverse(R), Q.t())
    projection_matrix = torch.matmul(matrix, xTx_xT)

    return projection_matrix

def orthogonalize(projection_matrix, feature_matrix):
    """
    """
    num_obs = projection_matrix.shape[0]
    identity = torch.eye(num_obs)
    orthogonalized_matrix = identity - projection_matrix
    orthogonalized_features = torch.matmul(orthogonalized_matrix.float(), feature_matrix)

    return orthogonalized_features




# def gram_schmidt(vv):
#     """
#     """
#     def projection(u, v):
#         return (v * u).sum() / (u * u).sum() * u
#     pdb.set_trace()
#     nk = vv.size(0)
#     uu = torch.zeros_like(vv, device=vv.device)
#     uu[:, 0] = vv[:, 0].clone()
#     for k in range(1, nk):
#         vk = vv[k,:].clone()
#         uk = 0
#         for j in range(0, k):
#             uj = uu[:, j].clone()
#             uk = uk + projection(uj, vk)
#         uu[:, k] = vk - uk
#     for k in range(nk):
#         uk = uu[:, k].clone()
#         uu[:, k] = uk / uk.norm()
#     return uu


# def orthogonalize(projection_matrix, feature_matrix):
#     """
#     """
#     pdb.set_trace()
#     n = projection_matrix.dim()[0]
#     identity = torch.eye(n)

#     orthogonalized_matrix = identity - projection_matrix
#     orthogonalized_features = orthogonalized_matrix * feature_matrix

#     return orthogonalized_features



# def gram_schmidt(matrix):
#     """
#     """

#     og_vectors = []
#     for i in range(matrix.shape[1]):
#         v = matrix[:, i].clone()

#         for u in og_vectors:
#             v -= projection(u=u, z=matrix[:, i])
#         og_vectors.append(v)
    
#     og_vectors = torch.stack(og_vectors, dim=1)
#     return og_vectors


# # proj = (torch.dot(u, self._v)
# #         / torch.dot(self._v, self._v)) * self._v

# def projection(z, u):
#     """
#     """
    
#     if u.shape[0] != z.shape[0]:
#         raise ValueError('Dimension of u does not match dimension of v')

#     proj = (torch.dot(z, u) / torch.dot(u, u)) * u

#     return proj
    
