
import torch 
import pdb
import numpy as np 
import math

from torch import nn
from src.models.base import BaseModel
from src.data.utils import safe_normalize


class DeepCoxPH(BaseModel):
    """
    """
    def __init__(self,
                 deep: nn.Module, 
                 structured_input_dim: int, 
                 out_dim: int, 
                 output_dim: int,
                 orthogonalize: bool,
                 **kwargs):
        super(DeepCoxPH, self).__init__()

        self.deep = deep 
        self.output_dim = output_dim
        self.structured_input_dim = structured_input_dim
        self.orthogonalize = orthogonalize
        self.overall_dim = self.structured_input_dim + self.deep.out_dim
        self.linear = nn.Linear(in_features=self.overall_dim, out_features=self.output_dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tanh = nn.Tanh()

    def forward(self, tabular_data, images, **kwargs):
        
        unstructured = self.deep(images)
        if self.orthogonalize:
            unstructured_orth = self._orthogonalize(tabular_data, unstructured)
            features_concatenated = torch.cat((tabular_data, unstructured_orth), axis=1)
        else:
            features_concatenated = torch.cat((tabular_data, unstructured), axis=1)
        
        # riskscore = self.linear(features_concatenated.float())
        riskscore = self.tanh(self.linear(features_concatenated.float()))
        
        return riskscore

    def predict_on_images(self, images, tabular_data, **kwargs):
        """
        """
        unstructured = self.deep(images)
        if self.orthogonalize:
            unstructured_orth = self._orthogonalize(tabular_data, unstructured)
        else:
            unstructured_orth = unstructured
        
        weights = self.linear.weight.data
        unstructured_weights = weights[:, 0:unstructured_orth.shape[1]]
        out = unstructured_orth.matmul(unstructured_weights.t())

        return out

    def calculate_projection_matrix(self, matrix):
        """
        """
        XtX = torch.mm(matrix.t(), matrix)
        dmat = torch.eye(XtX.shape[0]).to(self.device) * torch.diagonal(XtX)
        CtC = XtX + dmat * 1e-09

        CtC = self.make_psd(CtC)
        
        R = torch.cholesky(CtC, upper=True)

        # calculate R-TDR-1
        RinvTD = torch.matmul(torch.inverse(R).T, dmat)
        RinvTDRinv = torch.matmul(RinvTD, torch.inverse(R))

        # singular value decompostion
        U, diags, _ = torch.svd(RinvTDRinv)

        # A = CR-1U
        CR_inv = torch.matmul(matrix, torch.inverse(R))
        A = torch.matmul(CR_inv, U)
        AtA = torch.matmul(A.T, A) + 1e-09 * diags
        AAtA = torch.matmul(A, AtA)
        pm = torch.matmul(AAtA, A.T)

        return pm 

    def make_psd(self, x):
        """
        """

        lamb = torch.min(torch.eig(x, eigenvectors=True)[0])
        lamb = lamb - math.sqrt(1e-9)
        # if smallest eigenvalue is negative => not semipositive definite
        if lamb < -1e-10:
            rho = 1 / (1 - lamb)
            x = rho * x + (1 - rho) * torch.eye(x.shape[0]).to(self.device)
            
        return x

    def orthogonalization(self, projection_matrix, feature_matrix):
        """
        """
        num_obs = projection_matrix.shape[0]
        identity = torch.eye(num_obs).to(self.device)
        orthogonalized_matrix = identity - projection_matrix
        orthogonalized_features = torch.matmul(orthogonalized_matrix.float(), feature_matrix.to(self.device))
        return orthogonalized_features
    

    def _orthogonalize(self, structured, unstructured):
        """orthogonalize unstructured latent representation of unstructured data
        on structured data
        """
        projection_matrix = self.calculate_projection_matrix(structured)
        unstructured_orthogonalized =  self.orthogonalization(projection_matrix, unstructured)
        
        return unstructured_orthogonalized