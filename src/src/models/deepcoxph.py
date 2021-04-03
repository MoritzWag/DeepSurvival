
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

    # def calculate_projection_matrix(self, matrix):
    #     """To calculate the projection matrix of X, the following needs to be 
    #     calculated: P = x*(xTx)^-1*xT. This can be achieved by applying the gram schmidt algorithm

    #     Args:
    #         matrix: {torch.Tensor} matrix for which the projection matrix needs to be identified. 
        
    #     Returns:
    #         projection_matrix: {torch.Tensor} projection matrix
    #     """
    #     Q, R = torch.qr(matrix)
    #     xTx_xT = torch.matmul(torch.inverse(R), Q.t())
    #     projection_matrix = torch.matmul(matrix, xTx_xT)

    #     pdb.set_trace()
        #return projection_matrix.to(self.device)

    # def calculate_projection_matrix(self, matrix):
    #     """ H = X(XtX)-1X = QR(RTR)−1RTQT
    #     """
    #     pdb.set_trace()
    #     Q, R = torch.qr(matrix)
    #     QR = torch.matmul(Q, R)
    #     RTR_inv = torch.inverse(torch.matmul(R.T, R))
    #     RTQT = torch.matmul(R.T, Q.T)
    #     QRRTR = torch.matmul(QR, RTR_inv)
    #     proj_mat = torch.matmul(QRRTR, RTQT)

    #     return proj_mat

    def calculate_projection_matrix(self, matrix):
        """
        """
        XtX = torch.mm(matrix.t(), matrix)
        dmat = torch.eye(XtX.shape[0]).to(self.device) * torch.diagonal(XtX)
        CtC = XtX + dmat * 1e-09

        CtC = self.make_psd(CtC)
        #pdb.set_trace()
        
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
        #pdb.set_trace()


    
    # def calculate_projection_matrix(self, matrix):
    #     """Demmler-Reinsch Orthogonalization 
    #     (cf. Ruppert er al. 2003, Semiparametric Regression, Appendix B.1.1)
    #     """

    #     XtX = torch.mm(matrix.t(), matrix)
    #     dmat = torch.eye(XtX.shape[0]).to(self.device) * torch.diagonal(XtX)
    #     CtC = XtX + dmat * 1e-09

    #     CtC = self.make_psd(CtC)

    #     pdb.set_trace()
    #     # Rm = torch.solve(torch.cholesky(A), torch.diag(XtX.dim(1)))
    #     chol = torch.cholesky(A)
    #     eye = torch.eye(XtX.shape[1]).to(self.device)
    #     Rm, _ = torch.solve(chol, eye)
    #     #Rm = torch.solve(torch.cholesky(A), torch.eye(XtX.shape[1]).to(self.device))
    #     dec = torch.svd(torch.matmul(torch.mm(Rm, dmat), Rm))


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