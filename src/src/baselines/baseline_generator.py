import torch 

from torch import nn
from src.architectures.network2d import Generator2d, Discriminator2d


class BaselineGenerator(nn.Module):
    """
    """
    def __init__(self,
                 discriminator,
                 generator,
                 survival_model):
        super(BaselineGenerator, self).__init__()

        self.discriminator = discriminator
        self.generator = generator
        self.survival_model = survival_model


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def build_model(self):
        pass
    
    def train(self):


        # train discriminators
        out_src = self.discriminator(x_real)
        out_domain = self.survival_model(x_fake)

        # train generator

    
    def validate(self):
        pass

    def domain_loss(self):
        pass 

    def adversarial_loss(self):
        pass 

    def reconstruction_loss(self, x_real, x_reconstruct):

        loss = torch.mean(torch.abs(x_real - x_reconstruct))

        return loss

        