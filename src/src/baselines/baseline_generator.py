import torch 

from torch import nn
from src.architectures.network2d import Generator2d, Discriminator2d


class BaselineGenerator(nn.Module):
    """
    """
    def __init__(self,
                 discriminator,
                 generator,
                 survival_model,
                 data_type,
                 generator_params):
        super(BaselineGenerator, self).__init__()

        self.discriminator = discriminator
        self.generator = generator
        self.survival_model = survival_model
        self.data_type = data_type
        self.generator_params = generator_params

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def build_model(self):
        pass
    
    def train(self, train_gen):
        """
        """
        
        for step in range(self.generator_params['n_steps']):
            self.discriminator.train()
            self.generator.train()

            try:
                batch = next(train_gen_iter)
            except:
                train_gen_iter = iter(train_gen)
                batch = next(train_gen_iter)

            # derive labels for batch
            label_true, label_target = self.generate_labels(batch=batch)

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

    def generate_labels(batch):
        """
        """
        prediction = self.survival_model(**batch)
        log_prediction = torch.log(prediction)

        # pseudo_label = 
        # should it be -1/1 or 0/1
        return label_true, label_target
        



        