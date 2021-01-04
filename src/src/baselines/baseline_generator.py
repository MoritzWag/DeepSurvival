import torch 
import pdb
import time 
import datetime 

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
                 c_dim,
                 img_size,
                 generator_params,
                 discriminator_params,
                 trainer_params):
        super(BaselineGenerator, self).__init__()

        self.discriminator = discriminator
        self.generator = generator
        self.survival_model = survival_model
        self.data_type = data_type

        self.c_dim = c_dim
        self.img_size = img_size
        self.trainer_params = trainer_params

        # discriminator params
        self.d_lr = discriminator_params['learning_rate']
        self.d_beta1 = discriminator_params['beta1']
        self.d_beta2 = discriminator_params['beta2']
        self.d_repeat_num = discriminator_params['repeat_num']
        self.d_conv_dim = discriminator_params['conv_dim']

        # generator params
        self.g_lr = generator_params['learning_rate']
        self.g_beta1 = generator_params['beta1']
        self.g_beta2 = generator_params['beta2']
        self.g_repeat_num = generator_params['repeat_num']
        self.g_conv_dim = generator_params['conv_dim']


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.build_model()

    def build_model(self):
        """
        """

        # configure discriminator
        self.D = self.discriminator(self.img_size, self.d_conv_dim, self.c_dim, self.d_repeat_num)

        # configure generator 
        self.G = self.generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)

        # configure optimizer for D and G
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.d_beta1, self.d_beta2])
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.g_beta1, self.g_beta2])

        for p in self.survival_model.parameters():
            p.requires_grad = False

        # put them on the device
        self.G.to(self.device).float()
        self.D.to(self.device).float()

    def train(self, train_gen):
        """
        """
        print("start training")
        start_time = time.time()
        for step in range(self.trainer_params['n_steps']):
            self.D.train()
            self.G.train()

            try:
                batch = next(train_gen_iter)
            except:
                train_gen_iter = iter(train_gen)
                batch = next(train_gen_iter)

            # derive labels for batch
            label_target, label_origin = self.generate_labels(batch=batch)
            

            ############################################################
            #
            # 1.) Train the Discriminator
            #
            ############################################################
            
            # Compute loss with real images
            out_src = self.D(batch['images'])
            d_loss_real =  - torch.mean(out_src)
            

            # Compute loss with fake images
            x_fake = self.G(batch['images'], label_target.float())
            out_src = self.D(x_fake)
            d_loss_fake = torch.mean(out_src)
            out_domain = self.survival_model.predict_on_images(x_fake)
            
            # Backward and optimize
            d_loss = d_loss_real + d_loss_fake 
            d_loss.backward()
            self.d_optimizer.step()

            # Logging
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()


            ############################################################
            #
            # 2.) Train the Generator
            #
            ############################################################

            if (step + 1) % self.trainer_params['n_critic'] == 0:
                # Original-to-target domain 
                x_fake = self.G(batch['images'], label_target.float())
                out_src = self.D(x_fake)
                out_domain = self.survival_model.predict_on_images(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = self.domain_loss(out_domain, label_target)

                # Target-to-original domain
                x_reconstruct = self.G(x_fake, label_origin)
                g_loss_rec = self.reconstruction_loss(batch['images'], x_reconstruct)

                # Backward and optimize
                g_loss = g_loss_fake + self.trainer_params['lambda_rec'] * g_loss_rec + self.trainer_params['lambda_cls'] * g_loss_cls
                g_loss.backward()
                self.g_optimizer.step()

                # Logging
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()
            

            # print out training info
            if (step  + 1) % self.trainer_params['log_step'] == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, step+1, self.trainer_params['n_steps'])
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)
    
    def validate(self, val_gen):
        """
        """
        pass

    def test(self, test_gen):
        """
        """
        pass

    def domain_loss(self, preds, target_label, alpha=0.25, tolerance=0.5):
        """Adaptive quantile loss.
        """

        errors = torch.zeros(size=preds.shape)
        losses = []
        for idx in range(preds.shape[0]):
            if target_label[idx, 1] == 1.:
                tolerance = -1 * tolerance
            errors[idx] = torch.abs(preds[idx] - tolerance)
            if target_label[idx, 1] == 1.:
                if preds[idx] > tolerance:
                    loss = alpha * errors[idx]
                else:
                    loss = (1 - alpha) * errors[idx]
            else:
                if preds[idx] > tolerance:
                    try:
                        loss = (1 - alpha) * errors[idx]
                    except:
                        pdb.set_trace()
                else:
                    loss = alpha * errors[idx]
            losses.append(loss)

        loss = torch.mean(torch.stack(losses))
        
        return loss

    def reconstruction_loss(self, x_real, x_reconstruct):
        """Cyclic reconstruction loss between reconstructed 
        to original domain from generated images from target domain.
        """

        loss = torch.mean(torch.abs(x_real - x_reconstruct))

        return loss

    def generate_labels(self, batch):
        """
        """

        prediction = self.survival_model.predict_on_images(images=batch['images'])
        log_prediction = torch.log(prediction).squeeze(1)
        
        label_target = torch.zeros(size=(log_prediction.shape[0], 2))
        label_origin = torch.zeros(size=(log_prediction.shape[0], 2))
        for idx in range(label_target.shape[0]):
            if log_prediction[idx] > 0: 
                label_target[idx, 1] = 1.0
                label_origin[idx, 0] = 1.0 
            else:
                label_target[idx, 0] = 1.0 
                label_origin[idx, 1] = 1.0
    
        return label_target.float(), label_origin.float()
        



        