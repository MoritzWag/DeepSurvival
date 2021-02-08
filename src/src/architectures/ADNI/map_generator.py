import torch
import pdb
import torch.nn as nn

from src.architectures.ADNI.utils import (ACTIVATION,
                                          deconv2d_bn_block,
                                          conv2d_bn_block,
                                          crop_and_concat,
                                          conv2d_block,
                                          conv3d_block,
                                          conv2d_bn_block,
                                          Identity)


class UNet(nn.Module):
    '''
    portings for the models found in the reference reference's official repo
    https://github.com/baumgach/vagan-code
    '''
    def __init__(self, 
                 n_dim=1, 
                 conv_dim=16, 
                 batch_norm=True, 
                 dimensions=2, 
                 c_dim=2,
                 activation=ACTIVATION,
                 **kwargs):
        super(UNet, self).__init__()
        if dimensions == 2:
            conv_block = conv2d_bn_block if batch_norm else conv2d_block
        else:
            conv_block = conv3d_block
        max_pool = nn.MaxPool2d(2) if int(dimensions) is 2 else nn.MaxPool3d(2)
        act = activation

        self.down0 = nn.Sequential(
            conv_block(n_dim+c_dim, conv_dim, activation=act),
            conv_block(conv_dim, conv_dim, activation=act)
        )
        self.down1 = nn.Sequential(
            max_pool,

            conv_block(conv_dim, 2*conv_dim, activation=act),
            conv_block(2*conv_dim, 2*conv_dim, activation=act),
        )
        self.down2 = nn.Sequential(
            max_pool,
            conv_block(2*conv_dim, 4*conv_dim, activation=act),
            conv_block(4*conv_dim, 4*conv_dim, activation=act),
        )
        self.down3 = nn.Sequential(
            max_pool,
            conv_block(4*conv_dim, 8*conv_dim, activation=act),
            conv_block(8*conv_dim, 8*conv_dim, activation=act),
        )

        self.up3 = deconv2d_bn_block(8*conv_dim, 4*conv_dim, activation=act)

        self.conv5 = nn.Sequential(
            conv_block(8*conv_dim, 4*conv_dim, activation=act),
            conv_block(4*conv_dim, 4*conv_dim, activation=act),
        )
        self.up2 = deconv2d_bn_block(4*conv_dim, 2*conv_dim, activation=act)

        self.conv6 = nn.Sequential(
            conv_block(4*conv_dim, 2*conv_dim, activation=act),
            conv_block(2*conv_dim, 2*conv_dim, activation=act),
        )
        self.up1 = deconv2d_bn_block(2*conv_dim, conv_dim, activation=act)

        self.conv7 = nn.Sequential(
            conv_block(2*conv_dim, conv_dim, activation=act),
            conv_block(conv_dim, n_dim, activation=Identity),
        )

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(10, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 12 * 12, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    
    def forward(self, x, c):
        
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)

        x0 = self.down0(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        xu3 = self.up3(x3)
        cat3 = crop_and_concat(xu3, x2)
        x5 = self.conv5(cat3)
        xu2 = self.up2(x5)
        cat2 = crop_and_concat(xu2, x1)
        x6 = self.conv6(cat2)
        xu1 = self.up1(x6)
        cat1 = crop_and_concat(xu1, x0)
        x7 = self.conv7(cat1)

        xs = self.localization(x7)
        xs = xs.view(-1, 10 * 12 * 12)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        return x7, theta                     