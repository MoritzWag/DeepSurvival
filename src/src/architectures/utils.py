
import pdb 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.nn.utils import spectral_norm


class Identity(nn.Module):
    def __init__(self, inplace=False):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

ACTIVATION = nn.ReLU


def crop_and_concat(upsampled, bypass, crop=False):
    if crop:
        c = (bypass.size()[2] - upsampled.size()[2]) // 2
        bypass = F.pad(bypass, (-c, -c, -c, -c))
    return torch.cat((upsampled, bypass), 1)

def conv2d_bn_block(in_channels, out_channels, kernel=3, stride=1, padding=1, momentum=0.01, activation=ACTIVATION):
    '''
    returns a block conv-bn-activation
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride, padding=padding),
        activation(),
        nn.BatchNorm2d(out_channels, momentum=momentum),
    )

def deconv2d_bn_block(in_channels, out_channels, use_upsample=False, kernel=4, stride=2, padding=1, momentum=0.01, activation=ACTIVATION):
    '''
    returns a block deconv-bn-activation
    NB: use_upsample = True helps to remove chessboard artifacts:
    https://distill.pub/2016/deconv-checkerboard/
    '''
    if use_upsample:
        up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        )
    else:
        up = nn.ConvTranspose2d(in_channels, out_channels, kernel, stride=stride, padding=padding)
    return nn.Sequential(
        up,
        nn.BatchNorm2d(out_channels, momentum=momentum),
        activation(),
    )


def dense_layer_bn(in_dim, out_dim, momentum=0.01, activation=ACTIVATION):
    '''
    returns a block linear-bn-activation
    '''
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim, momentum=momentum),
        activation()
    )

def conv3d_bn_block(in_channels, out_channels, kernel=3, stride=1, padding=1, momentum=0.01, activation=ACTIVATION):
    '''
    returns a block 3Dconv-3Dbn-activation
    '''
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel, stride=stride, padding=padding),
        activation(),
        nn.BatchNorm3d(out_channels, momentum=momentum),
    )

def conv2d_block(in_channels, out_channels, kernel=3, stride=1, padding=1, activation=ACTIVATION):
    '''
    returns a block conv-activation
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride=stride, padding=padding),
        activation(),
    )

def conv3d_block(in_channels, out_channels, kernel=3, stride=1, padding=1, activation=ACTIVATION):
    '''
    returns a block 3D conv-activation
    '''
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel, stride=stride, padding=padding),
        activation(),
    )


def conv2d_sn_block(in_channels, out_channels, kernel_size, stride, padding, activation, bias=False):
    """
    """
    return nn.Sequential(
        spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)),
        activation(),

    )

def conv2d_intnorm_block(in_channels, out_channels, kernel_size, stride, padding, activation=ACTIVATION, bias=False):
    """
    returns a block 2dconv-InstanceNorm-activation
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
        nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True),
        activation(inplace=True),
    )

def conv2d_transposed_intnorm_block(in_channels, out_channels, kernel_size, stride, padding, activation=ACTIVATION, bias=False):
    """
    returns a block 3dconvtranpose-InstanceNorm-activation
    """
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
        nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True),
        activation(inplace=True)
    )

def conv3d_transposed_intnorm_block(in_channels, out_channels, kernel_size, stride, padding, activation=ACTIVATION, bias=False):
    """
    returns a block 3dconvtranpose-InstanceNorm-activation
    """
    return nn.Sequential(
        nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
        nn.InstanceNorm3d(out_channels, affine=True, track_running_stats=True),
        activation(inplace=True)
    )

def conv3d_intnorm_block(in_channels, out_channels, kernel_size, stride, padding, activation=ACTIVATION, bias=False):
    """
    returns a block 3dconv-InstanceNorm-activation
    """
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
        nn.InstanceNorm3d(out_channels, affine=True, track_running_stats=True),
        activation(inplace=True),
    )

def conv2d_mp_block(in_channels, out_channels, kernel_size, activation=ACTIVATION):
    """
    returns a block 2dconv-MaxPool-activation
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size),
        activation(),
        nn.MaxPool2d(kernel_size=2),
    )

class ResidualBlockInstanceNorm(nn.Module):
    """
    Residual Block with instance normalization
    """
    def __init__(self, dim_in, dim_out, dimensions):
        super(ResidualBlockInstanceNorm, self).__init__()

        if dimensions == 2:
            conv_block = conv2d_intnorm_block
        else:
            conv_block = conv3d_intnorm_block
        
        self.main = nn.Sequential(
            conv_block(in_channels=dim_in, out_channels=dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            conv_block(in_channels=dim_out, out_channels=dim_out, kernel_size=3, stride=1, padding=1, activation=Identity, bias=False)
        )
       
    def forward(self, x):
        return x + self.main(x)


class ResidualBlock(nn.Module):
    """Residual block architecture."""

    def __init__(self, in_channels: int, dimensions: int):
        """Initialize module."""
        super(ResidualBlock, self).__init__()

        if dimensions == 2:
            self.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            )
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.conv2 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            )
            self.bn2 = nn.BatchNorm2d(in_channels)
        
        else:
            self.conv1 = nn.Conv3d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            )
            self.bn1 = nn.BatchNorm3d(in_channels)
            self.conv2 = nn.Conv3d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            )
            self.bn2 = nn.BatchNorm3d(in_channels)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        """Forward pass."""
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += x
        out = self.relu(out)

        return out


class ResidualBottleneckBlock(nn.Module):
    """Residual bottleneck block architecture."""

    def __init__(self, in_channels: int, dimensions:int, bottleneck_filters: int = 64):
        """Initialize module."""
        super(ResidualBottleneckBlock, self).__init__()

        if dimensions == 2:
            self.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=bottleneck_filters,
                kernel_size=1,
                bias=False,
            )
            self.bn1 = nn.BatchNorm2d(bottleneck_filters)
            self.conv2 = nn.Conv2d(
                in_channels=bottleneck_filters,
                out_channels=bottleneck_filters,
                kernel_size=3,
                padding=1,
                bias=False,
            )
            self.bn2 = nn.BatchNorm2d(bottleneck_filters)
            self.conv3 = nn.Conv2d(
                in_channels=bottleneck_filters,
                out_channels=in_channels,
                kernel_size=1,
                bias=False,
            )
            self.bn3 = nn.BatchNorm2d(in_channels)
        else:
            self.conv1 = nn.Conv3d(
                in_channels=in_channels,
                out_channels=bottleneck_filters,
                kernel_size=1,
                bias=False,
            )
            self.bn1 = nn.BatchNorm3d(bottleneck_filters)
            self.conv2 = nn.Conv3d(
                in_channels=bottleneck_filters,
                out_channels=bottleneck_filters,
                kernel_size=3,
                padding=1,
                bias=False,
            )
            self.bn2 = nn.BatchNorm3d(bottleneck_filters)
            self.conv3 = nn.Conv3d(
                in_channels=bottleneck_filters,
                out_channels=in_channels,
                kernel_size=1,
                bias=False,
            )
            self.bn3 = nn.BatchNorm3d(in_channels)

        self.relu = nn.LeakyReLU()

    def forward(self, x):

        """Forward pass."""
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out += x
        out = self.relu(out)

        return out


def weight_init(m):
    """
    Usage:
        model = Model()
        model.applay(weight_init)
    """

    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data, gain=init.calculate_gain('relu'))
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data, gain=init.calculate_gain('relu'))
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data, gain=init.calculate_gain('relu'))
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data, gain=init.calculate_gain('relu'))
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data, gain=init.calculate_gain('tanh'))
        init.normal_(m.bias.data)

