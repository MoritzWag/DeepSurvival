
import pdb 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class Identity(nn.Module):

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
        #activation(),
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
        nn.BatchNorm3d(out_channels, momentum=momentum),
        activation(),
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



class ResidualBlock(nn.Module):
    """Residual block architecture."""

    def __init__(self, in_channels: int):
        """Initialize module."""
        super(ResidualBlock, self).__init__()

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



# class ResidualBlock(nn.Module):
#     """Residual Block architecture.
#     """

#     def __init__(self, in_channels: int, dimensions: int):
#         super(ResidualBlock, self).__init__()

#         if dimensions == 2 :
#             conv_block = conv2d_bn_block
#         else:
#             conv_block = conv3d_bn_block
        
#         self.conv1 = conv_block(in_channels=in_channels,
#                                 out_channels=in_channels)
#         self.conv2 = conv_block(in_channels=in_channels,
#                                 out_channels=in_channels)

#         self.relu = nn.ReLU()

#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.conv2(out)
        
#         out += x
#         out = self.relu(out)

#         return out



class ResidualBottleneckBlock(nn.Module):
    """Residual bottleneck block architecture."""

    def __init__(self, in_channels: int, bottleneck_filters: int = 64):
        """Initialize module."""
        super(ResidualBottleneckBlock, self).__init__()

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



# class ResidualBottleneckBlock(nn.Module):
#     """Residual bottleneck block architecture.
#     """

#     def __init__(self, in_channels: int, dimensions: int, bottleneck_filters: int = 64):
#         super(ResidualBottleneckBlock, self).__init__()
    
#         if dimensions == 2:
#             conv_block = conv2d_bn_block
#         else:
#             conv_block = conv3d_bn_block
        
#         self.conv1 = conv_block(in_channels=in_channels,
#                                 out_channels=bottleneck_filters,
#                                 kernel=1,
#                                 padding=0)
#         self.conv2 = conv_block(in_channels=bottleneck_filters,
#                                 out_channels=bottleneck_filters,
#                                 kernel=3,
#                                 padding=1)
#         self.conv3 = conv_block(in_channels=bottleneck_filters,
#                                 out_channels=in_channels,
#                                 kernel=1,
#                                 padding=0)
        
#         self.relu = nn.LeakyReLU()

#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.conv2(out)
#         out = self.conv3(out)

#         out += x
#         out = self.relu(x)

#         return out

    

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


# def weight_init(m):
#     """
#     Usage:
#         model = Model()
#         model.applay(weight_init)
#     """

#     if isinstance(m, nn.Conv1d):
#         init.normal_(m.weight.data)
#         if m.bias is not None:
#             init.normal_(m.bias.data)
#     elif isinstance(m, nn.Conv2d):
#         init.xavier_normal_(m.weight.data)
#         if m.bias is not None:
#             init.normal_(m.bias.data)
#     elif isinstance(m, nn.Conv3d):
#         init.xavier_normal_(m.weight.data)
#         if m.bias is not None:
#             init.normal_(m.bias.data)
#     elif isinstance(m, nn.ConvTranspose1d):
#         init.normal_(m.weight.data)
#         if m.bias is not None:
#             init.normal_(m.bias.data)
#     elif isinstance(m, nn.ConvTranspose2d):
#         init.xavier_normal_(m.weight.data)
#         if m.bias is not None:
#             init.normal_(m.bias.data)
#     elif isinstance(m, nn.ConvTranspose3d):
#         init.xavier_normal_(m.weight.data)
#         if m.bias is not None:
#             init.normal_(m.bias.data)
#     elif isinstance(m, nn.BatchNorm1d):
#         init.normal_(m.weight.data, mean=1, std=0.02)
#         init.constant_(m.bias.data, 0)
#     elif isinstance(m, nn.BatchNorm2d):
#         init.normal_(m.weight.data, mean=1, std=0.02)
#         init.constant_(m.bias.data, 0)
#     elif isinstance(m, nn.BatchNorm3d):
#         init.normal_(m.weight.data, mean=1, std=0.02)
#         init.constant_(m.bias.data, 0)
#     elif isinstance(m, nn.Linear):
#         init.xavier_normal_(m.weight.data)
#         init.normal_(m.bias.data)