"""Code adapted from pytorch-3dunet buildingblocks.

Source: https://github.com/wolny/pytorch-3dunet/tree/master/pytorch3dunet/unet3d/buildingblocks.py
"""


from functools import partial

import torch
from torch import nn as nn
from torch.nn import functional as F

from .se import ChannelSELayer3D, ChannelSpatialSELayer3D, SpatialSELayer3D


def create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding,
                dropout_prob, is3d):
    """
    Create a list of modules with together constitute a single conv layer with non-linearity
    and optional batchnorm/groupnorm.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size(int or tuple): size of the convolving kernel
        order (string): order of things, e.g.
            'cr' -> conv + ReLU
            'gcr' -> groupnorm + conv + ReLU
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
            'bcr' -> batchnorm + conv + ReLU
            'cbrd' -> conv + batchnorm + ReLU + dropout
            'cbrD' -> conv + batchnorm + ReLU + dropout2d
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
        dropout_prob (float): dropout probability
        is3d (bool): is3d (bool): if True use Conv3d, otherwise use Conv2d
    Return:
        list of tuple (name, module)
    """
    assert 'c' in order, "Conv layer MUST be present"
    assert order[0] not in 'rle', 'Non-linearity cannot be the first operation in the layer'

    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU(inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'c':
            # add learnable bias only in the absence of batchnorm/groupnorm
            bias = not ('g' in order or 'b' in order)
            if is3d:
                conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
            else:
                conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)

            modules.append(('conv', conv))
        elif char == 'g':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                num_channels = in_channels
            else:
                num_channels = out_channels

            # use only one group if the given number of groups is greater than the number of channels
            if num_channels < num_groups:
                num_groups = 1

            assert num_channels % num_groups == 0, f'Expected number of channels in input to be divisible by num_groups. num_channels={num_channels}, num_groups={num_groups}'
            modules.append(('groupnorm', nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is3d:
                bn = nn.BatchNorm3d
            else:
                bn = nn.BatchNorm2d

            if is_before_conv:
                modules.append(('batchnorm', bn(in_channels)))
            else:
                modules.append(('batchnorm', bn(out_channels)))
        elif char == 'd':
            modules.append(('dropout', nn.Dropout(p=dropout_prob)))
        elif char == 'D':
            modules.append(('dropout2d', nn.Dropout2d(p=dropout_prob)))
        else:
            raise ValueError(f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c', 'd', 'D']")

    return modules


class SingleConv(nn.Sequential):
    """
    Basic convolutional module consisting of a Conv3d, non-linearity and optional batchnorm/groupnorm. The order
    of operations can be specified via the `order` parameter

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding
        dropout_prob (float): dropout probability, default 0.1
        is3d (bool): if True use Conv3d, otherwise use Conv2d
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='gcr', num_groups=8,
                 padding=1, dropout_prob=0.1, is3d=True):
        """
        Initialize SingleConv module.

        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        kernel_size : int or tuple, optional
            Size of the convolving kernel (default: 3)
        order : str, optional
            Determines the order of layers (default: 'gcr')
        num_groups : int, optional
            Number of groups for GroupNorm (default: 8)
        padding : int or tuple, optional
            Zero-padding added to all sides (default: 1)
        dropout_prob : float, optional
            Dropout probability (default: 0.1)
        is3d : bool, optional
            If True use Conv3d, otherwise Conv2d (default: True)
        """
        super(SingleConv, self).__init__()

        for name, module in create_conv(in_channels, out_channels, kernel_size, order,
                                        num_groups, padding, dropout_prob, is3d):
            self.add_module(name, module)


class DoubleConv(nn.Sequential):
    """
    A module consisting of two consecutive convolution layers (e.g. BatchNorm3d+ReLU+Conv3d).
    We use (Conv3d+ReLU+GroupNorm3d) by default.
    This can be changed however by providing the 'order' argument, e.g. in order
    to change to Conv3d+BatchNorm3d+ELU use order='cbe'.
    Use padded convolutions to make sure that the output (H_out, W_out) is the same
    as (H_in, W_in), so that you don't have to crop in the decoder path.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        encoder (bool): if True we're in the encoder path, otherwise we're in the decoder
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
        upscale (int): number of the convolution to upscale in encoder if DoubleConv, default: 2
        dropout_prob (float or tuple): dropout probability for each convolution, default 0.1
        is3d (bool): if True use Conv3d instead of Conv2d layers
    """

    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, order='gcr',
                 num_groups=8, padding=1, upscale=2, dropout_prob=0.1, is3d=True):
        """
        Initialize DoubleConv module.

        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        encoder : bool
            If True we're in the encoder path, otherwise decoder
        kernel_size : int or tuple, optional
            Size of the convolving kernel (default: 3)
        order : str, optional
            Determines the order of layers (default: 'gcr')
        num_groups : int, optional
            Number of groups for GroupNorm (default: 8)
        padding : int or tuple, optional
            Zero-padding added to all sides (default: 1)
        upscale : int, optional
            Number of the convolution to upscale in encoder (default: 2)
        dropout_prob : float or tuple, optional
            Dropout probability for each convolution (default: 0.1)
        is3d : bool, optional
            If True use Conv3d instead of Conv2d (default: True)
        """
        super(DoubleConv, self).__init__()
        if encoder:
            # we're in the encoder path
            conv1_in_channels = in_channels
            if upscale == 1:
                conv1_out_channels = out_channels
            else:
                conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # check if dropout_prob is a tuple and if so
        # split it for different dropout probabilities for each convolution.
        if isinstance(dropout_prob, list) or isinstance(dropout_prob, tuple):
            dropout_prob1 = dropout_prob[0]
            dropout_prob2 = dropout_prob[1]
        else:
            dropout_prob1 = dropout_prob2 = dropout_prob

        # conv1
        self.add_module('SingleConv1',
                        SingleConv(conv1_in_channels, conv1_out_channels, kernel_size, order, num_groups,
                                   padding=padding, dropout_prob=dropout_prob1, is3d=is3d))
        # conv2
        self.add_module('SingleConv2',
                        SingleConv(conv2_in_channels, conv2_out_channels, kernel_size, order, num_groups,
                                   padding=padding, dropout_prob=dropout_prob2, is3d=is3d))


class ResNetBlock(nn.Module):
    """
    Residual block that can be used instead of standard DoubleConv in the Encoder module.
    Motivated by: https://arxiv.org/pdf/1706.00120.pdf

    Notice we use ELU instead of ReLU (order='cge') and put non-linearity after the groupnorm.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='cge', num_groups=8, is3d=True, **kwargs):
        """
        Initialize ResNetBlock module.

        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        kernel_size : int or tuple, optional
            Size of the convolving kernel (default: 3)
        order : str, optional
            Determines the order of layers (default: 'cge')
        num_groups : int, optional
            Number of groups for GroupNorm (default: 8)
        is3d : bool, optional
            If True use Conv3d, otherwise Conv2d (default: True)
        **kwargs : dict
            Additional keyword arguments
        """
        super(ResNetBlock, self).__init__()

        if in_channels != out_channels:
            # conv1x1 for increasing the number of channels
            if is3d:
                self.conv1 = nn.Conv3d(in_channels, out_channels, 1)
            else:
                self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.conv1 = nn.Identity()

        # residual block
        self.conv2 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=order, num_groups=num_groups,
                                is3d=is3d)
        # remove non-linearity from the 3rd convolution since it's going to be applied after adding the residual
        n_order = order
        for c in 'rel':
            n_order = n_order.replace(c, '')
        self.conv3 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=n_order,
                                num_groups=num_groups, is3d=is3d)

        # create non-linearity separately
        if 'l' in order:
            self.non_linearity = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif 'e' in order:
            self.non_linearity = nn.ELU(inplace=True)
        else:
            self.non_linearity = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Forward pass through the ResNetBlock.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output tensor after residual connection and non-linearity
        """
        # apply first convolution to bring the number of channels to out_channels
        residual = self.conv1(x)

        # residual block
        out = self.conv2(residual)
        out = self.conv3(out)

        out += residual
        out = self.non_linearity(out)

        return out


class ResNetBlockSE(ResNetBlock):
    """
    ResNetBlock with Squeeze and Excitation module.

    Extends ResNetBlock by adding a Squeeze and Excitation layer at the end.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='cge', num_groups=8, se_module='scse', **kwargs):
        """
        Initialize ResNetBlockSE module.

        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        kernel_size : int or tuple, optional
            Size of the convolving kernel (default: 3)
        order : str, optional
            Determines the order of layers (default: 'cge')
        num_groups : int, optional
            Number of groups for GroupNorm (default: 8)
        se_module : str, optional
            Type of SE module: 'scse', 'cse', or 'sse' (default: 'scse')
        **kwargs : dict
            Additional keyword arguments
        """
        super(ResNetBlockSE, self).__init__(
            in_channels, out_channels, kernel_size=kernel_size, order=order,
            num_groups=num_groups, **kwargs)
        assert se_module in ['scse', 'cse', 'sse']
        if se_module == 'scse':
            self.se_module = ChannelSpatialSELayer3D(num_channels=out_channels, reduction_ratio=1)
        elif se_module == 'cse':
            self.se_module = ChannelSELayer3D(num_channels=out_channels, reduction_ratio=1)
        elif se_module == 'sse':
            self.se_module = SpatialSELayer3D(num_channels=out_channels)

    def forward(self, x):
        """
        Forward pass through the ResNetBlockSE.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output tensor after ResNet block and SE module
        """
        out = super().forward(x)
        out = self.se_module(out)
        return out


class Encoder(nn.Module):
    """
    A single module from the encoder path consisting of the optional max
    pooling layer (one may specify the MaxPool kernel_size to be different
    from the standard (2,2,2), e.g. if the volumetric data is anisotropic
    (make sure to use complementary scale_factor in the decoder path) followed by
    a basic module (DoubleConv or ResNetBlock).

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int or tuple): size of the convolving kernel
        apply_pooling (bool): if True use MaxPool3d before DoubleConv
        pool_kernel_size (int or tuple): the size of the window
        pool_type (str): pooling layer: 'max' or 'avg'
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
        upscale (int): number of the convolution to upscale in encoder if DoubleConv, default: 2
        dropout_prob (float or tuple): dropout probability, default 0.1
        is3d (bool): use 3d or 2d convolutions/pooling operation
    """

    def __init__(self, in_channels, out_channels, conv_kernel_size=3, apply_pooling=True,
                 pool_kernel_size=2, pool_type='max', basic_module=DoubleConv, conv_layer_order='gcr',
                 num_groups=8, padding=1, upscale=2, dropout_prob=0.1, is3d=True):
        """
        Initialize Encoder module.

        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        conv_kernel_size : int or tuple, optional
            Size of the convolving kernel (default: 3)
        apply_pooling : bool, optional
            If True use MaxPool3d/AvgPool3d before basic module (default: True)
        pool_kernel_size : int or tuple, optional
            Size of the pooling window (default: 2)
        pool_type : str, optional
            Pooling layer type: 'max' or 'avg' (default: 'max')
        basic_module : nn.Module, optional
            Basic module class (default: DoubleConv)
        conv_layer_order : str, optional
            Determines the order of layers (default: 'gcr')
        num_groups : int, optional
            Number of groups for GroupNorm (default: 8)
        padding : int or tuple, optional
            Zero-padding added to all sides (default: 1)
        upscale : int, optional
            Number of the convolution to upscale (default: 2)
        dropout_prob : float or tuple, optional
            Dropout probability (default: 0.1)
        is3d : bool, optional
            Use 3d or 2d operations (default: True)
        """
        super(Encoder, self).__init__()
        assert pool_type in ['max', 'avg']
        if apply_pooling:
            if pool_type == 'max':
                if is3d:
                    self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)
                else:
                    self.pooling = nn.MaxPool2d(kernel_size=pool_kernel_size)
            else:
                if is3d:
                    self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size)
                else:
                    self.pooling = nn.AvgPool2d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=True,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups,
                                         padding=padding,
                                         upscale=upscale,
                                         dropout_prob=dropout_prob,
                                         is3d=is3d)

    def forward(self, x):
        """
        Forward pass through the Encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output tensor after optional pooling and basic module
        """
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x


class Decoder(nn.Module):
    """
    A single module for decoder path consisting of the upsampling layer
    (either learned ConvTranspose3d or nearest neighbor interpolation)
    followed by a basic module (DoubleConv or ResNetBlock).

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int or tuple): size of the convolving kernel
        scale_factor (int or tuple): used as the multiplier for the image H/W/D in
            case of nn.Upsample or as stride in case of ConvTranspose3d, must reverse the MaxPool3d operation
            from the corresponding encoder
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
        upsample (str): algorithm used for upsampling:
            InterpolateUpsampling:   'nearest' | 'linear' | 'bilinear' | 'trilinear' | 'area'
            TransposeConvUpsampling: 'deconv'
            No upsampling:           None
            Default: 'default' (chooses automatically)
        dropout_prob (float or tuple): dropout probability, default 0.1
    """

    def __init__(self, in_channels, out_channels, conv_kernel_size=3, scale_factor=2, basic_module=DoubleConv,
                 conv_layer_order='gcr', num_groups=8, padding=1, upsample='default',
                 dropout_prob=0.1, is3d=True):
        """
        Initialize Decoder module.

        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        conv_kernel_size : int or tuple, optional
            Size of the convolving kernel (default: 3)
        scale_factor : int or tuple, optional
            Multiplier for image dimensions or stride (default: 2)
        basic_module : nn.Module, optional
            Basic module class (default: DoubleConv)
        conv_layer_order : str, optional
            Determines the order of layers (default: 'gcr')
        num_groups : int, optional
            Number of groups for GroupNorm (default: 8)
        padding : int or tuple, optional
            Zero-padding added to all sides (default: 1)
        upsample : str, optional
            Upsampling algorithm (default: 'default')
        dropout_prob : float or tuple, optional
            Dropout probability (default: 0.1)
        is3d : bool, optional
            Use 3d operations (default: True)
        """
        super(Decoder, self).__init__()

        # perform concat joining per default
        concat = True

        # don't adapt channels after join operation
        adapt_channels = False

        if upsample is not None and upsample != 'none':
            if upsample == 'default':
                if basic_module == DoubleConv:
                    upsample = 'nearest'  # use nearest neighbor interpolation for upsampling
                    concat = True  # use concat joining
                    adapt_channels = False  # don't adapt channels
                elif basic_module == ResNetBlock or basic_module == ResNetBlockSE:
                    upsample = 'deconv'  # use deconvolution upsampling
                    concat = False  # use summation joining
                    adapt_channels = True  # adapt channels after joining

            # perform deconvolution upsampling if mode is deconv
            if upsample == 'deconv':
                self.upsampling = TransposeConvUpsampling(in_channels=in_channels, out_channels=out_channels,
                                                          kernel_size=conv_kernel_size, scale_factor=scale_factor,
                                                          is3d=is3d)
            else:
                self.upsampling = InterpolateUpsampling(mode=upsample)
        else:
            # no upsampling
            self.upsampling = NoUpsampling()
            # concat joining
            self.joining = partial(self._joining, concat=True)

        # perform joining operation
        self.joining = partial(self._joining, concat=concat)

        # adapt the number of in_channels for the ResNetBlock
        if adapt_channels is True:
            in_channels = out_channels

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=False,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups,
                                         padding=padding,
                                         dropout_prob=dropout_prob,
                                         is3d=is3d)

    def forward(self, encoder_features, x):
        """
        Forward pass through the Decoder.

        Parameters
        ----------
        encoder_features : torch.Tensor
            Features from the corresponding encoder
        x : torch.Tensor
            Input tensor from previous decoder layer

        Returns
        -------
        torch.Tensor
            Output tensor after upsampling, joining, and basic module
        """
        x = self.upsampling(encoder_features=encoder_features, x=x)
        x = self.joining(encoder_features, x)
        x = self.basic_module(x)
        return x

    @staticmethod
    def _joining(encoder_features, x, concat):
        """
        Join encoder features with decoder output.

        Parameters
        ----------
        encoder_features : torch.Tensor
            Features from the corresponding encoder
        x : torch.Tensor
            Input tensor from previous decoder layer
        concat : bool
            If True concatenate, otherwise add

        Returns
        -------
        torch.Tensor
            Joined tensor
        """
        if concat:
            return torch.cat((encoder_features, x), dim=1)
        else:
            return encoder_features + x


def create_encoders(in_channels, f_maps, basic_module, conv_kernel_size, conv_padding,
                    conv_upscale, dropout_prob,
                    layer_order, num_groups, pool_kernel_size, is3d):
    """
    Create encoder path consisting of Encoder modules.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    f_maps : list or tuple
        Number of feature maps at each level
    basic_module : nn.Module
        Basic module class for encoder
    conv_kernel_size : int or tuple
        Size of the convolving kernel
    conv_padding : int or tuple
        Zero-padding added to all sides
    conv_upscale : int
        Number of the convolution to upscale
    dropout_prob : float
        Dropout probability
    layer_order : str
        Determines the order of layers
    num_groups : int
        Number of groups for GroupNorm
    pool_kernel_size : int or tuple
        Size of the pooling window
    is3d : bool
        Use 3d operations

    Returns
    -------
    nn.ModuleList
        List of Encoder modules
    """
    # create encoder path consisting of Encoder modules. Depth of the encoder is equal to `len(f_maps)`
    encoders = []
    for i, out_feature_num in enumerate(f_maps):
        if i == 0:
            # apply conv_coord only in the first encoder if any
            encoder = Encoder(in_channels, out_feature_num,
                              apply_pooling=False,  # skip pooling in the firs encoder
                              basic_module=basic_module,
                              conv_layer_order=layer_order,
                              conv_kernel_size=conv_kernel_size,
                              num_groups=num_groups,
                              padding=conv_padding,
                              upscale=conv_upscale,
                              dropout_prob=dropout_prob,
                              is3d=is3d)
        else:
            encoder = Encoder(f_maps[i - 1], out_feature_num,
                              basic_module=basic_module,
                              conv_layer_order=layer_order,
                              conv_kernel_size=conv_kernel_size,
                              num_groups=num_groups,
                              pool_kernel_size=pool_kernel_size,
                              padding=conv_padding,
                              upscale=conv_upscale,
                              dropout_prob=dropout_prob,
                              is3d=is3d)

        encoders.append(encoder)

    return nn.ModuleList(encoders)


def create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order,
                    num_groups, upsample, dropout_prob, is3d):
    """
    Create decoder path consisting of Decoder modules.

    Parameters
    ----------
    f_maps : list or tuple
        Number of feature maps at each level
    basic_module : nn.Module
        Basic module class for decoder
    conv_kernel_size : int or tuple
        Size of the convolving kernel
    conv_padding : int or tuple
        Zero-padding added to all sides
    layer_order : str
        Determines the order of layers
    num_groups : int
        Number of groups for GroupNorm
    upsample : str
        Upsampling algorithm
    dropout_prob : float
        Dropout probability
    is3d : bool
        Use 3d operations

    Returns
    -------
    nn.ModuleList
        List of Decoder modules
    """
    # create decoder path consisting of the Decoder modules. The length of the decoder list is equal to `len(f_maps) - 1`
    decoders = []
    reversed_f_maps = list(reversed(f_maps))
    for i in range(len(reversed_f_maps) - 1):
        if basic_module == DoubleConv and upsample != 'deconv':
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
        else:
            in_feature_num = reversed_f_maps[i]

        out_feature_num = reversed_f_maps[i + 1]

        decoder = Decoder(in_feature_num, out_feature_num,
                          basic_module=basic_module,
                          conv_layer_order=layer_order,
                          conv_kernel_size=conv_kernel_size,
                          num_groups=num_groups,
                          padding=conv_padding,
                          upsample=upsample,
                          dropout_prob=dropout_prob,
                          is3d=is3d)
        decoders.append(decoder)
    return nn.ModuleList(decoders)


class AbstractUpsampling(nn.Module):
    """
    Abstract class for upsampling. A given implementation should upsample a given 5D input tensor using either
    interpolation or learned transposed convolution.
    """

    def __init__(self, upsample):
        """
        Initialize AbstractUpsampling.

        Parameters
        ----------
        upsample : callable
            Upsampling function
        """
        super(AbstractUpsampling, self).__init__()
        self.upsample = upsample

    def forward(self, encoder_features, x):
        """
        Forward pass through upsampling.

        Parameters
        ----------
        encoder_features : torch.Tensor
            Features from the corresponding encoder
        x : torch.Tensor
            Input tensor to upsample

        Returns
        -------
        torch.Tensor
            Upsampled tensor
        """
        # get the spatial dimensions of the output given the encoder_features
        output_size = encoder_features.size()[2:]
        # upsample the input and return
        return self.upsample(x, output_size)


class InterpolateUpsampling(AbstractUpsampling):
    """
    Args:
        mode (str): algorithm used for upsampling:
            'nearest' | 'linear' | 'bilinear' | 'trilinear' | 'area'. Default: 'nearest'
            used only if transposed_conv is False
    """

    def __init__(self, mode='nearest'):
        """
        Initialize InterpolateUpsampling.

        Parameters
        ----------
        mode : str, optional
            Algorithm used for upsampling (default: 'nearest')
        """
        upsample = partial(self._interpolate, mode=mode)
        super().__init__(upsample)

    @staticmethod
    def _interpolate(x, size, mode):
        """
        Interpolate tensor to target size.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        size : tuple
            Target spatial dimensions
        mode : str
            Interpolation mode

        Returns
        -------
        torch.Tensor
            Interpolated tensor
        """
        return F.interpolate(x, size=size, mode=mode)


class TransposeConvUpsampling(AbstractUpsampling):
    """
    Args:
        in_channels (int): number of input channels for transposed conv
            used only if transposed_conv is True
        out_channels (int): number of output channels for transpose conv
            used only if transposed_conv is True
        kernel_size (int or tuple): size of the convolving kernel
            used only if transposed_conv is True
        scale_factor (int or tuple): stride of the convolution
            used only if transposed_conv is True
        is3d (bool): if True use ConvTranspose3d, otherwise use ConvTranspose2d
    """

    class Upsample(nn.Module):
        """
        Workaround the 'ValueError: requested an output size...' in the `_output_padding` method in
        transposed convolution. It performs transposed conv followed by the interpolation to the correct size if necessary.
        """

        def __init__(self, conv_transposed, is3d):
            """
            Initialize Upsample.

            Parameters
            ----------
            conv_transposed : nn.Module
                Transposed convolution layer
            is3d : bool
                Whether using 3D operations
            """
            super().__init__()
            self.conv_transposed = conv_transposed
            self.is3d = is3d

        def forward(self, x, size):
            """
            Forward pass with transposed conv and interpolation.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor
            size : tuple
                Target spatial dimensions

            Returns
            -------
            torch.Tensor
                Upsampled tensor
            """
            x = self.conv_transposed(x)
            return F.interpolate(x, size=size)

    def __init__(self, in_channels, out_channels, kernel_size=3, scale_factor=2, is3d=True):
        """
        Initialize TransposeConvUpsampling.

        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        kernel_size : int or tuple, optional
            Size of the convolving kernel (default: 3)
        scale_factor : int or tuple, optional
            Stride of the convolution (default: 2)
        is3d : bool, optional
            If True use ConvTranspose3d (default: True)
        """
        # make sure that the output size reverses the MaxPool3d from the corresponding encoder
        if is3d is True:
            conv_transposed = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size,
                                                 stride=scale_factor, padding=1, bias=False)
        else:
            conv_transposed = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,
                                                 stride=scale_factor, padding=1, bias=False)
        upsample = self.Upsample(conv_transposed, is3d)
        super().__init__(upsample)


class NoUpsampling(AbstractUpsampling):
    """
    No-operation upsampling that returns input unchanged.
    """

    def __init__(self):
        """
        Initialize NoUpsampling.
        """
        super().__init__(self._no_upsampling)

    @staticmethod
    def _no_upsampling(x, size):
        """
        Return input tensor unchanged.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        size : tuple
            Target size (ignored)

        Returns
        -------
        torch.Tensor
            Input tensor unchanged
        """
        return x
