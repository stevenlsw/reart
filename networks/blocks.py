import torch.nn as nn
import torch.nn.functional as F


def get_conv_layer(kernel_size, in_channels, out_channels, stride=1, pad_type='valid', use_bias=True):
    """
    returns a list of [pad, conv] => should be += to some list, then apply sequential
    """
    conv = nn.Conv1d(in_channels, out_channels,
                     kernel_size=kernel_size,
                     stride=stride, bias=use_bias)

    if pad_type == 'valid':
        return [conv]

    def ZeroPad1d(sizes):
        return nn.ConstantPad1d(sizes, 0)

    if pad_type == 'reflect':
        pad = nn.ReflectionPad1d
    elif pad_type == 'replicate':
        pad = nn.ReplicationPad1d
    elif pad_type == 'zero':
        pad = ZeroPad1d
    else:
        assert 0, "Unsupported padding type: {}".format(pad_type)

    pad_l = (kernel_size - 1) // 2
    pad_r = kernel_size - 1 - pad_l
    return [pad((pad_l, pad_r)), conv]


def get_acti_layer(acti='relu', inplace=True):
    if acti == 'relu':
        return [nn.ReLU(inplace=inplace)]
    elif acti == 'lrelu':
        return [nn.LeakyReLU(0.2, inplace=inplace)]
    elif acti == 'tanh':
        return [nn.Tanh()]
    elif acti == 'sigmoid':
        return [nn.Sigmoid()]
    elif acti == 'softplus':
        return [nn.Softplus()]
    elif acti == 'none':
        return []
    else:
        assert 0, "Unsupported activation: {}".format(acti)


def get_norm_layer(norm='none', norm_dim=None, channel_per_group=2):
    if norm == 'bn':
        return [nn.BatchNorm1d(norm_dim)]
    elif norm == 'gn':
        return [nn.GroupNorm(norm_dim // channel_per_group, norm_dim)]
    elif norm == 'in':
        return [nn.InstanceNorm1d(norm_dim, affine=True)]
    elif norm == 'none':
        return []
    else:
        assert 0, "Unsupported normalization: {}".format(norm)


def get_dropout_layer(dropout=None):
    if dropout is not None:
        return [nn.Dropout(p=dropout)]
    else:
        return []


def get_conv_block(kernel_size, in_channels, out_channels, stride=1, pad_type='valid', use_bias=True, inplace=True,
                   dropout=None, norm='none', acti='none', acti_first=False):
    """
    returns a list of [pad, conv, norm, acti] or [acti, pad, conv, norm]
    """

    layers = get_conv_layer(kernel_size, in_channels, out_channels, stride=stride, pad_type=pad_type, use_bias=use_bias)
    layers += get_dropout_layer(dropout)
    layers += get_norm_layer(norm, norm_dim=out_channels)
    acti_layers = get_acti_layer(acti, inplace=inplace)

    if acti_first:
        return acti_layers + layers
    else:
        return layers + acti_layers


def get_linear_block(in_dim, out_dim, dropout=None, norm='none', acti='relu'):

    use_bias = True
    layers = []
    layers.append(nn.Linear(in_dim, out_dim, bias=use_bias))
    layers += get_dropout_layer(dropout)
    layers += get_norm_layer(norm, norm_dim=out_dim)
    layers += get_acti_layer(acti)

    return layers


class MLPConv1d(nn.Module):
    def __init__(self, in_channel, mlp, bn=True, gn=False, activation="relu", last_activation='none'):
        super(MLPConv1d, self).__init__()

        norm = 'gn' if gn else ('bn' if bn else 'none')

        layers = []
        last_channel = in_channel
        for i, out_channel in enumerate(mlp):
            last_layer = (i == len(mlp) - 1)
            norm = norm if not last_layer else 'none'
            acti = last_activation if last_layer else activation
            use_bias = True if not last_layer else False
            layers += get_conv_block(1, last_channel, out_channel, norm=norm, acti=acti, use_bias=use_bias)
            last_channel = out_channel
        self.model = nn.Sequential(*layers)
        self.out_channel = last_channel

    def forward(self, input):  # input: [B, in_channel, 1]
        return self.model(input)  # [B, out_channel, 1]





