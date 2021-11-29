#
# adain.py
# implementation of adaptive instance normalization
#

import warnings
import torch
import torch.nn.functional as F


def do_normalization(x, groups, gamma, beta, eps=1e-5, device=torch.device('cuda')):
    """
    group normalization function for AdaIn
    """
    if x.ndim <= 2:
        raise ValueError(
            'Input dimension must be greater than 2',
            'including batch size dimension (first dimension).'
        )

    if not isinstance(groups, int):
        raise TypeError('Argument \'group\' type must be (int)')

    batch_size = x.shape[0]
    channels = x.shape[1]
    original_shape = x.shape

    if channels % groups != 0:
        raise ValueError('Argument \'group\' must be a divisor of the number of channel.')

    x = torch.reshape(x, (1, batch_size * groups, -1, 1))

    dummy_gamma = torch.ones(batch_size * groups, dtype=torch.float32).to(device)
    dummy_beta = torch.zeros(batch_size * groups, dtype=torch.float32).to(device)
    running_mu = torch.zeros(channels)
    running_std = torch.ones(channels)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        x = F.batch_norm(x, running_mu, running_std, weight=dummy_gamma, bias=dummy_beta, eps=eps)

    x = torch.reshape(x, original_shape)

    target_shape = [batch_size, channels] + [1] * (x.ndim - 2)
    gamma_broadcast = torch.broadcast_to(torch.reshape(gamma, target_shape), x.shape)
    beta_broadcast = torch.broadcast_to(torch.reshape(beta, target_shape), x.shape)

    return x * gamma_broadcast + beta_broadcast


def adain(x, s_scale, s_bias):
    return do_normalization(x, x.shape[1], s_scale, s_bias)
