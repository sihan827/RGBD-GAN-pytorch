#
# components.py
# network components
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PixelwiseNorm(nn.Module):
    """
    layer pixelwise normalization
    """
    def __init__(self, eps=10e-8):
        super(PixelwiseNorm, self).__init__()

        self.eps = eps

    def forward(self, x):
        # x: (b, c, h, w)
        return x / torch.sqrt(torch.sum(x**2, dim=1, keepdim=True) + self.eps)


class EqualizedLRConv2d(nn.Module):
    """
    2d convolution layer using equalized learning rate
    """
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True, gain=2):
        super(EqualizedLRConv2d, self).__init__()

        self.stride = stride
        self.padding = padding

        if type(kernel_size) == int:
            kernel_size = (kernel_size, kernel_size)
        self.scale = np.sqrt(gain / (in_ch * kernel_size[0] * kernel_size[1]))

        self.w = nn.parameter.Parameter(torch.Tensor(out_ch, in_ch, *kernel_size))
        self.b = None

        if bias:
            self.b = nn.parameter.Parameter(torch.Tensor(out_ch))
            nn.init.zeros_(self.b)

        nn.init.normal_(self.w)

    def forward(self, x):
        return F.conv2d(x, self.w * self.scale, self.b, self.stride, self.padding)


class EqualizedLRLinear(nn.Module):
    """
    linear layer using equalized learning rate
    """
    def __init__(self, in_ch, out_ch, bias=True, gain=2):
        super(EqualizedLRLinear, self).__init__()

        self.scale = np.sqrt(gain / in_ch)

        self.w = nn.parameter.Parameter(torch.Tensor(out_ch, in_ch))
        self.b = None

        if bias:
            self.b = nn.parameter.Parameter(torch.Tensor(out_ch))
            nn.init.zeros_(self.b)

        nn.init.normal_(self.w)

    def forward(self, x):
        return F.linear(x, self.w * self.scale, self.b)


class MinibatchStd(nn.Module):
    """
    calculate minibatch std to avoid mode collapse
    """
    def __init__(self):
        super(MinibatchStd, self).__init__()

    def forward(self, x):
        size = list(x.size())
        size[1] = 1

        std = torch.std(x, dim=0)
        mean = torch.mean(std)
        return torch.cat((x, mean.repeat(size)), dim=1)


class Blur2d(nn.Module):
    """
    2d blurring layer
    """
    def __init__(self, w):
        super(Blur2d, self).__init__()
        self.w = w

    def forward(self, x):
        b, ch, h, w = x.shape
        x = torch.reshape(x, (b * ch, 1, h, w))
        x = F.conv2d(x, self.w, stride=1, padding=1)
        x = torch.reshape(x, (b, ch, h, w))
        return x