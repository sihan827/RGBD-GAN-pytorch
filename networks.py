#
# networks.py
# network architectures
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from util.components import *


class DiscriminatorBlockBase(nn.Module):
    """
    initial block for discriminator
    """
    def __init__(self, ch, out_dim=1):
        super(DiscriminatorBlockBase, self).__init__()

        self.conv1 = EqualizedLRConv2d(ch, ch, 3, 1, 1)
        self.conv2 = EqualizedLRConv2d(ch, ch, 4, 1, 0)
        self.flatten = nn.Flatten()
        self.linear = EqualizedLRLinear(ch, out_dim, gain=1)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.linear(self.flatten(x))

        return x


class DiscriminatorBlock(nn.Module):
    """
    component block for discriminator
    """
    def __init__(self, in_ch, out_ch, enable_blur=False, res=False, bn=False, device=torch.device('cuda')):
        super(DiscriminatorBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.res = res
        self.enable_blur = enable_blur
        self.device = device

        self.conv1 = EqualizedLRConv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = EqualizedLRConv2d(out_ch, out_ch, 3, 1, 1)
        self.relu = nn.LeakyReLU(0.2)
        self.pool = nn.AvgPool2d(2, 2)

        if res:
            self.conv_shortcut = EqualizedLRConv2d(in_ch, out_ch, 3, 1, 1)
        if bn:
            self.bn1 = nn.BatchNorm2d(out_ch)
            self.bn2 = nn.BatchNorm2d(out_ch)
        else:
            self.bn1 = lambda x: x
            self.bn2 = lambda x: x

        if enable_blur:
            k = np.asarray([1, 2, 1]).astype('f')
            k = k[:, None] * k[:, None]
            k = k / np.sum(k)
            blur_k = torch.tensor(np.asarray(k)[None, None, :]).to(self.device)
            self.blur = Blur2d(blur_k)

    def forward(self, x):
        x_ = self.relu(self.bn1(self.conv1(x)))
        if self.res:
            shortcut = self.conv_shortcut(x)
            x = self.bn2(self.conv2(x_)) + shortcut
        else:
            x = self.bn2(self.conv2(x_))
        x = self.relu(x)
        if self.enable_blur:
            x = self.blur(self.pool(x))
        else:
            x = self.pool(x)

        return x


class Discriminator(nn.Module):
    """
    discriminator for GAN
    """
    def __init__(self, ch=512, out_dim=1, enable_blur=False, res=False, device=torch.device('cuda')):
        super(Discriminator, self).__init__()
        self.max_stage = 17
        self.enable_blur = enable_blur

        self.blocks = nn.Sequential(
            DiscriminatorBlockBase(ch, out_dim), # (4, 4)
            DiscriminatorBlock(ch, ch, enable_blur=enable_blur, res=res, device=device),  # (8, 8)
            DiscriminatorBlock(ch, ch, enable_blur=enable_blur, res=res, device=device),  # (16, 16)
            DiscriminatorBlock(ch, ch, enable_blur=enable_blur, res=res, device=device),  # (32, 32)
            DiscriminatorBlock(ch // 2, ch, enable_blur=enable_blur, res=res, device=device),  # (64, 64)
            DiscriminatorBlock(ch // 4, ch // 2, enable_blur=enable_blur, res=res, device=device)  # (128, 128)
        )

        self.ins = nn.Sequential(
            EqualizedLRConv2d(3, ch, 1, 1, 0),
            EqualizedLRConv2d(3, ch, 1, 1, 0),
            EqualizedLRConv2d(3, ch, 1, 1, 0),
            EqualizedLRConv2d(3, ch, 1, 1, 0),
            EqualizedLRConv2d(3, ch // 2, 1, 1, 0),
            EqualizedLRConv2d(3, ch // 4, 1, 1, 0)
        )

        self.pool = nn.AvgPool2d(2, 2)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x, stage, return_hidden=False):
        stage = min(stage, self.max_stage - 1e-8)
        alpha = stage - math.floor(stage)
        stage = math.floor(stage)

        if stage % 2 == 0:
            k = (stage - 2) // 2
            x = self.relu(self.ins[k + 1](x))
            for i in reversed(range(0, (k + 1) + 1)):
                if i == 3:
                    feat = x
                x = self.blocks[i](x)
        else:
            k = (stage - 1) // 2
            x_0 = self.relu(self.ins[k](self.pool(x)))
            x_1 = self.blocks[k + 1](self.relu(self.ins[k + 1](x)))
            assert 0. <= alpha < 1.
            x = (1.0 - alpha) * x_0 + alpha * x_1

            for i in reversed(range(k + 1)):
                if i == 3:
                    feat = x
                x = self.blocks[i](x)
        if return_hidden:
            return x, feat
        else:
            return x


class DCGANBlock(nn.Module):
    """
    component block for PGGAN generator
    """
    def __init__(self, ch=512, ch_in=512, enable_blur=False, device=torch.device('cuda')):
        super(DCGANBlock, self).__init__()
        self.ch = ch
        self.ch_in = ch_in
        self.enable_blur = enable_blur
        self.device = device

        self.noise1 = None
        self.noise2 = None

        self.conv1 = EqualizedLRConv2d(ch_in, ch, 3, 1, 1)
        self.conv2 = EqualizedLRConv2d(ch, ch, 3, 1, 1)
        self.relu = nn.LeakyReLU(0.2)
        self.up = nn.Upsample(scale_factor=2)
        self.pn = PixelwiseNorm()

        if enable_blur:
            k = np.asarray([1, 2, 1]).astype('f')
            k = k[:, None] * k[None, :]
            k = k / np.sum(k)
            blur_k = torch.tensor(np.asarray(k)[None, None, :]).to(device)
            self.blur = Blur2d(blur_k)

    def forward(self, x, add_noise=False):
        if self.enable_blur:
            x = self.blur(self.up(x))
        else:
            x = self.up(x)

        x = self.conv1(x)
        if add_noise:
            pass
        x = self.relu(x)
        x = self.pn(x)

        x = self.conv2(x)
        if add_noise:
            pass
        x = self.relu(x)
        x = self.pn(x)

        return x


class DCGANGenerator(nn.Module):
    """
    generator for PGGAN
    """
    def __init__(self, in_ch=128, ch=512, enable_blur=False, rgbd=False, use_encoder=False, use_occupancy_net=False, initial_depth=None, device=torch.device('cuda')):
        super(DCGANGenerator, self).__init__()
        self.in_ch = in_ch
        self.ch = ch
        self.max_stage = 17
        self.rgbd = rgbd
        self.use_occupancy = use_occupancy_net
        self.enable_blur = enable_blur
        self.device = device

        # check if image is rgbd or rgb
        out_ch = 4 if rgbd else 3

        if initial_depth is None:
            initial_depth = 1.

        if self.rgbd:
            self.linear = EqualizedLRLinear(in_ch + 9, ch * 4 * 4)
        else:
            self.linear = EqualizedLRLinear(in_ch, ch * 4 * 4)
        self.blocks = nn.Sequential(
            DCGANBlock(ch, ch, enable_blur=enable_blur),  # (8, 8)
            DCGANBlock(ch, ch, enable_blur=enable_blur),  # (16, 16)
            DCGANBlock(ch, ch, enable_blur=enable_blur),  # (32, 32)
            DCGANBlock(ch // 2, ch, enable_blur=enable_blur),  # (64, 64)
            DCGANBlock(ch // 4, ch // 2, enable_blur=enable_blur)  # (128, 128)
        )
        self.outs = nn.Sequential(
            EqualizedLRConv2d(ch, out_ch, 1, 1, 0, gain=1),
            EqualizedLRConv2d(ch, out_ch, 1, 1, 0, gain=1),
            EqualizedLRConv2d(ch, out_ch, 1, 1, 0, gain=1),
            EqualizedLRConv2d(ch // 2, out_ch, 1, 1, 0, gain=1),
            EqualizedLRConv2d(ch // 4, out_ch, 1, 1, 0, gain=1)
        )

        self.up = nn.Upsample(scale_factor=2)

        if use_encoder:
            pass
        if use_occupancy_net:
            pass

        # initialize depth weight and bias
        for out in self.outs:
            nn.init.zeros_(out.w[-1])
            nn.init.constant_(out.b[-1], math.log(math.e ** (initial_depth - 1)))

        self.n_blocks = len(self.blocks)
        self.image_size = 128

    # def make_hidden(self, batch_size):
    #     """
    #     make latent vectors z
    #     """
    #     z = torch.rand((batch_size, self.in_ch)).to(self.device)
    #     norm = torch.sqrt(torch.sum(z ** 2, dim=1, keepdim=True) / self.in_ch + 1e-8)
    #     return torch.divide(z, norm)

    def forward(self, z, stage, theta=None, style_mixing_rate=None, add_noise=False, return_feature=False):
        stage = min(stage, self.max_stage - 1e-8)
        alpha = stage - math.floor(stage)
        stage = math.floor(stage)

        if self.rgbd and theta is None:
            assert False, 'theta is None'

        # if rgbd generator, add theta to input x
        if self.rgbd:
            x = torch.cat([z, theta * 10], dim=1)
        else:
            x = z

        x = self.linear(x).reshape(z.shape[0], self.ch, 4, 4)

        if stage % 2 == 0:
            k = (stage - 2) // 2
            for i in range(0, (k + 1)):
                x = self.blocks[i](x, add_noise=add_noise)
                if return_feature and i == 2:
                    feat = x
            x = self.outs[k](x)
        else:
            k = (stage - 1) // 2
            for i in range(k):
                x = self.blocks[i](x, add_noise=add_noise)
                if return_feature and i == 2:
                    feat = x
            x_0 = self.up(self.outs[k-1](x))
            x_1 = self.outs[k](self.blocks[k](x, add_noise=add_noise))
            assert 0. <= alpha < 1.
            x = (1. - alpha) * x_0 + alpha * x_1

        if self.rgbd:
            depth = 1 / (F.softplus(x[:, -1:]) + 1e-4)
            x = x[:, :3]
            x = torch.cat([x, depth], dim=1)

        if self.training:
            if return_feature:
                return x, feat
            else:
                return x
        else:
            min_sample_img_size = 64
            if x.data.shape[2] < min_sample_img_size:
                return F.interpolate(x, size=min_sample_img_size)
            else:
                return x

