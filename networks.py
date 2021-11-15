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


class FromRGB(nn.Module):
    """
    FromRGB block in front of D
    """
    def __init__(self, in_ch, out_ch):
        super(FromRGB, self).__init__()
        self.conv = EqualizedLRConv2d(in_ch, out_ch, kernel_size=(1, 1), stride=(1, 1))
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class ToRGB(nn.Module):
    """
    ToRGB block behind of G
    """
    def __init__(self, in_ch, out_ch, rgbd=False, gain=2):
        super(ToRGB, self).__init__()
        self.conv = EqualizedLRConv2d(in_ch, out_ch, kernel_size=(1, 1), stride=(1, 1), gain=gain)

        if rgbd:
            nn.init.constant_(self.conv.w[-1], 0)
            nn.init.constant_(self.conv.b[-1], 0)

    def forward(self, x):
        return self.conv(x)


class GeneratorBlock(nn.Module):
    """
    generator block for PGGAN
    """
    def __init__(self, in_ch, out_ch, rgbd=False, base=False):
        super(GeneratorBlock, self).__init__()
        if base:
            self.upsample = None
            if rgbd:
                # add conditional channel to latent z
                # rigid transform parameters 전부 사용 (9) vs 오직 x,y rotation만 사용 (4)
                self.conv1 = EqualizedLRConv2d(in_ch + 4, out_ch, kernel_size=(4, 4), stride=(1, 1), padding=(3, 3))
            else:
                self.conv1 = EqualizedLRConv2d(in_ch, out_ch, kernel_size=(4, 4), stride=(1, 1), padding=(3, 3))
        else:
            self.upsample = nn.Upsample(scale_factor=2)
            self.conv1 = EqualizedLRConv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv2 = EqualizedLRConv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu = nn.LeakyReLU(0.2)
        # self.pixelnorm = PixelwiseNorm()

        # nn.init.normal_(self.conv1.w)
        # nn.init.normal_(self.conv2.w)
        # nn.init.zeros_(self.conv1.b)
        # nn.init.zeros_(self.conv2.b)

    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = F.normalize(x)
        # x = self.pixelnorm(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = F.normalize(x)
        # x = self.pixelnorm(x)
        return x


class DiscriminatorBlock(nn.Module):
    """
    discriminator block for PGGAN
    """
    def __init__(self, in_ch, out_ch, base=False):
        super(DiscriminatorBlock, self).__init__()
        if base:
            # self.minibatchstd = MinibatchStd()
            # minibatchstd 사용시 conv1 in_ch에 1 더하기
            self.conv1 = EqualizedLRConv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.conv2 = EqualizedLRConv2d(out_ch, out_ch, kernel_size=(4, 4), stride=(1, 1))
            self.outlayer = nn.Sequential(
                nn.Flatten(),
                nn.Linear(out_ch, 1)
            )
        else:
            self.minibatchstd = None
            self.conv1 = EqualizedLRConv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.conv2 = EqualizedLRConv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.outlayer = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.relu = nn.LeakyReLU(0.2)
        # nn.init.normal_(self.conv1.w)
        # nn.init.normal_(self.conv2.w)
        # nn.init.zeros_(self.conv1.b)
        # nn.init.zeros_(self.conv2.b)

    def forward(self, x):
        # if self.minibatchstd is not None:
        #     x = self.minibatchstd(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.outlayer(x)
        return x


class PGGANGenerator(nn.Module):
    """
    generator architecture for PGGAN
    """
    def __init__(self, latent_size, out_res, ch=512, rgbd=False):
        super(PGGANGenerator, self).__init__()
        self.depth = 1
        self.alpha = 1
        self.fade_iters = 0
        self.upsample = nn.Upsample(scale_factor=2)
        self.current_net = nn.ModuleList([GeneratorBlock(latent_size, ch, rgbd=rgbd, base=True)])  # 4x4
        self.rgbd = rgbd
        img_ch = 4 if rgbd else 3
        self.toRGBs = nn.ModuleList([ToRGB(latent_size, img_ch, rgbd=rgbd, gain=1)])

        for d in range(2, int(np.log2(out_res))):
            if d < 5:
                # 8x8, 16x16, 32x32
                in_ch, out_ch = ch, ch
            else:
                # 64x64, 128x128, etc.
                in_ch, out_ch = int(ch / 2 ** (d - 5)), int(ch / 2 ** (d - 4))

            self.current_net.append(GeneratorBlock(in_ch, out_ch))
            self.toRGBs.append(ToRGB(out_ch, img_ch, rgbd=rgbd, gain=1))

    def forward(self, x, theta=None):
        # prepare latent vector with theta condition if rgbd is True
        if self.rgbd and theta is None:
            assert False, "if rgbd is True, theta must not be None"
        if self.rgbd:
            # 10을 곱하느냐 마느냐
            x = torch.cat([x, theta * 10], dim=1)

        for block in self.current_net[:self.depth - 1]:
            x = block(x)
        out = self.current_net[self.depth - 1](x)
        x_rgb = self.toRGBs[self.depth - 1](out)

        if self.alpha < 1:
            x_old = self.upsample(x)
            old_rgb = self.toRGBs[self.depth - 2](x_old)
            x_rgb = (1 - self.alpha) * old_rgb + self.alpha * x_rgb

            self.alpha += self.fade_iters

        if self.rgbd:
            depth = 1 / (F.softplus(x_rgb[:, -1:]) + 1e-4)
            x_rgb = x_rgb[:, :3]
            x_rgb = torch.cat([x_rgb, depth], dim=1)

        return x_rgb

    def growing_net(self, num_iters):
        self.fade_iters = 1 / num_iters
        self.alpha = 1/num_iters

        self.depth += 1


class PGGANDiscriminator(nn.Module):
    """
    discriminator architecture for PGGAN
    """
    def __init__(self, latent_size, out_res, ch=512):
        super(PGGANDiscriminator, self).__init__()
        self.depth = 1
        self.alpha = 1
        self.fade_iters = 0

        self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.current_net = nn.ModuleList([DiscriminatorBlock(latent_size, latent_size, base=True)])  # 4x4
        self.fromRGBs = nn.ModuleList([FromRGB(3, latent_size)])

        for d in range(2, int(np.log2(out_res))):
            if d < 5:
                # 8x8, 16x16, 32x32
                in_ch, out_ch = ch, ch
            else:
                # 64x64, 128x128
                in_ch, out_ch = int(ch / 2 ** (d - 4)), int(ch / 2 ** (d - 5))
            self.current_net.append(DiscriminatorBlock(in_ch, out_ch))
            self.fromRGBs.append(FromRGB(3, in_ch))

    def forward(self, x_rgb):
        x = self.fromRGBs[self.depth - 1](x_rgb)
        x = self.current_net[self.depth - 1](x)
        if self.alpha < 1:
            x_rgb = self.downsample(x_rgb)
            x_old = self.fromRGBs[self.depth - 2](x_rgb)
            x = (1 - self.alpha) * x_old + self.alpha * x
            self.alpha += self.fade_iters

        for block in reversed(self.current_net[:self.depth - 1]):
            x = block(x)

        return x

    def growing_net(self, num_iters):

        self.fade_iters = 1 / num_iters
        self.alpha = 1 / num_iters

        self.depth += 1



