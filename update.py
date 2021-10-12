#
# update.py
# update codes for GAN
#


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from utils.components import *
from utils.loss import loss_dcgan_dis, loss_dcgan_gen, Rotate3dLoss


def update_camera_matrices(mat, axis1, axis2, theta, device=torch.device('cuda')):
    """
    camera parameters update(rotates for theta) for get_camera_matrices function
    axis1, axis2 : 0~2 integer
    theta : tensor of rotation degree
    returns camera matrices of minibatch
    """
    rot = torch.zeros_like(mat).to(device)
    rot[:, range(4), range(4)] = 1
    rot[:, axis1, axis1] = torch.cos(theta)
    rot[:, axis1, axis2] = -torch.sin(theta)
    rot[:, axis2, axis1] = torch.sin(theta)
    rot[:, axis2, axis2] = torch.cos(theta)
    mat = torch.matmul(rot, mat)
    return mat


def get_camera_matrices(thetas, order=(0, 1, 2), device=torch.device('cuda')):
    """
    generate camera matrices from thetas
    thetas : batchszie x 6 (x_rot, y_rot, z_rot, x_translate, y_translate, z_translate)
    """
    mat = torch.zeros((len(thetas), 4, 4), dtype=torch.float32).to(device)
    mat[:, range(4), range(4)] = torch.tensor([1, 1, -1, 1], dtype=torch.float32).to(device)
    mat[:, 2, 3] = 1

    # update rotation for mat
    for i in order:  # y_rot, x_rot, z_rot
        mat = update_camera_matrices(mat, (i + 1) % 3, (i + 2) % 3, thetas[:, i])

    # update translation for mat
    mat[:, :3, 3] = mat[:, :3, 3] + thetas[:, 3:]

    return mat


class RGBDGAN():
    """
    class of RGBD-GAN
    """
    def __init__(self, models, config, device=torch.device('cuda'), **kwargs):
        if len(models) == 2:
            models = models + [None]

        self.gen, self.dis, self.smoothed_gen = models
        self.device = device

        # stage manager
        self.config = config

        # parse kwargs for updater
        self.smoothing = kwargs.pop('smoothing')
        self.lambda_gp = kwargs.pop('lambda_gp')

        self.total_gpu = kwargs.pop('total_gpu')
        self.prior = kwargs.pop('prior')

        lambda_geometric = self.config.lambda_geometric if self.config.lambda_geometric else 3
        self.rotate_3d_loss = Rotate3dLoss(lambda_geometric=lambda_geometric)
        self.rotate_3d_loss_feature = Rotate3dLoss(norm='l2', lambda_geometric=lambda_geometric)
        self.stage_interval = list(map(int, self.config.stage_interval.split(',')))

        self.camera_param_range = torch.tensor([config.x_rotate, config.y_rotate, config.z_rotate,
                                                config.x_translate, config.y_translate, config.z_translate]).to(device)

    @property
    def stage(self):
        return self.get_stage()

    def get_stage(self):
        for i, interval in enumerate(self.stage_interval):
            if self.config.iteration + 1 <= interval:
                return i - 1 + (self.config.iteration - self.stage_interval[i - 1]) / (interval - self.stage_interval[i - 1])

        return self.config.max_stage - 1e-8

    def get_z_fake_data(self, batch_size):
        return self.gen.make_hidden(batch_size)

    def update(self, x_real):
        use_rotate = True if self.config.iteration > self.config.start_rotation else False







