#
# update.py
# update codes for GAN
#


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from util.components import *
from util.loss import loss_dcgan_dis, loss_dcgan_gen, Rotate3dLoss
from util.pggan import downsize_real, make_hidden


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
    for i in order:  # x_rot, y_rot, z_rot
        mat = update_camera_matrices(mat, (i + 1) % 3, (i + 2) % 3, thetas[:, i], device=device)

    # update translation for mat
    mat[:, :3, 3] = mat[:, :3, 3] + thetas[:, 3:]

    return mat


class RGBGAN():
    """
    class of RGB-GAN updater in one epoch
    """
    def __init__(self, models, optimizer, dataset, schedule, latent_size, device=torch.device('cuda')):
        self.gen = models['generator'].to(device)
        self.dis = models['discriminator'].to(device)

        self.optim_g = optimizer['generator']
        self.optim_d = optimizer['discriminator']

        self.dataset = dataset
        self.schedule = schedule
        self.latent_size = latent_size
        self.device = device


    def get_z_fake_data(self, batch_size):
        return make_hidden(self.config['ch'], batch_size, self.device)

    def update(self, x_real, iteration):
        use_rotate = True if iteration > self.config['start_rotation'] else False

        self.gen.train()
        self.dis.train()

        if self.config['generator_architecture'] == "stylegan":
            pass
        optim_g = self.optimizer['gen']
        optim_d = self.optimizer['dis']

        optim_g.zero_grad()
        optim_d.zero_grad()

        stage = self.get_stage(iteration)
        batch_size = self.config['batchsize']

        z = torch.cat([self.get_z_fake_data(batch_size // 2)] * 2, dim=0)

        thetas = self.prior.sample(batch_size)
        random_camera_matrices = get_camera_matrices(thetas, device=self.device)
        thetas = torch.cat([torch.cos(thetas[:, :3]), torch.sin(thetas[:, :3]), thetas[:, 3:]], dim=1)

        x_real = x_real.to(self.device)
        x_real = downsize_real(x_real, stage)
        v_x_real = torch.autograd.Variable(x_real, requires_grad=True).to(self.device)
        image_size = x_real.shape[2]

        x_fake = self.gen(z, stage, thetas)
        v_x_fake = torch.autograd.Variable(x_fake, requires_grad=True).to(self.device)


        if self.config['bigan']:
            assert False, "bigan is not supported"

        y_fake, feat = self.dis(v_x_fake[:, :3], stage=stage, return_hidden=True)
        # gan loss for generator
        loss_gen = loss_dcgan_gen(y_fake)
        assert not torch.isnan(loss_gen.data)

        if use_rotate:
            # 3d loss
            loss_rotate, warped_dp = self.rotate_3d_loss(x_fake[:batch_size // 2],
                                                         random_camera_matrices[:batch_size // 2],
                                                         x_fake[batch_size // 2:],
                                                         random_camera_matrices[batch_size // 2:],
                                                         iteration >= self.config['start_occlusion_aware'])
            if 'rotate_feature' in self.config:
                if self.config['rotate_feature']:
                    downsample_rate = x_real.shape[2] // feat.shape[2]
                    depth = F.avg_pool2d(x_real[:, -1:], downsample_rate, downsample_rate, 0)
                    feat = torch.cat([feat, depth], dim=1)
                    loss_rotate_feature, _ = self.rotate_3d_loss_feature(feat[:batch_size // 2],
                                                                         random_camera_matrices[:batch_size // 2],
                                                                         feat[batch_size // 2:],
                                                                         random_camera_matrices[batch_size // 2:],
                                                                         iteration >= self.config['start_occlusion_aware'])
                    loss_rotate += loss_rotate_feature

            if self.config['lambda_depth'] > 0:
                # depth loss
                loss_rotate += torch.mean(F.relu(self.config['depth_min'] - x_fake[:, -1]) ** 2) * self.config['lambda_depth']
            assert not torch.isnan(loss_rotate.data)

            lambda_rotate = self.config['lambda_rotate'] if 'lambda_rotate' in self.config else 2
            lambda_rotate = lambda_rotate if image_size <= 128 else lambda_rotate * 2
            loss_gen += loss_rotate * lambda_rotate

            if self.config['use_occupancy_net_loss']:
                pass

        loss_gen.backward()
        if self.config['generator_architecture'] == 'stylegan':
            pass
        optim_g.step()

        if self.smoothed_gen is not None:
            pass

        if self.config['bigan']:
            assert False, "bigan is not supported"

        y_fake, feat = self.dis(x_fake[:, :3], stage=stage, return_hidden=True)
        y_real = self.dis(v_x_real, stage=stage)
        loss_dis = loss_dcgan_dis(y_fake, y_real)

        if not self.config['sn'] and self.lambda_gp > 0:
            grad_x_perturbed, = torch.autograd.grad([y_real], [v_x_real], grad_outputs=torch.ones_like(y_real),
                                                    create_graph=True)
            grad_l2 = torch.sqrt(torch.sum(grad_x_perturbed ** 2, dim=(1, 2, 3)))
            loss_l2 = torch.sum((grad_l2 - 0.0) ** 2) / torch.prod(torch.tensor(grad_l2.shape)).to(self.device)
            loss_gp = self.lambda_gp * loss_l2

            loss_dis = loss_dis + loss_gp

        if use_rotate and 'rotate_feature' in self.config:
            if self.config['rotate_feature']:
                downsample_rate = x_real.shape[2] // feat.shape[2]
                depth = F.avg_pool2d(x_real[:, -1:], downsample_rate, downsample_rate, 0)
                feat = torch.cat([feat, depth], dim=1)
                loss_rotate_feature, _ = self.rotate_3d_loss_feature(feat[:batch_size // 2],
                                                                     random_camera_matrices[:batch_size // 2],
                                                                     feat[batch_size // 2:],
                                                                     random_camera_matrices[batch_size // 2:],
                                                                     iteration >= self.config['start_occlusion_aware'])
                loss_dis -= loss_rotate_feature

                if self.config['sn'] and self.lambda_gp > 0:
                    grad_x_perturbed, = torch.autograd.grad([feat], [v_x_fake], grad_outputs=torch.ones_like(feat),
                                                            create_graph=True)
                    print(grad_x_perturbed)
                    grad_l2 = torch.sqrt(torch.sum(grad_x_perturbed ** 2, dim=(1, 2, 3)))
                    loss_l2 = torch.sum((grad_l2 - 0.0) ** 2) / torch.prod(torch.tensor(grad_l2.shape)).to(self.device)
                    loss_gp = self.lambda_gp * loss_l2
                    loss_dis += loss_gp

        assert not torch.isnan(loss_dis.data)

        loss_dis.backward()
        optim_d.step()

        return loss_gen, loss_dis



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
        self.prior = kwargs.pop('prior')
        self.optimizer = kwargs.pop('optimizer')

        lambda_geometric = self.config['lambda_geometric']
        self.rotate_3d_loss = Rotate3dLoss(lambda_geometric=lambda_geometric)
        self.rotate_3d_loss_feature = Rotate3dLoss(norm='l2', lambda_geometric=lambda_geometric)
        self.stage_interval = list(map(int, self.config['stage_interval'].split(',')))

        self.camera_param_range = torch.tensor([config['x_rotate'], config['y_rotate'], config['z_rotate'],
                                                config['x_translate'], config['y_translate'], config['z_translate']]).to(device)

    def get_stage(self, iteration):
        for i, interval in enumerate(self.stage_interval):
            if iteration + 1 <= interval:
                return i - 1 + (iteration - self.stage_interval[i - 1]) / (interval - self.stage_interval[i - 1])

        return self.config['max_stage'] - 1e-8

    def get_z_fake_data(self, batch_size):
        return make_hidden(self.config['ch'], batch_size, self.device)

    def update(self, x_real, iteration):
        use_rotate = True if iteration > self.config['start_rotation'] else False

        self.gen.train()
        self.dis.train()

        if self.config['generator_architecture'] == "stylegan":
            pass
        optim_g = self.optimizer['gen']
        optim_d = self.optimizer['dis']

        optim_g.zero_grad()
        optim_d.zero_grad()

        stage = self.get_stage(iteration)
        batch_size = self.config['batchsize']

        z = torch.cat([self.get_z_fake_data(batch_size // 2)] * 2, dim=0)

        thetas = self.prior.sample(batch_size)
        random_camera_matrices = get_camera_matrices(thetas, device=self.device)
        thetas = torch.cat([torch.cos(thetas[:, :3]), torch.sin(thetas[:, :3]), thetas[:, 3:]], dim=1)

        x_real = x_real.to(self.device)
        x_real = downsize_real(x_real, stage)
        v_x_real = torch.autograd.Variable(x_real, requires_grad=True).to(self.device)
        image_size = x_real.shape[2]

        x_fake = self.gen(z, stage, thetas)
        v_x_fake = torch.autograd.Variable(x_fake, requires_grad=True).to(self.device)


        if self.config['bigan']:
            assert False, "bigan is not supported"

        y_fake, feat = self.dis(v_x_fake[:, :3], stage=stage, return_hidden=True)
        # gan loss for generator
        loss_gen = loss_dcgan_gen(y_fake)
        assert not torch.isnan(loss_gen.data)

        if use_rotate:
            # 3d loss
            loss_rotate, warped_dp = self.rotate_3d_loss(x_fake[:batch_size // 2],
                                                         random_camera_matrices[:batch_size // 2],
                                                         x_fake[batch_size // 2:],
                                                         random_camera_matrices[batch_size // 2:],
                                                         iteration >= self.config['start_occlusion_aware'])
            if 'rotate_feature' in self.config:
                if self.config['rotate_feature']:
                    downsample_rate = x_real.shape[2] // feat.shape[2]
                    depth = F.avg_pool2d(x_real[:, -1:], downsample_rate, downsample_rate, 0)
                    feat = torch.cat([feat, depth], dim=1)
                    loss_rotate_feature, _ = self.rotate_3d_loss_feature(feat[:batch_size // 2],
                                                                         random_camera_matrices[:batch_size // 2],
                                                                         feat[batch_size // 2:],
                                                                         random_camera_matrices[batch_size // 2:],
                                                                         iteration >= self.config['start_occlusion_aware'])
                    loss_rotate += loss_rotate_feature

            if self.config['lambda_depth'] > 0:
                # depth loss
                loss_rotate += torch.mean(F.relu(self.config['depth_min'] - x_fake[:, -1]) ** 2) * self.config['lambda_depth']
            assert not torch.isnan(loss_rotate.data)

            lambda_rotate = self.config['lambda_rotate'] if 'lambda_rotate' in self.config else 2
            lambda_rotate = lambda_rotate if image_size <= 128 else lambda_rotate * 2
            loss_gen += loss_rotate * lambda_rotate

            if self.config['use_occupancy_net_loss']:
                pass

        loss_gen.backward()
        if self.config['generator_architecture'] == 'stylegan':
            pass
        optim_g.step()

        if self.smoothed_gen is not None:
            pass

        if self.config['bigan']:
            assert False, "bigan is not supported"

        y_fake, feat = self.dis(x_fake[:, :3], stage=stage, return_hidden=True)
        y_real = self.dis(v_x_real, stage=stage)
        loss_dis = loss_dcgan_dis(y_fake, y_real)

        if not self.config['sn'] and self.lambda_gp > 0:
            grad_x_perturbed, = torch.autograd.grad([y_real], [v_x_real], grad_outputs=torch.ones_like(y_real),
                                                    create_graph=True)
            grad_l2 = torch.sqrt(torch.sum(grad_x_perturbed ** 2, dim=(1, 2, 3)))
            loss_l2 = torch.sum((grad_l2 - 0.0) ** 2) / torch.prod(torch.tensor(grad_l2.shape)).to(self.device)
            loss_gp = self.lambda_gp * loss_l2

            loss_dis = loss_dis + loss_gp

        if use_rotate and 'rotate_feature' in self.config:
            if self.config['rotate_feature']:
                downsample_rate = x_real.shape[2] // feat.shape[2]
                depth = F.avg_pool2d(x_real[:, -1:], downsample_rate, downsample_rate, 0)
                feat = torch.cat([feat, depth], dim=1)
                loss_rotate_feature, _ = self.rotate_3d_loss_feature(feat[:batch_size // 2],
                                                                     random_camera_matrices[:batch_size // 2],
                                                                     feat[batch_size // 2:],
                                                                     random_camera_matrices[batch_size // 2:],
                                                                     iteration >= self.config['start_occlusion_aware'])
                loss_dis -= loss_rotate_feature

                if self.config['sn'] and self.lambda_gp > 0:
                    grad_x_perturbed, = torch.autograd.grad([feat], [v_x_fake], grad_outputs=torch.ones_like(feat),
                                                            create_graph=True)
                    print(grad_x_perturbed)
                    grad_l2 = torch.sqrt(torch.sum(grad_x_perturbed ** 2, dim=(1, 2, 3)))
                    loss_l2 = torch.sum((grad_l2 - 0.0) ** 2) / torch.prod(torch.tensor(grad_l2.shape)).to(self.device)
                    loss_gp = self.lambda_gp * loss_l2
                    loss_dis += loss_gp

        assert not torch.isnan(loss_dis.data)

        loss_dis.backward()
        optim_d.step()

        return loss_gen, loss_dis










