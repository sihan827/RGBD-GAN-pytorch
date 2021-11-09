#
# trainer.py
# codes for training GAN
#
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from util.components import *
from util.loss import loss_gen_bce, loss_dis_bce, Rotate3dLoss


class TrainerPGGAN:
    """
    class of PGGAN for generating RGB images
    """
    # elements for PGGAN_RGB()
    # - model : generator, discriminator
    # - optimizer : optim_gen, optim_dis
    # - schedule
    # - latent_size
    # - dataset
    # - device
    # - out_res
    # - lambda for gradient penalty
    # - iteration
    # - root_path
    def __init__(self, config, device=torch.device('cuda')):
        self.gen = config['generator']
        self.dis = config['discriminator']

        self.optim_g = config['optim_gen']
        self.optim_d = config['optim_dis']

        self.dataset = config['dataset']

        self.schedule = config['schedule']
        self.latent_size = config['latent_size']
        self.lambda_gp = config['lambda_gp']
        self.out_res = config['out_res']
        self.iteration = config['iteration']

        self.checkpoint_dir = config['root_path'] + 'checkpoint/'
        self.out_dir = config['root_path'] + 'output/'
        self.weight_dir = config['root_path'] + 'weight/'

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        if not os.path.exists(self.loss_dir):
            os.makedirs(self.loss_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.device = device

        self.fixed_latent = torch.randn(16, self.latent_size, 1, 1, device=device)

    def train(self):
        # initial definition
        running_loss_d = 0.
        running_loss_g = 0.
        iter_num = 0

        epoch_losses_d = []
        epoch_losses_g = []

        batch_size = self.schedule[1][0]
        growing = self.schedule[2][0]

        data_loader = DataLoader(dataset=self.dataset, batch_size=batch_size, shuffle=True, num_workers=8)

        tot_iter_num = len(self.dataset) / batch_size
        self.gen.fade_iters = (1 - self.gen.alpha) / self.schedule[0][1] / (2 * tot_iter_num)
        self.dis.fade_iters = (1 - self.dis.alpha) / self.schedule[0][1] / (2 * tot_iter_num)

        size = 2 ** (self.gen.depth + 1)
        print("Output Resolution: %d x %d" % (size, size))

        for epoch in range(1, self.iteration + 1):
            self.gen.train()
            epoch_loss_d = 0.
            epoch_loss_g = 0.
            if epoch - 1 in self.schedule[0]:
                if 2 ** (self.gen.depth + 1) < self.out_res:
                    c = self.schedule[0].index(epoch - 1)
                    batch_size = self.schedule[1][c]
                    growing = self.schedule[2][c]
                    data_loader = DataLoader(dataset=self.dataset, batch_size=batch_size, shuffle=True, num_workers=8)
                    tot_iter_num = len(self.dataset) / batch_size
                    self.gen.growing_net(growing * tot_iter_num)
                    self.dis.growing_net(growing * tot_iter_num)
                    size = 2 ** (self.gen.depth + 1)
                    print("Output Resolution: %d x %d" % (size, size))

            print("epoch: %i/%i" % (int(epoch), int(self.iteration)))
            databar = tqdm(data_loader)

            for i, samples in enumerate(databar):

                # update D
                if size != self.out_res:
                    samples = F.interpolate(samples[0], size=size).to(self.device)
                else:
                    samples = samples[0].to(self.device)
                self.optim_d.zero_grad()
                latent_z = torch.randn(samples.size(0), self.latent_size, 1, 1, device=self.device)
                x_fake = self.gen(latent_z)
                y_fake = self.dis(x_fake.detach())
                y_real = self.dis(samples)

                # gradient penalty
                eps = torch.rand(samples.size(0), 1, 1, 1, device=self.device)
                eps = eps.expand_as(samples)
                x_hat = eps * samples + (1 - eps) * x_fake.detach()
                x_hat.requires_grad = True
                px_hat = self.dis(x_hat)
                grad = torch.autograd.grad(outputs=px_hat.sum(), inputs=x_hat, create_graph=True)[0]
                grad_norm = grad.view(samples.size(0), -1).norm(2, dim=1)
                gradient_penalty = self.lambda_gp * ((grad_norm - 1) ** 2).mean()

                # backpropagate D loss
                # bce loss function vs. D의 출력을 바로 사용
                loss_d = loss_dis_bce(y_fake, y_real) + gradient_penalty
                # loss_d = y_fake.mean() - y_real.mean() + gradient_penalty
                loss_d.backward()
                self.optim_d.step()

                # update G
                self.optim_g.zero_grad()
                y_fake = self.dis(x_fake)

                # backpropagate G loss
                # bce loss function vs. D의 출력을 바로 사용
                loss_g = loss_gen_bce(y_fake)
                # loss_g = -y_fake.mean()
                loss_g.backward()
                self.optim_g.step()

                running_loss_d += loss_d.item()
                running_loss_g += loss_g.item()

                epoch_loss_d += loss_d.item()
                epoch_loss_g += loss_g.item()

                iter_num += 1

                # print current loss
                if i % 500 == 0:
                    running_loss_d /= iter_num
                    running_loss_g /= iter_num
                    print('iteration: %d, gp: %.2f' % (i, gradient_penalty))
                    databar.set_description('loss_d: %.3f   loss_g: %.3f' % (running_loss_d, running_loss_g))
                    iter_num = 0
                    running_loss_d = 0.
                    running_loss_g = 0.

            # get total losses of one epoch
            epoch_losses_d.append(epoch_loss_d / tot_iter_num)
            epoch_losses_g.append(epoch_loss_g / tot_iter_num)

            # get checkpoint and save + generate samples
            checkpoint = {'gen': self.gen.state_dict(),
                           'optim_g': self.optim_g.state_dict(),
                           'dis': self.dis.state_dict(),
                           'optim_d': self.optim_d.state_dict(),
                           'epoch_losses_d': epoch_losses_d,
                           'epoch_losses_g': epoch_losses_g,
                           'fixed_latent': self.fixed_latent,
                           'depth': self.gen.depth,
                           'alpha': self.gen.alpha
                           }

            with torch.no_grad():
                # depth 추가 시 이 쪽을 수정하여 출력 grid에 rgb와 depth가 붙어있도록 수정해야함.
                self.gen.eval()
                if epoch == self.iteration:
                    torch.save(checkpoint, self.checkpoint_dir + 'checkpoint_epoch_%d.pth' % epoch)
                    torch.save(self.gen.state_dict(), self.weight_dir + 'gen_weight_epoch_%d.pth' % epoch)
                out_imgs = self.gen(self.fixed_latent)
                out_grid = make_grid(
                    out_imgs, normalize=True, nrow=4, scale_each=True, padding=int(0.5*(2**self.gen.depth))
                ).permute(1, 2, 0)
                plt.imshow(out_grid.cpu())
                plt.savefig(self.out_dir + 'size_%i_epoch_%d' % (size, epoch))


