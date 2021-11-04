#
# trainer.py
# codes for training GAN
#
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from util.components import *
from util.loss import loss_gen_bce, loss_dis_bce, Rotate3dLoss
from util.camera_param import CameraParam
from util.save_images import generate_sample


class TrainerPGGAN:
    """
    class of PGGAN for generating RGB images
    """
    def __init__(self, config_train, config_transform=None, device=torch.device('cuda')):
        self.gen = config_train['generator']
        self.dis = config_train['discriminator']

        self.optim_g = config_train['optim_gen']
        self.optim_d = config_train['optim_dis']

        self.dataset = config_train['dataset']

        self.schedule = config_train['schedule']
        self.latent_size = config_train['latent_size']
        self.lambda_gp = config_train['lambda_gp']
        self.in_res = config_train['in_res']
        self.out_res = config_train['out_res']
        self.iteration = config_train['iteration']

        self.checkpoint_dir = config_train['root_path'] + 'checkpoint/'
        self.out_dir = config_train['root_path'] + 'output/'
        self.weight_dir = config_train['root_path'] + 'weight/'
        self.loss_dir = config_train['root_path'] + 'loss/'

        self.rgbd = config_train['rgbd']

        if self.rgbd:
            self.start_rotation = config_transform['start_rotation']
            self.start_occlusion_aware = config_transform['start_occlusion_aware']
            self.rotate_3d_loss = Rotate3dLoss(lambda_geometric=config_transform['lambda_geometric'])
            self.lambda_depth = config_transform['lambda_depth']
            self.depth_min = config_transform['depth_min']
            self.camera_param = CameraParam(
                config_transform['train_x_rotate'],
                config_transform['train_y_rotate'],
                config_transform['train_z_rotate'],
                config_transform['train_x_translate'],
                config_transform['train_y_translate'],
                config_transform['train_z_translate'],
                device=device
            )

            self.test_x_rotate = config_transform['test_x_rotate']
            self.test_y_rotate = config_transform['test_y_rotate']
            self.test_z_rotate = config_transform['test_z_rotate']
            self.test_x_translate = config_transform['test_x_translate']
            self.test_y_translate = config_transform['test_y_translate']
            self.test_z_translate = config_transform['test_z_translate']

        self.device = device
        self.fixed_latent = self.make_hidden(16)

    def make_hidden(self, batch_size):
        z = torch.normal(0, 1, size=(batch_size, self.latent_size, 1, 1))
        z /= torch.sqrt(torch.sum(z * z, dim=1, keepdims=True) / self.latent_size + 1e-8)
        return z.to(self.device)

    def train(self):
        # initial definition
        running_loss_d = 0.
        running_loss_g = 0.
        iter_num = 0

        epoch_losses_d = np.zeros(self.iteration)
        epoch_losses_g = np.zeros(self.iteration)

        if self.in_res == 4:
            c = 0
        else:
            c = int(np.log2(self.in_res) - 3)

        batch_size = self.schedule[1][c]
        growing = self.schedule[2][c]

        data_loader = DataLoader(dataset=self.dataset, batch_size=batch_size, shuffle=True, num_workers=8)

        tot_iter_num = len(self.dataset) / batch_size
        self.gen.depth = c + 2
        self.dis.depth = c + 2

        size = 2 ** (self.gen.depth + 1)
        print("Output Resolution: %d x %d" % (size, size))

        for epoch in range(1, self.iteration + 1):
            use_rotate = False
            if self.rgbd:
                if epoch >= self.start_rotation:
                    use_rotate = True

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

            print("epoch: %i/%i, batch size: %i" % (int(epoch), int(self.iteration), int(batch_size)))
            databar = tqdm(data_loader)

            for i, samples in enumerate(databar):

                # prepare samples
                if size != self.out_res:
                    samples = F.interpolate(samples[0], size=size).to(self.device)
                else:
                    samples = samples[0].to(self.device)

                # prepare random_camera_matrix
                thetas = None
                if self.rgbd:
                    thetas = self.camera_param.get_sample_param(samples.size(0))
                    random_camera_matrix = self.camera_param.get_ex_matrices(thetas)
                    thetas = torch.reshape(
                        torch.cat([torch.cos(thetas[:, :3]), torch.sin(thetas[:, :3]), thetas[:, 3:]], dim=1),
                        (samples.size(0), 9, 1, 1)
                    )

                # update D
                self.optim_d.zero_grad()
                if self.rgbd:
                    latent_z = torch.cat(
                        [self.make_hidden(samples.size(0) // 2)] * 2,
                        dim=0
                    )
                else:
                    latent_z = self.make_hidden(samples.size(0))

                x_fake = self.gen(latent_z, theta=thetas)
                y_fake = self.dis(x_fake[:, :3].detach())
                y_real = self.dis(samples)

                # gradient penalty
                eps = torch.rand(samples.size(0), 1, 1, 1, device=self.device)
                eps = eps.expand_as(samples)
                x_hat = eps * samples + (1 - eps) * x_fake[:, :3].detach()
                x_hat.requires_grad = True
                px_hat = self.dis(x_hat)
                grad = torch.autograd.grad(outputs=px_hat.sum(), inputs=x_hat, create_graph=True)[0]
                grad_norm = grad.view(samples.size(0), -1).norm(2, dim=1)
                gradient_penalty = self.lambda_gp * ((grad_norm - 1) ** 2).mean()

                # backpropagate D loss
                # bce loss function vs. D의 출력을 바로 사용
                loss_d = loss_dis_bce(y_fake, y_real) + gradient_penalty
                # loss_d = y_fake.mean() - y_real.mean() + gradient_penalty
                assert not torch.isnan(loss_d.data)
                loss_d.backward()
                self.optim_d.step()

                # update G
                self.optim_g.zero_grad()
                y_fake = self.dis(x_fake[:, :3])

                loss_rotate = 0.
                if use_rotate:
                    # 3d loss
                    loss_rotate, warped_dp = self.rotate_3d_loss(
                        x_fake[:samples.size(0) // 2],
                        random_camera_matrix[:samples.size(0) // 2],
                        x_fake[samples.size(0) // 2:],
                        random_camera_matrix[samples.size(0) // 2:],
                        epoch >= self.start_occlusion_aware
                    )

                    if self.lambda_depth > 0:
                        # depth regularization
                        loss_rotate += torch.mean(F.relu(self.depth_min - x_fake[:, -1]) ** 2) * self.lambda_depth

                    assert not torch.isnan(loss_rotate.data)
                    lambda_rotate = 4 if size <= self.out_res else 4
                    loss_rotate = loss_rotate * lambda_rotate

                # backpropagate G loss
                # bce loss function vs. D의 출력을 바로 사용
                loss_g = loss_gen_bce(y_fake) + loss_rotate
                assert not torch.isnan(loss_g.data)
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
            epoch_losses_d[epoch - 1] = (epoch_loss_d / tot_iter_num)
            epoch_losses_g[epoch - 1] = (epoch_loss_g / tot_iter_num)

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
                if self.rgbd:
                    sample_x = generate_sample(self.gen, self.fixed_latent, self.test_y_rotate, device=self.device)
                    plt.imshow(sample_x.cpu())
                    plt.savefig(self.out_dir + 'size_%i_epoch_%d' % (size, epoch))
                else:
                    out_imgs = self.gen(self.fixed_latent)
                    out_grid = make_grid(
                        out_imgs, normalize=True, nrow=4, scale_each=True, padding=int(0.5*(2**self.gen.depth))
                    ).permute(1, 2, 0)
                    plt.imshow(out_grid.cpu())
                    plt.savefig(self.out_dir + 'size_%i_epoch_%d' % (size, epoch))
                plt.close()

                # save loss graph
                plt.figure(figsize=(8, 6))
                plt.title('loss')
                plt.plot(epoch_losses_d, '-', color='red', label='D')
                plt.plot(epoch_losses_g, '-', color='blue', label='G')
                plt.xlabel('iteration')
                plt.ylabel('loss')
                plt.legend()
                plt.tight_layout()
                plt.savefig(self.loss_dir + 'loss')
                plt.close()



