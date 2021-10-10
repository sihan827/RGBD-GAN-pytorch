#
# loss.py
# loss functions for RGBD-GAN
#

import torch
import torch.nn.functional as F
import numpy as np

from warp import *


def loss_dcgan_gen(y_fake, focal_loss_gamma=0.):
    """
    loss function for DCGAN generator
    softplus function used
    """
    fake_sum = torch.sum(F.softplus(-y_fake) * F.sigmoid(-y_fake) ** focal_loss_gamma)
    fake_shape = torch.tensor(np.prod(y_fake.data.shape))
    return fake_sum / fake_shape


def loss_dcgan_dis(y_fake, y_real):
    """
    loss function for DCGAN discriminator
    softplus function used
    """
    fake_loss = torch.sum(F.softplus(y_fake)) / torch.tensor(np.prod(y_fake.data.shape))
    real_loss = torch.sum(F.softplus(y_real)) / torch.tensor(np.prod(y_real.data.shape))

    return fake_loss + real_loss


class Rotate3dLoss:
    """
    class for calculating 3d loss for two camera parameters' images
    """
    def __init__(self, K=None, norm='l1', lambda_geometric=3, device=torch.device('cuda')):
        self.size = None
        self.K = K
        self.inv_K = None
        self.norm = norm
        self.lambda_geometric = lambda_geometric
        self.p = None
        self.device = device

    def init_params(self, size=4):
        """
        initialize camera intrinsic parameter and image pixel homogeneous coordinate
        """
        # camera intrinsic parameter and inverse (3, 3)
        if self.size is None:
            if self.K is not None:
                self.K = torch.tensor(self.K[:3, :3], dtype=torch.float32)
                self.K[:2] *= size / self.K[0, 2] / 2
                self.size = size
            else:
                self.size = size
                self.K = torch.tensor([[size * 2, 0, size / 2], [0, size * 2, size / 2], [0, 0, 1]], dtype=torch.float32)
        else:
            self.size = size
            self.K[:2] *= size / self.K[0, 2] / 2

        self.K = self.K.to(self.device)
        self.inv_K = torch.linalg.inv(self.K)

        # homogeneous grid of image pixels (3, h * w)
        self.p = torch.tensor(np.asarray(
            np.meshgrid(np.arange(size), np.arange(size)) + np.array([np.ones((size, size))])
        ).reshape(3, -1), dtype=torch.float32).to(self.device)

    def __call__(self, img_1, theta_1, img_2, theta_2, occlusion_aware=False, debug=False, max_depth=None, min_depth=None):
        """
        calculate 3d loss of image and warped image
        img : (b, 4, h, w) -> rgb and depth
        theta : (b, 3, 4) -> rotation and translation
        img_1 : theta_1 camera extrinsic parameter image
        img_2 : theta_2 camera extrinsic parameter image
        """
        # check image size
        if self.size != img_1.shape[-1]:
            self.init_params(size=img_1.shape[-1])

        # get depth from images
        # (b, 1, h, w) -> (b, 1, h * w)
        d_1 = img_1[:, -1:].reshape(img_1.shape[0], 1, -1)
        d_2 = img_2[:, -1:].reshape(img_2.shape[0], 1, -1)

        # if theta is not tensor type, change type to tensor
        if not isinstance(theta_1, torch.Tensor):
            theta_1 = torch.tensor(theta_1).to(self.device)
        if not isinstance(theta_2, torch.Tensor):
            theta_2 = torch.tensor(theta_2).to(self.device)

        # split rotation and translation from theta
        R1 = theta_1[:, :3, :3]
        R2 = theta_2[:, :3, :3]
        t1 = theta_1[:, :3, -1:]
        t2 = theta_2[:, :3, -1:]

        # calculate R and t between img_1 and img_2
        R = torch.matmul(R2.transpose(2, 1), R1)
        inv_R = R.transpose(2, 1)
        t = torch.matmul(R1.transpose(2, 1), t2 - t1)

        # img_1's depth * pixels projected from c2
        new_dp_1 = warp(self.K, self.inv_K, R, t, d_1, self.p)

        # img_2's depth * pixels projected from c1
        new_dp_2 = inv_warp(self.K, self.inv_K, inv_R, t, d_2, self.p)

        # use new_dp and image for bilinear sampling to warp RGBD image
        # warp img_2 to c1 using new_dp_1
        warped_1, not_out_1 = bilinear_sampling(img_2, new_dp_1)
        # warp img_1 to c2 using new_dp_2
        warped_2, not_out_2 = bilinear_sampling(img_1, new_dp_2)

        # make original target to calculate loss with warped one
        warped_1_target = torch.cat([img_1[:, :-1].permute(0, 2, 3, 1).reshape(-1, img_1.shape[1] - 1),
                                     new_dp_1[:, :, 2].reshape(-1, 1)], dim=1) * not_out_1[:, None]
        warped_2_target = torch.cat([img_2[:, :-1].permute(0, 2, 3, 1).reshape(-1, img_2.shape[1] - 1),
                                     new_dp_2[:, :, 2].reshape(-1, 1)], dim=1) * not_out_2[:, None]

        if occlusion_aware:
            pass

        if max_depth is not None:
            pass

        if min_depth is not None:
            pass

        # select loss type (l1 or l2)
        if self.norm == 'l1':
            criteria = F.l1_loss
        else:
            criteria = F.mse_loss

        # calculate 3d loss
        # loss between original rgb and warped rgb
        loss = criteria(warped_1[:, :-1], warped_1_target[:, :-1]) + criteria(warped_2[:, :-1], warped_2_target[:, :-1])
        # loss between projected depth and warped depth
        # hyperparameter as lambda_geometric
        loss += (criteria(warped_1[:, -1], warped_1_target[:, -1]) + criteria(warped_2[:, -1], warped_2_target[:, -1])) * self.lambda_geometric

        return loss, torch.cat([new_dp_1, new_dp_2], dim=0)





