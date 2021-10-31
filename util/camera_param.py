import torch
import numpy as np


class CameraParam:
    """
    prepare camera parameters for training from uniform distribution
    angle is x, y, z axis radian values and clipped by rotation range in config file
    """
    def __init__(self, x_rotate, y_rotate, z_rotate,
                 x_translate, y_translate, z_translate, device=torch.device('cuda')):

        self.rotation_range = torch.tensor([
            x_rotate, y_rotate, z_rotate
        ]).to(device)
        self.camera_param_range = torch.tensor([
            x_rotate, y_rotate, z_rotate, x_translate, y_translate, z_translate
        ])
        self.device = device

    def get_sample_param(self, batch_size):
        """
        generate camera parameters using uniform distribution
        :returns: batch_size x 6 tensor (x_rotate, y_rotate, z_rotate, x_translate, y_translate, z_translate)
        """
        thetas = torch.FloatTensor(batch_size // 2, 6).uniform_(-1, 1).to(self.device)
        eps = torch.FloatTensor(batch_size // 2, 6).uniform_(0, 0.5).to(self.device)
        sign = torch.tensor(np.random.choice(2, size=(batch_size // 2, 3)) * 2 - 1).to(self.device)

        eps[:, :3] = eps[:, :3] * (sign * (self.rotation_range == 3.1415) + torch.abs(sign) *
                                   (self.rotation_range != 3.1415)) * torch.clip(1 / (self.rotation_range + 1e-8), 0, 1)

        thetas2 = -eps * torch.sign(thetas) + thetas
        thetas = torch.cat([thetas, thetas2], dim=0)

        return thetas

    def get_ex_matrices(self, thetas, order=(0, 1, 2)):
        """
        make camera extrinsic matrix using thetas which is returned by get_sample_param
        :returns: batch_size x 4 x 4 tensor
        """
        mat = torch.zeros((len(thetas), 4, 4), dtype=torch.float32).to(self.device)
        mat[:, range(4), range(4)] = torch.tensor([1, 1, -1, 1], dtype=torch.float32).to(self.device)
        mat[:, 2, 3] = 1

        # update rotation for mat
        for i in order: # x_rotate, y_rotate, z_rotate
            axis1 = (i + 1) % 3
            axis2 = (i + 2) % 3

            rot = torch.zeros_like(mat).to(self.device)
            rot[:, range(4), range(4)] = 1
            rot[:, axis1, axis1] = torch.cos(thetas[:, i])
            rot[:, axis1, axis2] = -torch.sin(thetas[:, i])
            rot[:, axis2, axis1] = torch.sin(thetas[:, i])
            rot[:, axis2, axis2] = torch.cos(thetas[:, i])

            mat = torch.matmul(rot, mat)

        # update translation for mat
        mat[:, :3, 3] = mat[:, :3, 3] + thetas[:, 3:]

        return mat

