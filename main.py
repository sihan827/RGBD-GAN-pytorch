#
# main.py
#

import argparse
import os
import sys
import yaml

import numpy as np
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms


def sample_generate(gen, dst, config, rows=8, cols=8, z=None, seed=0, subdir='preview'):
    """
    generate samples from generator and save
    """
    np.random.seed(seed)
    n_images = cols
    if z is None:
        pass


def get_dataset(path):
    """
    get dataset
    """
    transform = transforms.Compose([transforms.ToTensor()])
    return ImageFolder(path, transform=transform)


class RunningHelper:
    """
    prepare configs, optimizers, device and etc.
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda:' + str(config['gpu']))

    @property
    def keep_smoothed_gen(self):
        return self.config['keep_smoothed_gan']

    @property
    def stage_interval(self):
        return self.config['stage_interval']

    def print_log(self, msg):
        print('[Device {}] {}'.format(self.device.index, msg))

    def make_optimizer_adam(self, model, lr, beta1, beta2):
        self.print_log('Use Adam Optimizer with lr = {}, beta1 = {}, beta2 = {}'.format(lr, beta1, beta2))
        return torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", "-g", type=int, default=0)
    parser.add_argument("--config_path", type=str, default="configs/ffhq_pggan.yml")
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config['gpu'] = args.gpu
    print(config['stage_interval'])


