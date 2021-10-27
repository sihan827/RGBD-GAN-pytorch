#
# main.py
#

import argparse
import os
import sys
import yaml
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

from util.save_images import convert_batch_images
from util.pggan import make_hidden
from networks import PGGANGenerator, PGGANDiscriminator
from trainer import TrainerPGGAN


def sample_generate(gen, dst, stage, config, rows=8, cols=8, z=None, seed=0, subdir='preview'):
    """
    generate samples from generator and save
    """
    #device = torch.device('cuda:' + str(config['gpu']))
    device = torch.device('cuda')
    torch.random.manual_seed(seed)
    n_images = cols
    if z is None:
        if config['rgb']:
            z = make_hidden(config['ch'], rows * cols, device)
        else:
            z = make_hidden(config['ch'], n_images, device)
            z = torch.tile(z[:, None], (1, rows) + (1,) * (z.ndim - 1)).reshape(rows * cols, *z.shape[1:])
    else:
        z = z[:n_images * rows]
        # 텐서에 디바이스 설정부터 시작
    if config['rgb']:
        theta = None
    else:
        theta = torch.zeros((rows * cols, 6), dtype=torch.float32).to(device)
        theta[:, 1] = torch.tile(torch.linspace(-config['test_y_rotate'], config['test_y_rotate'], rows), (cols,))

        from update import get_camera_matrices
        random_camera_matrices = get_camera_matrices(theta)
        theta = torch.cat([torch.cos(theta[:, :3]), torch.sin(theta[:, :3]), theta[:, 3:]], dim=1)

    gen.eval()
    with torch.no_grad():
        x = gen(z, stage=stage, theta=theta)

    torch.random.seed()
    x = convert_batch_images(x, rows, cols)

    preview_dir = '{}/{}'.format(dst, subdir)
    if not os.path.exists(preview_dir):
        os.makedirs(preview_dir)

    preview_path = preview_dir + '/image_latest.png'
    Image.fromarray(x).save(preview_path)


def get_dataset(path, out_res):
    """
    get dataset
    """
    transform = transforms.Compose([
        transforms.Resize(out_res),
        transforms.CenterCrop(out_res),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return ImageFolder(path, transform=transform)


class CameraParamPrior:
    """
    prepare camera parameters for training from uniform distribution
    """
    def __init__(self, config):
        #self.device = torch.device('cuda:' + str(config['gpu']))
        self.device = torch.device('cuda')
        self.rotation_range = torch.tensor([
            config['x_rotate'], config['y_rotate'], config['z_rotate']]
        ).to(self.device)
        self.camera_param_range = torch.tensor([
            config['x_rotate'], config['y_rotate'], config['z_rotate'],
            config['x_translate'], config['y_translate'], config['z_translate']
        ]).to(self.device)

    def sample(self, batch_size):
        thetas = torch.FloatTensor(batch_size // 2, 6).uniform_(-1, 1).to(self.device)
        eps = torch.FloatTensor(batch_size // 2, 6).uniform_(0, 0.5).to(self.device)
        sign = torch.tensor(np.random.choice(2, size=(batch_size // 2, 3)) * 2 - 1).to(self.device)

        eps[:, :3] = eps[:, :3] * (sign * (self.rotation_range == 3.1415) +
                                          torch.abs(sign) * (self.rotation_range != 3.1415)) * \
                     torch.clip(1 / (self.rotation_range + 1e-8), 0, 1)
        thetas2 = -eps * torch.sign(thetas) + thetas
        thetas = torch.cat([thetas, thetas2], dim=0)

        thetas = thetas * self.camera_param_range[None]
        return thetas


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./", help="directory contains the data and outputs")
    parser.add_argument("--config_path", type=str, default="configs/ffhq_pggan_test.yml", help="config file path")
    args = parser.parse_args()

    # make directory (if not exists)
    root = args.root
    checkpoint_dir = root + 'checkpoint/'
    out_dir = root + 'output/'
    weight_dir = root + 'weight/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

    # get yml config file
    with open(args.config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # get dataset
    data_dir = config['dataset_path']
    out_res = config['out_res']
    dataset = get_dataset(data_dir, out_res)

    # get config options from yml file
    size_starting_epochs = [int(epochs) for epochs in config['size_starting_epochs'].split(',')]
    batchsize = [int(bsize) for bsize in config['batchsize'].split(',')]
    growing_epochs = [int(epochs) for epochs in config['growing_epochs'].split(',')]

    schedule = [size_starting_epochs, batchsize, growing_epochs]
    iteration = config['iteration']
    ch = config['ch']

    adam_lr_g = config['adam_lr_g']
    adam_lr_d = config['adam_lr_d']
    adam_beta1 = config['adam_beta1']
    adam_beta2 = config['adam_beta2']
    lambda_gp = config['lambda_gp']

    device = torch.device('cuda' if torch.cuda.is_available() and config['use_cuda'] else 'cpu')

    if config['architecture'] == 'pggan':
        dis = PGGANDiscriminator(ch, out_res, ch=ch).to(device)
        gen = PGGANGenerator(ch, out_res, ch=ch).to(device)
    else:
        dis = None
        gen = None

    optim_d = torch.optim.Adam(dis.parameters(), lr=adam_lr_d, betas=(adam_beta1, adam_beta2))
    optim_g = torch.optim.Adam(gen.parameters(), lr=adam_lr_g, betas=(adam_beta1, adam_beta2))

    config_train = {
        'generator': gen,
        'discriminator': dis,
        'optim_gen': optim_g,
        'optim_dis': optim_d,
        'dataset': dataset,
        'schedule': schedule,
        'latent_size': ch,
        'iteration': iteration,
        'lambda_gp': lambda_gp,
        'out_res': out_res,
        'root_path': root
    }

    trainer = TrainerPGGAN(config=config_train, device=device)

    trainer.train()





