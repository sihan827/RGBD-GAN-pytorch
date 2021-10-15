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
from update import RGBDGAN
from util.pggan import make_hidden


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


def get_dataset(path):
    """
    get dataset
    """
    transform = transforms.Compose([transforms.ToTensor()])
    return ImageFolder(path, transform=transform)


def setup_generator(config):
    """
    prepare generator for training
    """
    #device = torch.device('cuda:' + str(config['gpu']))
    device = torch.device('cuda')
    rgbd = False if config['rgb'] else True
    if config['generator_architecture'] == 'stylegan':
        assert False, f"{config['generator_architecture']} is not suppoerted"
    elif config['generator_architecture'] == 'dcgan':
        from networks import DCGANGenerator
        generator = DCGANGenerator(config['ch'], enable_blur=config['enable_blur'], rgbd=rgbd, use_encoder=config['bigan'],
                                   use_occupancy_net=config['use_occupancy_net_loss'], device=device)
    elif config['generator_architecture'] == 'deepvoxels':
        assert False, f"{config['generator_architecture']} is not suppoerted"
    else:
        assert False, f"{config['generator_architecture']} is not suppoerted"

    return generator


def setup_discriminator(config):
    """
    prepare discriminator for training
    """
    #device = torch.device('cuda:' + str(config['gpu']))
    device = torch.device('cuda')
    from networks import Discriminator
    num_z = 1 if config['generator_architecture'] == "dcgan" else 2
    discriminator = None
    if config["bigan"]:
        pass
    else:
        discriminator = Discriminator(ch=config['ch'], enable_blur=config['enable_blur'], res=config['res_dis'],
                                      device=device)

    return discriminator


class RunningHelper:
    """
    prepare configs, optimizers, device and etc.
    """
    def __init__(self, config):
        self.config = config
        #self.device = torch.device('cuda:' + str(config['gpu']))
        self.device = torch.device('cuda')

    @property
    def keep_smoothed_gen(self):
        return self.config['keep_smoothed_gen']

    @property
    def stage_interval(self):
        return self.config['stage_interval']

    def print_log(self, msg):
        print('[Device {}] {}'.format(self.device.index, msg))

    def make_optimizer_adam(self, model, lr, beta1, beta2):
        self.print_log('Use Adam Optimizer with lr = {}, beta1 = {}, beta2 = {}'.format(lr, beta1, beta2))
        return torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))


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
    parser = argparse.ArgumentParser()
    #parser.add_argument("--gpu", "-g", type=int, default=0)
    parser.add_argument("--config_path", type=str, default="configs/ffhq_pggan_test.yml")
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    #config['gpu'] = args.gpu
    print(config['stage_interval'])

    running_helper = RunningHelper(config)

    # setup generator and discriminator
    generator = setup_generator(config)
    discriminator = setup_discriminator(config)

    models = [generator, discriminator]
    model_names = ['Generator', 'Discriminator']

    # if keep_smoothed_gen is True
    if running_helper.keep_smoothed_gen:
        pass

    # set gpu to model
    generator.to(running_helper.device)
    discriminator.to(running_helper.device)

    if torch.cuda.device_count() > 1:
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)

    # 일단 기본 ToTensor로 0~1 사이 값으로 학습시킨 후 실제 코드대로 -1~1사이 값으로 학습
    # 일단 데이터 10000개만으로 학습
    dataset = get_dataset(config['dataset_path'])
    train_set, val_set = torch.utils.data.random_split(dataset, [69000, 1000])
    print(len(val_set))
    train_loader = data.DataLoader(val_set, batch_size=config['batchsize'], shuffle=True, num_workers=2)

    prior = CameraParamPrior(config)
    iteration = config['iteration']

    optimizer = None
    if config['generator_architecture'] == "stylegan":
        pass
    elif config['generator_architecture'] == 'dcgan':
        optimizer = {
            "gen": running_helper.make_optimizer_adam(generator, config['adam_alpha_g'], config['adam_beta1'],
                                                      config['adam_beta2']),
            "dis": running_helper.make_optimizer_adam(discriminator, config['adam_alpha_d'], config['adam_beta1'],
                                                      config['adam_beta2'])
        }
    elif config['generator_architecture'] == 'deepvoxels':
        pass

    update_args = {
        "models": models,
        "optimizer": optimizer,
        "config": config,
        "lambda_gp": config['lambda_gp'],
        "smoothing": config['smoothing'],
        "prior": prior
    }

    gan = None
    if config['rgb']:
        pass
    else:
        gan = RGBDGAN(**update_args)

    for epoch in tqdm(range(iteration), position=0, leave=True):
        for idx, (x_real, _) in enumerate(train_loader, 0):
            loss_gen, loss_dis = gan.update(x_real, epoch)

            if idx % 100 == 0:
                print(epoch, iteration, '||','loss_gen : ' ,round(loss_gen.data.tolist(),4), ', loss_dis : ', round(loss_dis.data.tolist(),4))

            if epoch % config['evaluation_sample_interval'] == 0:
                stage = gan.get_stage(epoch)
                sample_generate(generator, config['out'], stage, config, rows=8, cols=8)






