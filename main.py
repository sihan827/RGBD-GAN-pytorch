#
# main.py
#

import argparse
import os
import yaml
from PIL import Image
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

from util.pggan import make_hidden
from networks import PGGANGenerator, PGGANDiscriminator
from trainer import TrainerPGGAN


def get_dataset(path, out_res):
    """
    get dataset
    """
    transform = transforms.Compose([
        transforms.Resize(out_res),
        transforms.CenterCrop(out_res),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Lambda(lambda x: x * 2. - 1.)
    ])

    return ImageFolder(path, transform=transform)


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
    loss_dir = root + 'loss/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    if not os.path.exists(loss_dir):
        os.makedirs(loss_dir)

    # get yml config file
    with open(args.config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # get dataset
    data_dir = config['dataset_path']
    in_res = config['in_res']
    out_res = config['out_res']
    dataset = get_dataset(data_dir, out_res)

    # get config options from yml file
    size_starting_epochs = [int(epochs) for epochs in config['size_starting_epochs'].split(',')]
    batchsize = [int(bsize) for bsize in config['batchsize'].split(',')]
    growing_epochs = [int(epochs) for epochs in config['growing_epochs'].split(',')]

    # schedule, epoch, channel(z dimension)
    schedule = [size_starting_epochs, batchsize, growing_epochs]
    iteration = config['iteration']
    ch = config['ch']

    # learning rate, lambda for gradient penalty
    adam_lr_g = config['adam_lr_g']
    adam_lr_d = config['adam_lr_d']
    adam_beta1 = config['adam_beta1']
    adam_beta2 = config['adam_beta2']
    lambda_gp = config['lambda_gp']

    # OSGAN config
    osgan = config['osgan']

    # rgbd, transform configs
    rgbd = config['rgbd']
    if rgbd:
        start_rotation = config['start_rotation']
        start_occlusion_aware = config['start_occlusion_aware']
        lambda_geometric = config['lambda_geometric']
        lambda_depth = config['lambda_depth']
        depth_min = config['depth_min']

        train_x_rotate = config['x_rotate']
        train_y_rotate = config['y_rotate']
        train_z_rotate = config['z_rotate']
        train_x_translate = config['x_translate']
        train_y_translate = config['y_translate']
        train_z_translate = config['z_translate']

        test_x_rotate = config['test_x_rotate']
        test_y_rotate = config['test_y_rotate']
        test_z_rotate = config['test_z_rotate']
        test_x_translate = config['test_x_translate']
        test_y_translate = config['test_y_translate']
        test_z_translate = config['test_z_translate']

    # device
    device = torch.device('cuda' if torch.cuda.is_available() and config['use_cuda'] else 'cpu')

    # select GAN model
    if config['architecture'] == 'pggan':
        dis = PGGANDiscriminator(ch, out_res, res=config['res_dis'], ch=ch).to(device)
        gen = PGGANGenerator(ch, out_res, ch=ch, rgbd=rgbd).to(device)
    else:
        dis = None
        gen = None

    # prepare optimizers
    optim_d = torch.optim.Adam(dis.parameters(), lr=adam_lr_d, betas=(adam_beta1, adam_beta2))
    optim_g = torch.optim.Adam(gen.parameters(), lr=adam_lr_g, betas=(adam_beta1, adam_beta2))

    # config values for training
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
        'in_res': in_res,
        'out_res': out_res,
        'root_path': root,
        'osgan': osgan,
        'rgbd': rgbd
    }

    # config values for transform
    config_transform = {}

    if rgbd:
        config_transform['start_rotation'] = start_rotation
        config_transform['start_occlusion_aware'] = start_occlusion_aware
        config_transform['lambda_geometric'] = lambda_geometric
        config_transform['lambda_depth'] = lambda_depth
        config_transform['depth_min'] = depth_min

        config_transform['train_x_rotate'] = train_x_rotate
        config_transform['train_y_rotate'] = train_y_rotate
        config_transform['train_z_rotate'] = train_z_rotate
        config_transform['train_x_translate'] = train_x_translate
        config_transform['train_y_translate'] = train_y_translate
        config_transform['train_z_translate'] = train_z_translate

        config_transform['test_x_rotate'] = test_x_rotate
        config_transform['test_y_rotate'] = test_y_rotate
        config_transform['test_z_rotate'] = test_z_rotate
        config_transform['test_x_translate'] = test_x_translate
        config_transform['test_y_translate'] = test_y_translate
        config_transform['test_z_translate'] = test_z_translate

    trainer = TrainerPGGAN(config_train=config_train, config_transform=config_transform, device=device)

    trainer.train()





