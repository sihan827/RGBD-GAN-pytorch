import argparse
import os
import sys
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn

from networks import PGGANGenerator
from util.save_results import generate_sample_rgbd, convert_batch_images_rgbd, save_batch_sample_rgbd


def make_z(batch_size, z_size, device=torch.device('cuda')):
    z = torch.normal(0, 1, size=(batch_size, z_size, 1, 1))
    z /= torch.sqrt(torch.sum(z * z, dim=1, keepdims=True) / z_size + 1e-8)
    return z.to(device)


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--cp_path", required=True, type=str,  help="directory contains the generator checkpoint")
    parser.add_argument("--save_path", type=str, default="./generated_samples", help="directory sample saved")
    parser.add_argument("--grid_mode", type=bool, default=False,
                        help="if True, make a sample images grid with 4 samples. else, make each images with 1 sample")
    parser.add_argument("--y_rotate", type=float, default=0.6108, help="y_angle value to rotate samples")
    args = parser.parse_args()

    save_path = args.save_path
    cp_path = args.cp_path
    grid_mode = args.grid_mode
    y_rotate = args.y_rotate

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cp = torch.load(cp_path)

    latent_size = cp['gen']['current_net.0.conv1.w'].shape[1] - 4
    ch = cp['gen']['current_net.0.conv1.w'].shape[0]
    out_res = 2 ** (cp['depth'] + 1)

    print("latent size: %4d, channels: %4d, out resolution: %3d" % (latent_size, ch, out_res))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = PGGANGenerator(latent_size, out_res, ch=ch, rgbd=True).to(device)
    generator.load_state_dict(cp['gen'])
    generator.depth = cp['depth']
    generator.alpha = cp['alpha']
    generator.eval()

    latent_z = make_z(4, latent_size, device=device)
    sample_x = generate_sample_rgbd(generator, latent_z, y_rotate, device=device)

    if grid_mode:
        img, depth = convert_batch_images_rgbd(sample_x, rows=8)
        save_batch_sample_rgbd(img, path=os.path.join(save_path, "sample_%.5f.png" % y_rotate), depth=depth, axis=False)
    else:
        list_theta = torch.linspace(-y_rotate, y_rotate, 8)
        img, depth = convert_batch_images_rgbd(sample_x, rows=8, grid_mode=False)
        for num_sample in range(4):
            for num_angle in range(8):
                plt.figure()
                plt.imshow(img[num_sample * 8 + num_angle].permute(1, 2, 0))
                plt.axis("off")
                plt.savefig(
                    os.path.join(save_path, "image_sample_%d_angle_%d_%.5f.png"
                                 % (num_sample, num_angle, list_theta[num_angle].item())),
                    bbox_inches="tight",
                    pad_inches=0
                )
                plt.close()

                plt.figure()
                plt.imshow(depth[num_sample * 8 + num_angle].permute(1, 2, 0), cmap='plasma')
                plt.axis("off")
                plt.savefig(
                    os.path.join(save_path, "depth_sample_%d_angle_%d_%.5f.png"
                                 % (num_sample, num_angle, list_theta[num_angle].item())),
                    bbox_inches="tight",
                    pad_inches=0,
                )
                plt.close()





