#
# save_results.py
# functions for generated sample images
#

import torch
import matplotlib.pyplot as plt


def generate_sample_rgbd(gen, fixed_z, test_y_rotate, angle_range=8, num_sample=4, device=torch.device("cuda")):
    """
    generate sample images with multiple y angles using given generator model 
    """
    latent_z = fixed_z[:num_sample]
    latent_z = torch.cat(
        [torch.cat([latent_z[i].reshape(1, -1, 1, 1)] * angle_range, dim=0) for i in range(num_sample)],
        dim=0
    )

    theta = torch.zeros((num_sample * angle_range, 6), dtype=torch.float32).to(device)
    theta[:, 1] = torch.tile(torch.linspace(-test_y_rotate, test_y_rotate, angle_range), (num_sample, ))
    theta = torch.reshape(
        # only use x,y rotations with cos, sin functions
        # torch.cat([torch.cos(theta[:, :3]), torch.sin(theta[:, :3]), theta[:, 3:]], dim=1),
        torch.cat([torch.cos(theta[:, :2]), torch.sin(theta[:, :2])], dim=1),
        # (theta.shape[0], 9, 1, 1)
        (theta.shape[0], 4, 1, 1)
    )

    sample_x = gen(latent_z, theta=theta)

    return sample_x


def convert_batch_images_rgbd(x, rows, grid_mode=True):
    """
    convert batch sample images and depths to one grid
    """
    x = x.cpu()
    depth = x[:, -1:]

    x = x[:, :-1]
    x = torch.clip(x * 127.5 + 127.5, 0., 255.).type(torch.uint8)
    depth = torch.clip(1 / (depth + 1e-8) * 128., 0., 255.).type(torch.uint8)
    _, _, h, w = x.shape
    
    if grid_mode:
        x = x.reshape((-1, rows, 3, h, w))
        depth = depth.reshape((-1, rows, 1, h, w))
        x = x.permute((0, 3, 1, 4, 2))
        depth = depth.permute((0, 3, 1, 4, 2))
        x = x.reshape((-1, rows * w, 3))
        depth = depth.reshape((-1, rows * w, 1))

    else:
        pass

    return x, depth


def save_batch_sample_rgbd(image, path, depth=None, axis=True):
    """
    save images and depth plasma cmaps
    """
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(image)
    plt.subplot(2, 1, 2)
    plt.imshow(depth, cmap='plasma')
    if not axis:
        plt.axis("off")
    plt.savefig(path)
    plt.close()


def save_loss_graph(epoch_losses_d, epoch_losses_g, path, x=None, p=None):
    plt.figure(figsize=(8, 6))
    plt.title('loss')
    if x is not None:
        plt.plot(x, epoch_losses_d, '-', color='red', label='D')
        plt.plot(x, epoch_losses_g, '-', color='blue', label='G')
        if p is not None:
            plt.plot(x, p, '-', color='green', label='P')
        plt.xlabel('elapsed time')
        plt.ylabel('loss')
    else:
        plt.plot(epoch_losses_d, '-', color='red', label='D')
        plt.plot(epoch_losses_g, '-', color='blue', label='G')
        if p is not None:
            plt.plot(p, '-', color='green', label='P')
        plt.xlabel('iteration')
        plt.ylabel('loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()



