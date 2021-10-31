#
# save_images.py
# functions for generated sample images
#

import torch


def generate_sample(gen, fixed_z, test_y_rotate, angle_range=8, num_sample=4, device=torch.device("cuda")):

    latent_z = fixed_z[:num_sample]
    latent_z = torch.tile(latent_z[:, None], (1, angle_range) +
                          (1,) * (latent_z.ndim - 1)).reshape(num_sample * angle_range, *latent_z.shape[1:])
    theta = torch.zeros((num_sample * angle_range, 6), dtype=torch.float32).to(device)
    theta[:, 1] = torch.tile(torch.linspace(-test_y_rotate, test_y_rotate, angle_range), (num_sample, ))
    theta = torch.reshape(
        torch.cat([torch.cos(theta[:, :3]), torch.sin(theta[:, :3]), theta[:, 3:]], dim=1),
        (theta.shape[0], 9, 1, 1)
    )

    sample_x = gen(latent_z, theta=theta)
    sample_x = convert_batch_images(sample_x, num_sample, angle_range)

    return sample_x


def convert_batch_images(x, rows, cols):
    """
    convert batch sample images to one grid
    """
    x = x.cpu()
    depth = torch.tile(x[:, -1], (1, 3, 1, 1))

    x = x[:, :-1]
    _, _, h, w = x.shape
    x = x.reshape((rows, cols, 3, h, w))

    depth = 1 / (depth + 1e-8)
    depth = depth.reshape((rows, cols, 3, h, w))
    x = torch.cat([x, depth], dim=1).reshape((rows * 2, cols, 3, h, w))

    x = x.permute((0, 3, 1, 4, 2))
    x = x.reshape((-1, cols * w, 3))

    return x
