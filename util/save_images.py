#
# save_images.py
# functions for generated sample images
#

import numpy as np


def convert_batch_images(x, rows, cols):
    """
    convert batch sample images to one grid
    """
    x = x.cpu()
    rgbd = False
    if x.shape[1] == 4:
        rgbd = True
        depth = np.tile(x[:, -1:], (1, 3, 1, 1))
        x = x[:, :-1]

    x = np.asarray(np.clip(x.data * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
    _, _, h, w = x.shape
    x = x.reshape((rows, cols, 3, h, w))
    if rgbd:
        depth = np.array(np.clip(1 / depth * 128, 0.0, 255.0), dtype=np.uint8)
        depth = depth.reshape((rows, cols, 3, h, w))
        x = np.concatenate([x, depth], axis=1).reshape((rows * 2, cols, 3, h, w))

    x = x.transpose((0, 3, 1, 4, 2))
    x = x.reshape((-1, cols * w, 3))
    return x
