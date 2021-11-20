#
# warp.py
# functions for warping RGBD images
#

import torch


def warp(K, inv_K, R, t, d, p):
    """
    calculate projected depth*pixel in other camera
    """
    new_dp = torch.matmul(torch.matmul(torch.matmul(K, R), inv_K), d * p) - torch.matmul(torch.matmul(K, R), t)
    return new_dp.transpose(2, 1)


def inv_warp(K, inv_K, inv_R, t, d, p):
    """
    calculate projected depth*pixel in other camera
    inverse rotation and translation are applied in function 'warp'
    """
    new_dp = torch.matmul(torch.matmul(torch.matmul(K, inv_R), inv_K), d * p) + torch.matmul(K, t)
    return new_dp.transpose(2, 1)


def bilinear_sampling(img, dp):
    """
    warp rgbd images using projected depth * pixel and bilinear sampling
    """
    b, hw, _ = dp.shape
    _, _, h, w = img.shape
    dp = dp.reshape(-1, 3) # reshape dp to (b*h*w, 3)

    # homogeneous coord (wx, wy, w) -> real coord (x, y, 1)
    # restrict depth not to negative value
    u = dp[:, 0] / torch.clip(dp[:, 2], 1e-4, 10000)
    v = dp[:, 1] / torch.clip(dp[:, 2], 1e-4, 10000)

    # in deepvoxel, x, y is opposite (ignore this comment)
    u, v = v, u

    u0 = u.type(torch.int32)
    u1 = u0 + 1
    v0 = v.type(torch.int32)
    v1 = v0 + 1

    # define weights
    w1 = (u1 - u) * (v1 - v)
    w2 = (u - u0) * (v1 - v)
    w3 = (u1 - u) * (v - v0)
    w4 = (u - u0) * (v - v0)

    # make image coord for all images in batch size
    img_coord = torch.divide(torch.arange(b * hw), hw, rounding_mode='floor').type(torch.long)

    # find a point that is not in out-of-grid section after warping
    not_out = (u >= 0) * (u < h - 1) * (v >= 0) * (v < w - 1) * (dp[:, 2] > 1e-4)

    # make out-points to 0 using not_out
    u0 = (u0 * not_out).type(torch.long)
    u1 = (u1 * not_out).type(torch.long)
    v0 = (v0 * not_out).type(torch.long)
    v1 = (v1 * not_out).type(torch.long)
    w1 = (w1 * not_out)
    w2 = (w2 * not_out)
    w3 = (w3 * not_out)
    w4 = (w4 * not_out)

    # bilinear sampling
    warped = w1[:, None] * img[img_coord, :, u0, v0] + w2[:, None] * img[img_coord, :, u1, v0] + \
             w3[:, None] * img[img_coord, :, u0, v1] + w4[:, None] * img[img_coord, :, u1, v1]

    return warped, not_out


