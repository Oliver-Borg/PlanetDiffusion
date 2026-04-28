import numpy as np
import torch
from ...core.utils import array_to_image


def preprocess_image(img, num_channels=3):
    img = 2 * img - 1
    if img.ndim == 2:  # (h, w) to (h, w, 3)
        # img = np.expand_dims(img, axis=2)
        img = np.dstack([img]*num_channels)
    if img.shape[2] > num_channels:
        img = img[:, :, -num_channels:] # Take last n channels to include rivers
    while img.shape[2] < num_channels:
        img = np.dstack([img, np.zeros_like(img[:, :, 0])])

    img = np.transpose(img, (2, 0, 1))  # (h, w, c) -> (c, h, w)

    return torch.from_numpy(img).unsqueeze(0)


def tensor_to_img(x, bit_depth):
    x = x.cpu().numpy()
    x = np.transpose(x, (1, 2, 0))  # (c, h, w) -> (h, w, c)
    return array_to_image(x, bit_depth=bit_depth)
