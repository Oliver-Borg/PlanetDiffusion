from math import ceil
import zlib
import re
from typing import List
from concurrent.futures import ThreadPoolExecutor, wait
import logging
import itertools
import os

import torch
import torch.nn.functional as F

import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig()
logger = logging.getLogger(__name__)


# https://stackoverflow.com/a/63495323
POLL_INTERVAL = 0.1
MAX_WORKERS = 16


class FunctionWrapper:
    def __init__(self, func, *args, **kwargs) -> None:
        self.func = func
        self.args = args
        self.kwargs = kwargs


def run_concurrent(functions: List[FunctionWrapper], num_workers=MAX_WORKERS):
    if not functions:
        logger.info('No jobs to run')
        return  # Do nothing if there are no functions

    logger.info(f'Setting up ThreadPoolExecutor for {len(functions)} jobs')
    with ThreadPoolExecutor(max_workers=num_workers) as pool, \
            tqdm(total=len(functions)) as progress:

        all_futures = (pool.submit(f.func, *f.args, **f.kwargs)
                       for f in functions)
        to_process = set(itertools.islice(all_futures, num_workers))
        try:
            while to_process:
                just_finished, to_process = wait(
                    to_process, timeout=POLL_INTERVAL)
                to_process |= set(itertools.islice(
                    all_futures, len(just_finished)))

                for d in just_finished:
                    progress.set_description(f'Processed {d.result()}')
                    progress.update()

        except KeyboardInterrupt:
            logger.info(
                'Gracefully shutting down: Cancelling unscheduled tasks')

            # only futures that are not done will prevent exiting
            for future in to_process:
                future.cancel()

            logger.info('Waiting for in-progress tasks to complete')
            wait(to_process, timeout=None)
            logger.info('Cancellation successful')
from scipy.signal import convolve2d

class NoiseLoss(torch.nn.Module):
    def __init__(self):
        super(NoiseLoss, self).__init__()

    def forward(self, inputs, targets, *args, **kwargs):
        reduction = kwargs.get('reduction', 'mean')
        mse_loss = F.mse_loss(inputs, targets, reduction='none')
        loss = batched_noise_estimate(mse_loss, reduction=reduction)
        mse_loss = F.mse_loss(inputs, targets, reduction='mean')
        loss = mse_loss / mse_loss.item() * loss

        return loss

def batched_noise_estimate(ims: torch.Tensor, reduction='mean') -> torch.Tensor:
    # Estimate noise for each image in batch
    assert ims.ndim == 4
    assert ims.shape[1] == 1
    to_return = []
    for j in range(ims.shape[0]):
        to_return.append(estimate_noise(ims[j].detach().cpu().squeeze().numpy()))
    to_return = torch.tensor(to_return).to(ims.device)
    if reduction == 'mean':
        return to_return.mean()
    elif reduction == 'sum':
        return to_return.sum()
    elif reduction == 'none':
        return to_return
    


def estimate_noise(I):
  H, W = I.shape

  M = [[1, -2, 1],
       [-2, 4, -2],
       [1, -2, 1]]

  sigma = np.sum(np.sum(np.absolute(convolve2d(I, M))))
  sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (W-2) * (H-2))

  return sigma

def list_all_files(d, extensions=None, full_path=False, recursive=True):
    """List all files in a directory"""
    def main_gen():
        if full_path:
            if recursive:
                for dir_path, _, files in os.walk(d):
                    for f in files:
                        yield dir_path, f
            else:
                for f in os.listdir(d):
                    yield d, f

        else:
            if recursive:
                for _, _, files in os.walk(d):
                    yield from files
            else:
                yield from os.listdir(d)

    if extensions is None:
        yield from main_gen()
    else:
        if not isinstance(extensions, (tuple, list)):
            extensions = [extensions]

        if full_path:
            for d, f in main_gen():
                if f.split('.')[-1] in extensions:
                    yield d, f
        else:
            for f in main_gen():
                if f.split('.')[-1] in extensions:
                    yield f


def get_tiles_list(folder, ext, recursive=True):
    return set(map(lambda x: x.split('.')[0], list_all_files(folder, ext, recursive=recursive)))


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def regex_search(text, pattern, group=1, default=None):
    match = re.search(pattern, text)
    return match.group(group) if match else default


CHECKPOINT_REGEX = r'checkpoint-(\d+)'


def list_checkpoints(model_dir):
    checkpoints = {}
    for f in os.listdir(model_dir):
        checkpoint_dir = os.path.join(model_dir, f)
        if not os.path.isdir(checkpoint_dir):
            continue

        match = regex_search(f, CHECKPOINT_REGEX)
        if match is None:
            continue  # No match, ignore

        l = int(match)
        checkpoints[l] = checkpoint_dir

    return checkpoints


def get_latest_checkpoint(model_dir):
    if not os.path.exists(model_dir):
        return None

    checkpoints = list_checkpoints(model_dir)

    if not checkpoints:
        return None

    return checkpoints[max(checkpoints)]


def split_list(arr, ratios):
    """Split array according to ratios. Sum of ratios should be <= 1"""
    to_return = []
    offsets = [0] + list(map(lambda x: ceil(x * len(arr)), np.cumsum(ratios)))
    for offset_from, offset_to in zip(offsets[:-1], offsets[1:]):
        to_return.append(arr[offset_from:offset_to])
    return to_return


def roll_y(grid, amount):
    return np.roll(grid, amount, axis=0)


def roll_x(grid, amount):
    return np.roll(grid, amount, axis=1)


def roll_left(grid):
    a = roll_x(grid, -1)
    a[:, -1] = np.nan
    return a


def roll_right(grid):
    a = roll_x(grid, 1)
    a[:, 0] = np.nan
    return a


def roll_up(grid):
    a = roll_y(grid, -1)
    a[-1, :] = np.nan
    return a


def roll_down(grid):
    a = roll_y(grid, 1)
    a[0, :] = np.nan
    return a


def load_numpy(input_file):
    # TODO optimise?
    try:
        if input_file.endswith('.npz'):
            return np.load(input_file)['arr_0']
        elif input_file.endswith('.npy'):
            return np.load(input_file)

        raise Exception(f'Invalid numpy file: {input_file}')

    except zlib.error as e:
        raise Exception(f'Error opening "{input_file}" ({e})')


def mkdirs(f):
    os.makedirs(os.path.dirname(f), exist_ok=True)


def save_numpy(output_file, arr, compressed=True):
    mkdirs(output_file)
    try:
        if compressed:
            np.savez_compressed(output_file, arr)
        else:
            np.save(output_file, arr)
    except KeyboardInterrupt:
        try:
            os.remove(output_file)
        except OSError:
            pass
        raise


def save_image(path, img, **kwargs):
    mkdirs(path)
    # Helper method to save PIL image
    to_remove = False
    try:
        img.save(path, **kwargs)
    except KeyboardInterrupt:
        to_remove = True

    if to_remove:
        try:
            os.remove(path)
            print(f'Removed image "{path}" due to KeyboardInterrupt')
        except OSError as e:
            print(f'Unable to remove image "{path}" ({e})')

        raise KeyboardInterrupt


def normalise_image(img):
    scale = None
    if img.mode in ('L', 'LA', 'P', 'PA', 'RGB', 'RGBA'):
        scale = 2**8 - 1
    elif img.mode == 'I':
        _, _, _, mode = img.tile[0]

        if mode in ('I;16', 'I;16L', 'I;16B', 'I;16N'):
            scale = 2**16 - 1
        else:
            scale = 2**32 - 1
    else:
        raise Exception(f'Unsupported image mode: {img.mode}')

    return np.array(img, dtype=np.float32)/scale


def load_image(path, convert_to=None):
    # Helper method to load image into numpy.ndarray (H x W x C) with values
    # between 0 and 1, accounting for bit-depth
    # https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes
    img = Image.open(path)
    if exists(convert_to):
        img = img.convert(convert_to)

    return normalise_image(img)


def array_to_image(arr, bit_depth=8):
    if bit_depth == 16:
        max_val = 65535
        dtype = np.uint16
        mode = 'I;16'
    else:  # assume 8
        max_val = 255
        dtype = np.uint8
        mode = 'L'
        if len(arr.shape) == 3:
            arr = arr[:, :, 0]  # Take first channel

    arr = (max_val * arr + 0.5).clip(0, max_val).astype(dtype)
    return Image.fromarray(arr, mode=mode)


def create_image_grid(image_grid):
    # image_grid is a list of lists of images

    num_rows = len(image_grid)
    num_cols = -1
    w, h = -1, -1
    for image_row in image_grid:
        num_cols = max(num_cols, len(image_row))
        for image in image_row:
            im_w, im_h = image.size
            w = max(w, im_w)
            h = max(h, im_h)

    final_image = Image.new('RGBA', size=(w * num_cols, h*num_rows))

    for row, image_row in enumerate(image_grid):
        for col, image in enumerate(image_row):
            final_image.paste(image, (col * w, row * h))

    return final_image


def pad_or_crop_tensor(tensor: torch.Tensor, dimension: int, desired_num_values: int, pad_value: float = 0.0, pad_direction='left'):
    # Helper method to pad or crop 1D tensor
    actual_num_values = tensor.shape[dimension]

    if desired_num_values == actual_num_values:
        return tensor  # Already correct, do nothing

    if desired_num_values < actual_num_values:
        tensor = torch.narrow(
            tensor, dimension, 0, desired_num_values)

    else:  # desired_num_values > actual_num_values
        # Pad tensor (to the left or right) with pad_value
        # https://stackoverflow.com/questions/48686945/reshaping-a-tensor-with-padding-in-pytorch
        padding = [0] * (tensor.ndim*2)
        padding_dim = 2*dimension

        if pad_direction == 'left':
            padding_dim += 1

        elif pad_direction == 'right':
            pass
        else:
            raise ValueError(f'Unknown pad_direction: "{pad_direction}"')

        padding[padding_dim] = desired_num_values - actual_num_values

        tensor = F.pad(
            input=tensor,
            pad=tuple(reversed(padding)),
            mode='constant',
            value=pad_value
        )

    return tensor


def normalise_array(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)

    height_range = max_val - min_val
    if height_range > 0:
        arr = (arr - min_val) / height_range  # Normalise 0 to 1
    else:
        # All values are the same, set to 0
        arr = np.zeros_like(arr)
    return arr


def exists(x):
    return x is not None


def exactly_one_exists(a, b):
    return exists(a) ^ exists(b)


def tile_images(list_of_batched_images):
    """
    Helper method to tile images with varying number of channels

    each item in the list is a batch of images, of the form: (b, c, h, w)

    """

    to_return = []

    most_channels = 3 # Easier to display in wandb etc
    for batch in list_of_batched_images:
        most_channels = max(most_channels, batch.shape[1])

    if most_channels >= 5:
        raise ValueError(
            f'Number of channels ({most_channels}) is not between 1 and 4.')

    for batch in list_of_batched_images:
        batch_size, c, h, w = batch.shape

        if c != most_channels:  # Must do something
            if most_channels == 4:  # RGBA
                shape = (batch_size, 1, h, w)
                if c == 1:  # Convert grayscale to RGBA
                    # Duplicate across all channels except last (alpha=1)
                    batch = torch.cat(
                        [batch] * 3 +
                        [torch.ones(shape, device=batch.device,
                                    dtype=batch.dtype)],
                        dim=1
                    )
                else:  # Convert other to RGBA
                    # Fill middle channels with -1
                    # Set alpha to 1
                    batch = torch.cat(
                        [batch] +
                        [torch.full(shape, fill_value=-1, device=batch.device, dtype=batch.dtype)] * (most_channels - c - 1) +
                        [torch.ones(shape, device=batch.device,
                                    dtype=batch.dtype)],
                        dim=1
                    )

            elif most_channels == 3:  # RGB
                # Duplicate across all channels
                shape = (batch_size, 1, h, w)
                if c == 1:  # Convert grayscale to RGB
                    batch = torch.cat([batch] * 3, dim=1)  # channel-wise
                else:
                    batch = torch.cat(
                        [batch] +
                        [torch.full(shape, fill_value=-1, device=batch.device, dtype=batch.dtype)] *
                        (most_channels - c),
                        dim=1
                    )

            elif most_channels == 2:  # LA
                # Set alpha to 1
                # c must be 1
                shape = (batch_size, 1, h, w)
                batch = torch.cat(
                    [batch] +
                    [torch.ones(shape, device=batch.device,
                                dtype=batch.dtype)] *
                        (most_channels - c),
                    dim=1
                )

            else:  # most_channels == 1:  # L
                pass  # Do nothing (can't ever happen)

        to_return.append(batch)

    return to_return


def format_memory(bytes):
    return f'{bytes/1024**3:.1f} GiB'
