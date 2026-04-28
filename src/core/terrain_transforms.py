
import random
from functools import partial

import torch
from torchvision.transforms import functional as F
import numpy as np
from PIL import Image


class NormaliseTransform(torch.nn.Module):
    """ Normalise (0, 1) to (-1, 1) """

    def forward(self, img):
        return (img * 2) - 1


class UnnormaliseTransform(torch.nn.Module):
    """ Unnormalise (-1, 1) to (0, 1) """

    def forward(self, img):
        return (img + 1) * 0.5


rot90 = partial(torch.rot90, dims=(1, 2))  # rotate 90 anticlockwise
transforms = [
    torch.nn.Identity(),  # Do nothing
    torch.fliplr,  # horizontal flip
    torch.flipud,  # vertical flip
    rot90,
    partial(torch.rot90, k=2, dims=(1, 2)),  # rotate 180 anticlockwise
    # rotate 270 anticlockwise
    partial(torch.rot90, k=-1, dims=(1, 2)),
    [torch.fliplr, rot90],
    [torch.flipud, rot90]
]

class RandomTerrainTransform(torch.nn.Module):
    # allowed since flips are over the x or z axes (assuming y up),
    # as well as rotations, creating new, completely valid terrain
            
    def forward(self, img):
        transform = self.get_random_transform()
        return self.apply_transform(img, transform)

    @staticmethod
    def apply_transform(img, transform):
        if isinstance(transform, list):
            for t in transform:
                img = t(img)
        else:
            img = transform(img)

        return img

    @staticmethod
    def get_random_transform():
        return random.choice(transforms)


INTERPOLATION_MODES = {
    'adaptive': None,
    'nearest': F.InterpolationMode.NEAREST,
    'bicubic': F.InterpolationMode.BICUBIC,
}


class AdaptiveResizing(torch.nn.Module):
    """ Resize according to different rules, depending on input and output sizes """

    def __init__(self, size: int, interpolation_type='adaptive'):
        super().__init__()
        self.size = size
        self.interpolation_type = interpolation_type

    def forward(self, img: torch.Tensor):
        c, w, h = img.shape

        interpolation_mode = None
        if self.interpolation_type == 'adaptive':
            # Apply nearest interpolation if upscaling
            # Apply bicubic if downscaling by a significant amount
            if w - self.size <= 1 and h - self.size <= 1:
                interpolation_mode = F.InterpolationMode.NEAREST
            else:
                interpolation_mode = F.InterpolationMode.BICUBIC
        else:
            interpolation_mode = INTERPOLATION_MODES[self.interpolation_type]

        return F.resize(img, self.size, interpolation_mode)


def tensor_to_pil_image(pic, mode=None):

    if isinstance(pic, torch.Tensor):
        if pic.ndimension() not in {2, 3}:
            raise ValueError(
                f'pic should be 2/3 dimensional. Got {pic.ndimension()} dimensions.')

        elif pic.ndimension() == 2:
            # if 2D image, add channel dimension (CHW)
            pic = pic.unsqueeze(0)

        # check number of channels
        if pic.shape[-3] > 4:
            raise ValueError(
                f'pic should not have > 4 channels. Got {pic.shape[-3]} channels.')

    npimg = pic
    if pic.is_floating_point() and mode != 'F':
        pic = pic.mul(255).add_(0.5).clamp_(0, 255).byte()

    npimg = np.transpose(pic.cpu().numpy(), (1, 2, 0))

    if npimg.shape[2] == 1:
        expected_mode = None
        npimg = npimg[:, :, 0]
        if npimg.dtype == np.uint8:
            expected_mode = 'L'
        elif npimg.dtype == np.int16:
            expected_mode = 'I;16'
        elif npimg.dtype == np.int32:
            expected_mode = 'I'
        elif npimg.dtype == np.float32:
            expected_mode = 'F'
        if mode is not None and mode != expected_mode:
            raise ValueError(
                f'Incorrect mode ({mode}) supplied for input type {np.dtype}. Should be {expected_mode}')
        mode = expected_mode

    elif npimg.shape[2] == 2:
        permitted_2_channel_modes = ['LA']
        if mode is not None and mode not in permitted_2_channel_modes:
            raise ValueError(
                f'Only modes {permitted_2_channel_modes} are supported for 2D inputs')

        if mode is None and npimg.dtype == np.uint8:
            mode = 'LA'

    elif npimg.shape[2] == 4:
        permitted_4_channel_modes = ['RGBA', 'CMYK', 'RGBX']
        if mode is not None and mode not in permitted_4_channel_modes:
            raise ValueError(
                f'Only modes {permitted_4_channel_modes} are supported for 4D inputs')

        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGBA'
    else:
        permitted_3_channel_modes = ['RGB', 'YCbCr', 'HSV']
        if mode is not None and mode not in permitted_3_channel_modes:
            raise ValueError(
                f'Only modes {permitted_3_channel_modes} are supported for 3D inputs')
        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGB'

    if mode is None:
        raise TypeError(f'Input type {npimg.dtype} is not supported')

    return Image.fromarray(npimg, mode=mode)


class TensorToPILImage:
    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, pic):
        return tensor_to_pil_image(pic, self.mode)
