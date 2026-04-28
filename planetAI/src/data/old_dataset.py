try:
    from utils import PlanetConfig, profile, np_hex_to_rgb, np_rgb_to_hex
    from map_paster import gen_list, get_next_tile_pair, setup, change_channels, gen_tile_pair, get_tile, get_translated_coords, get_mask_config, get_k_val, _gen_image, _gen_full_image
    from utils import PlanetConfig, masking, np_rgb, overlap, open_image_array, create_mask, continuous_to_spread, timing, modal_resize, gray_to_land, slow_modal_resize, create_overlap_mask
    from sketch_gen import landcover_paint, dilate_paint, temperature_paint, get_buckets
    from dataclass_argparser import CustomArgumentParser
    from data_creator import cv2_rotate, nearest_rotate
    from modal_sketch import ModalSketch
    from sphere_mapping import SphereMapping
except:
    from .utils import PlanetConfig, profile, np_hex_to_rgb, np_rgb_to_hex
    from .map_paster import gen_list, get_next_tile_pair, setup, change_channels, gen_tile_pair, get_tile, get_translated_coords, get_mask_config, get_k_val, _gen_image, _gen_full_image
    from .utils import PlanetConfig, masking, np_rgb, overlap, open_image_array, create_mask, continuous_to_spread, timing, modal_resize, gray_to_land, slow_modal_resize, create_overlap_mask
    from .sketch_gen import landcover_paint, dilate_paint, temperature_paint, get_buckets
    from .dataclass_argparser import CustomArgumentParser
    from .data_creator import cv2_rotate, nearest_rotate
    from .modal_sketch import ModalSketch
    from .sphere_mapping import SphereMapping

from torch.utils.data import Dataset, DataLoader
from time import time
from torchvision import transforms
import torch
import numpy as np
import random
from cv2 import resize, INTER_NEAREST, dilate, erode, INTER_AREA
import cv2
from PIL import Image as img
import diffusers
from typing import Union, List
from dataclasses import replace
from tqdm import tqdm
from dataclasses import dataclass, field
import os
from math import ceil
from diffusers.models import AutoencoderKL

def dropout(img: np.ndarray, p=0.1, seed=None):
    '''
    Randomly replace the entire image with black pixels
    '''
    if random.Random(seed).random() < p:
        return np.zeros_like(img)
    return img

class NormaliseTransform(torch.nn.Module):
    """ Normalise (0, 1) to (-1, 1) """

    def forward(self, img):
        return (img * 2) - 1

class RandomMaskTransform(torch.nn.Module):
    def __init__(self, channels: int = 1, generator: torch.Generator = None):
        super().__init__()
        self.channels = channels
        self.generator = generator
    def forward(self, img):
        img[:self.channels] = masking(img[:self.channels], self.generator)
        return img
    
class ZeroTensorTransform(torch.nn.Module):
    """ Return a tensor of -1.0 (zero) """

    def forward(self, img):
        return torch.fill(img, -1.0)
    
class NoiseTransform(torch.nn.Module):
    """ Return a random tensor """

    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def forward(self, img):
        noise = diffusers.utils.torch_utils.randn_tensor(img.shape, generator=self.generator)
        return noise
    
class NormalisedNoiseTransform(torch.nn.Module):
    """ Return a random tensor """

    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def forward(self, img):
        noise = diffusers.utils.torch_utils.randn_tensor(img.shape, generator=self.generator)
        normalised_noise = noise - noise.min()
        normalised_noise = normalised_noise / normalised_noise.max() * 2 - 1
        return normalised_noise

class ScalingTransform(torch.nn.Module):
    """ Scale the image by a factor """

    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, img):
        img[0] *= self.factor
        return img
    
class InvertTransform(torch.nn.Module):
    """ Invert the image """

    def forward(self, img):
        return -img

class RAMDataset(Dataset):
    def __init__(self, 
                 planet_cfg: PlanetConfig,
                 target_image_channels: int=None,
                 cond_image_channels: int=None,
                 normalise: bool=True,
                 conditioning_dropout: float=0.0,
                 tile_size: int=256,
                 mode: str='train',
                 auto_encoder: AutoencoderKL | None = None) -> None:
        super().__init__()
        # Here we want to load the full untranslated images into memory
        # We can then take a random crop from each image (twice as large as the tile size)
        # This can be rotated and flipped then cropped to the tile size
        self.max_angle = 15 # TODO Maybe increase this
        self.planet_cfg = planet_cfg
        # setup(planet_cfg)
        self.seed = planet_cfg.planet_seed
        self.tile_size = tile_size
        self.normalise = normalise
        self.target_image_channels = target_image_channels if target_image_channels is not None else planet_cfg.output_channels()
        self.cond_image_channels = cond_image_channels if cond_image_channels is not None else planet_cfg.input_channels()
        self.generator = torch.Generator(device='cpu')
        if self.seed is not None:
            self.generator.manual_seed(self.seed)
        self.auto_encoder = auto_encoder

        cond_transforms = [
            # Convert Image or numpy array to tensor
            transforms.ToTensor()
        ]

        target_transforms = [
            # Convert Image or numpy array to tensor
            transforms.ToTensor()
        ]

        if normalise:
            cond_transforms.append(NormaliseTransform())
            target_transforms.append(NormaliseTransform())

        if self.planet_cfg.inpainting_channels > 0:
            cond_transforms.append(RandomMaskTransform(self.planet_cfg.inpainting_channels, self.generator))


        self.cond_transform = transforms.Compose(cond_transforms)
        self.target_transform = transforms.Compose(target_transforms)

        H = 256 * 2 ** planet_cfg.size
        W = 512 * 2 ** planet_cfg.size
        self.sphere_mapping = SphereMapping(shape=(H, W))
        h = 256 * 2 ** (planet_cfg.size - planet_cfg.downscale_offset)
        w = 512 * 2 ** (planet_cfg.size - planet_cfg.downscale_offset)
        h = int(h)
        w = int(w)
        self.H = H
        self.W = W
        self.h = H
        self.w = w

        self.size = planet_cfg.size
        self.downscale_offset = planet_cfg.downscale_offset
        self.delta = 2**self.downscale_offset
        self.down_tile_width = self.tile_size // self.delta

        self.dem = open_image_array(os.path.join(planet_cfg.data_dir, f'World_DEM_{W}x{H}.png'))
        self.land = open_image_array(os.path.join(planet_cfg.data_dir, f'World_LandCover_{W}x{H}.png'))
        self.sat = open_image_array(os.path.join(planet_cfg.data_dir, f'world.satellite.{W}x{H}.png'))
        self.temp = open_image_array(os.path.join(planet_cfg.data_dir, f'World_Temp_{W}x{H}.png'))
        self.downdem = resize(self.dem, (w, h), interpolation=INTER_AREA)
        # self.downland = open_image_array(os.path.join(planet_cfg.data_dir, f'World_LandCover_{w}x{h}.png'))
        self.downland = modal_resize(self.land, self.delta)
        self.downsat = resize(self.sat, (w, h), interpolation=INTER_AREA)
        self.downtemp = resize(self.temp, (w, h), interpolation=INTER_AREA)

        self.buckets = get_buckets(self.downdem, self.planet_cfg) if self.planet_cfg.bucketing_mode == 'uniform' else None
        self.downsketch = dilate_paint(self.downdem, planet_cfg.downscale_cfg, buckets=self.buckets)
        self.downland_sketch = landcover_paint(self.downland, planet_cfg.downscale_cfg)
        self.downland_sketch[self.downsketch == 0] = 0

        if 'modal' in planet_cfg.input_types+planet_cfg.output_types or\
            'downmodal' in planet_cfg.input_types+planet_cfg.output_types:
            self.modal_sketch = ModalSketch(planet_cfg, temp=self.temp, land=self.land, sat=self.sat)


        # self.temp[self.temp == 0] = 1
        self.temp[self.land == 0] = 0
        self.temp[(self.land > 0) & (self.temp < 255)] += 1

        # Do some extra processing on the temp image to make up for the resizing 
        # causing the edges to be colder than they are supposed to be
        self.temp_sketch = temperature_paint(self.temp, planet_cfg) 
        #                                                   replace(planet_cfg.downscale_cfg, 
        #                                                         dilate_iters=2, erode_iters=1, dilate_first=True,
        #                                                         erode_size=3, dilate_size=3, preserve_edges=True)
        # self.downtemp[self.downtemp == 0] = 1
        self.downtemp[self.downland == 0] = 0
        self.downtemp[(self.downland > 0) & (self.downtemp < 255)] += 1
        self.downtemp_sketch = temperature_paint(self.downtemp, planet_cfg)
        self.downtemp_sketch[self.downland == 0] = 0

        all_mask = np.dstack([self.downsketch > 0, self.downland_sketch > 0, self.downtemp > 0]).astype(np.uint8)*255

        # pth = os.path.join(planet_cfg.data_dir, 'debug')
        # os.makedirs(pth, exist_ok=True)
        # img.fromarray(self.downtemp_sketch).save(os.path.join(pth, 'downtemp_sketch.png'))
        # img.fromarray(self.downsat).save(os.path.join(pth, 'downsat.png'))
        # img.fromarray(gray_to_land(self.downland_sketch)).save(os.path.join(pth, 'downland_sketch.png'))
        # img.fromarray(self.downsketch).save(os.path.join(pth, 'downsketch.png'))

        # TODO: Investigate empty tiles
        tile_mask = dilate(self.downsketch, kernel=np.ones((3, 3)), iterations=0)
        # This prevents tiles (and neighbouring embeddings) that go off the bottom/top of the image
        # Also there is a mismatch between satellite imagery and others from y 0 to y 455 
        tile_mask[:self.down_tile_width//2 + ceil(455/self.delta), :] = 0
        # tile_mask[-self.down_tile_width//2:, :] = 0
        # This prevents tiles that go off the left/right of the image 
        # This shouldn't be necessary, but it is since the satellite imagery doesn't align perfectly
        # tile_mask[:, :self.down_tile_width//2] = 0
        # tile_mask[:, -self.down_tile_width//2:] = 0
        if self.planet_cfg.sphere_sampling:
            ys, xs = self.sphere_mapping.get_distributed_points(tile_mask)
        else:   
            ys, xs = np.where(tile_mask != 0)
            ys -= self.down_tile_width//2
            xs -= self.down_tile_width//2
        self.valid_tiles = list(zip(ys, xs))
        random.Random(0).shuffle(self.valid_tiles)
        # self.valid_tiles = self.valid_tiles[:int(len(self.valid_tiles)*0.01)]
        if mode == 'train':
            self.valid_tiles = self.valid_tiles[:int(len(self.valid_tiles)*0.9)]
        elif mode == 'val':
            self.valid_tiles = self.valid_tiles[int(len(self.valid_tiles)*0.9):]
        elif mode == 'test':
            self.valid_tiles = self.valid_tiles[int(len(self.valid_tiles)*0.9):]


        # Save all downsketches
        img.fromarray(self.downsketch).save(os.path.join(planet_cfg.data_dir, 'downsketch.png'))
        img.fromarray(gray_to_land(self.downland_sketch)).save(os.path.join(planet_cfg.data_dir, 'downland_sketch.png'))
        img.fromarray(self.downtemp_sketch).save(os.path.join(planet_cfg.data_dir, 'downtemp_sketch.png'))
        if hasattr(self, 'modal_sketch'):
            self.downmodal_sketch = self.modal_sketch.get_sketch(self.downland, self.downtemp_sketch)
            img.fromarray(self.downmodal_sketch).save(os.path.join(planet_cfg.data_dir, 'downmodal_sketch.png'))
    def __len__(self):
        return len(self.valid_tiles)
    

    @profile
    def __getitem__(self, idx):
        start_time = time()
        y, x = self.valid_tiles[idx]
        unrotated = get_tile(self.downsketch, y, x, self.down_tile_width)
        assert unrotated.sum() > 0, f"Empty tile: {y} {x}"
        oy = y + self.down_tile_width//2
        ox = x + self.down_tile_width//2
        max_angle = self.max_angle / 180 * np.pi
        # Rotate x, y about ox, oy by max_angle
        y_t = y
        x_t = x
        y_b = y + self.down_tile_width
        x_b = x + self.down_tile_width
        new_y_t = oy + (x_t - ox) * np.sin(max_angle) + (y_t - oy) * np.cos(max_angle)
        new_y_b = oy + (x_b - ox) * np.sin(max_angle) + (y_b - oy) * np.cos(max_angle)
        while (new_y_t < 0 or new_y_b > self.h) and max_angle > 0:
            max_angle -= 1 / 180 * np.pi
            new_y_t = oy + (x_t - ox) * np.sin(max_angle) + (y_t - oy) * np.cos(max_angle)
            new_y_b = oy + (x_b - ox) * np.sin(max_angle) + (y_b - oy) * np.cos(max_angle)
        max_angle *= 180 / np.pi
        max_angle = int(max_angle)
        ang = np.random.randint(-max_angle, max_angle) if max_angle > 0 else 0
        if ang == 0:
            ang += 1

        #######
        # ang *= 0
        ####### REMOVE TODO
        crop_y = y - self.down_tile_width//2
        crop_x = x - self.down_tile_width//2
        center = (float(self.tile_size), float(self.tile_size)) # TODO: Investigate + 0.5
        masks = [None, None, None, None]
        hflip = random.randint(0, 1)
        vflip = random.randint(0, 1)
        for i, mask in enumerate([self.dem, self.land, self.sat, self.temp_sketch]):
            masks[i] = get_tile(mask, crop_y*self.delta, crop_x*self.delta, 2*self.tile_size)
            if hflip:
                masks[i] = np.fliplr(masks[i]).copy()
            if vflip:
                masks[i] = np.flipud(masks[i]).copy()
            masks[i] = cv2_rotate(masks[i], ang, center) if i in [0, 2] \
                else nearest_rotate(masks[i], ang, center)
            masks[i] = get_tile(masks[i], self.tile_size//2, self.tile_size//2, self.tile_size)
        dem, land, sat, temp = masks

        any_zero = (dem == 0) | (land == 0) | (temp == 0)

        # dem[any_zero] = 0
        # land[any_zero] = 0
        # temp[any_zero] = 0

        downsketch = self.downsketch.copy()
        downland_sketch = self.downland_sketch.copy()
        downtemp_sketch = self.downtemp_sketch.copy()
        if hflip:
            downsketch = np.fliplr(downsketch).copy()
            downland_sketch = np.fliplr(downland_sketch).copy()
            downtemp_sketch = np.fliplr(downtemp_sketch).copy()
        if vflip:
            downsketch = np.flipud(downsketch).copy()
            downland_sketch = np.flipud(downland_sketch).copy()
            downtemp_sketch = np.flipud(downtemp_sketch).copy()

        mask_ratio = 1.0
        condition = []
        combined_sketch = np.zeros((self.tile_size, self.tile_size))

        for mask_type in self.planet_cfg.input_types:
            if 'mask' == mask_type:
                # TODO : Add mask size and coastal info
                iw = self.planet_cfg.inpainting_width
                # mask = create_mask(np.zeros((self.tile_size, self.tile_size)), iw, iw)
                mask = create_overlap_mask((self.tile_size, self.tile_size))
                if random.random() < self.planet_cfg.context_dropout:
                    mask[:, :] = 255
                mask_ratio = (mask > 0).mean() 
            elif mask_type == 'downland_sketch':
                mask = modal_resize(land, self.delta)
                mask = landcover_paint(mask, self.planet_cfg.downscale_cfg)
                mask = resize(mask, (self.tile_size, self.tile_size), interpolation=INTER_NEAREST)
                lc = self.planet_cfg.landcover_classes
                if self.planet_cfg.spread_landcover:
                    mask = continuous_to_spread(mask, lc)
                combined_sketch *= (lc+1)
                combined_sketch += (mask + 1) // (255//lc)
            elif mask_type == 'downsketch':
                mask = dilate_paint(dem, self.planet_cfg.downscale_cfg, buckets=self.buckets)
                mask = modal_resize(mask, self.delta)
                mask = resize(mask, (self.tile_size, self.tile_size), interpolation=INTER_NEAREST)
                dc = self.planet_cfg.colours
                combined_sketch *= (dc+1)
                combined_sketch += ((mask.astype(np.uint16) + 1) // (256//dc)).astype(np.uint8)
            elif mask_type == 'downtemp_sketch':
                mask = modal_resize(temp, self.delta)
                mask = resize(mask, (self.tile_size, self.tile_size), interpolation=INTER_NEAREST)
                combined_sketch *= (self.planet_cfg.temp_classes+1)
                combined_sketch += mask
            elif mask_type == 'modal':
                mask = modal_resize(temp, self.delta)
                mask = resize(mask, (self.tile_size, self.tile_size), interpolation=INTER_NEAREST)
                _downtemp_sketch = mask
                mask = self.modal_sketch.get_sketch(land, _downtemp_sketch)
            elif mask_type == 'downmodal':
                _downland = modal_resize(land, self.delta)
                _downtemp = modal_resize(temp, self.delta)
                missing = (_downland > 0) & (_downtemp == 0) 
                _temps, _temp_counts = np.unique(_downtemp, return_counts=True)
                if _temps.shape[0] > 1 and _temps[0] == 0:
                    _temps = _temps[1:]
                    _temp_counts = _temp_counts[1:]
                    _downtemp[missing] = _temps[np.argmax(_temp_counts)]
                mask = self.modal_sketch.get_sketch(_downland, _downtemp)
                mask = resize(mask, (self.tile_size, self.tile_size), interpolation=INTER_NEAREST)
                tc = self.planet_cfg.temp_classes
                combined_sketch *= (tc+1)
                combined_sketch += (resize(_downtemp, (self.tile_size, self.tile_size), interpolation=INTER_NEAREST)+1) // (256//tc)
                lc = self.planet_cfg.landcover_classes
                combined_sketch *= (lc+1)
                combined_sketch += resize(_downland, (self.tile_size, self.tile_size), interpolation=INTER_NEAREST) // (256//lc)
            elif mask_type == 'downsat_sketch':
                sat_hex = np_rgb_to_hex(sat)  
                mask = modal_resize(sat_hex, 32, exclude_zeros=True, is_hex=True)
                mask = np_hex_to_rgb(mask)
                mask = resize(mask, (self.tile_size, self.tile_size), interpolation=INTER_NEAREST)
            elif mask_type == 'dem':
                mask = dem
            elif mask_type == 'land':
                mask = land
                if self.planet_cfg.spread_landcover:
                    mask = continuous_to_spread(mask, self.planet_cfg.landcover_classes)
            elif mask_type == 'sat':
                mask = sat
            elif mask_type == 'temp':
                mask = temp
            else:
                raise ValueError(f"Unknown input mask type {mask_type}")
            condition.append(mask)
        combined_sketch += 1

        target = []
        for mask_type in self.planet_cfg.output_types:
            if mask_type == 'dem':
                mask = dem
            elif mask_type == 'land':
                mask = land
                if self.planet_cfg.spread_landcover:
                    mask = continuous_to_spread(mask, self.planet_cfg.landcover_classes)
            elif mask_type == 'sat':
                mask = sat
            elif mask_type == 'temp':
                mask = temp
            else:
                raise ValueError(f"Unknown output mask type {mask_type}")
            target.append(mask)
        
        condition = np.dstack(condition)
        target = np.dstack(target)
        

        assert len(condition.shape) == 3
        assert len(target.shape) == 3
        assert condition.shape[2] >= self.cond_image_channels
        assert target.shape[2] == self.target_image_channels

        down_sketch_index = self.planet_cfg.input_index('downsketch')
        downland_sketch_index = self.planet_cfg.input_index('downland_sketch')
        downtemp_sketch_index = self.planet_cfg.input_index('downtemp_sketch')

        if down_sketch_index > -1 and downland_sketch_index > -1:
            condition[:, :, downland_sketch_index][condition[:, :, down_sketch_index] == 0] = 0
        if down_sketch_index > -1 and downtemp_sketch_index > -1:
            condition[:, :, downtemp_sketch_index][condition[:, :, down_sketch_index] == 0] = 0
        
        if self.planet_cfg.replace_sketch:
            mask_index = self.planet_cfg.input_index('mask')
            mask = condition[:, :, mask_index]
            start_i = mask_index + 1
            for i in range(self.target_image_channels):
                condition[:, :, start_i+i][mask == 0] = target[:, :, i][mask == 0]

        # TODO: Colour dropout
        to_return = {}
        metadata = {
            'range': np.max(target) - np.min(target),
            'zoom': 2**self.planet_cfg.size,
            'tile_y': y*self.delta,
            'tile_x': x*self.delta,
            'vflip': vflip,
            'hflip': hflip,
            'k': 0,
            'factor': self.planet_cfg.size,
            'resolution': 0, #84375.0/(2**self.planet_cfg.size),
            'idx': idx,
            'tile_size': self.tile_size,
            'mask_channel': self.planet_cfg.input_index('mask'),
            'mask_ratio': mask_ratio,
        }
        to_return['target_image'] = self.target_transform(target.astype(np.float32)/255.0)
        to_return['cond_image'] = self.cond_transform(condition.astype(np.float32)/255.0)
        to_return['metadata'] = metadata
        to_return['combined_sketch'] = combined_sketch.astype(np.int16)
        if self.auto_encoder is not None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            to_return['original_target_image'] = to_return['target_image'].clone()
            y = to_return['target_image'].to(device).unsqueeze(0)
            if self.auto_encoder.config['in_channels'] == 3:
                sat_targets = y[:, 0:3]
                dem_targets = y[:, 3:4].repeat(1, 3, 1, 1)
                sat_latents = self.auto_encoder.encode(sat_targets).latent_dist.sample()
                dem_latents = self.auto_encoder.encode(dem_targets).latent_dist.sample()
                latent = torch.cat([sat_latents, dem_latents], dim=1)
            elif self.auto_encoder.config['in_channels'] == 4:
                latent = self.auto_encoder.encode(y).latent_dist.sample()
            else:
                raise ValueError('Autoencoder must have 3 or 4 input channels')

            to_return['target_image'] = latent.squeeze(0)


        # print(f"Time to load image: {time() - start_time:.2f}")
        start_time = time()
        to_return['metadata']['embedding'] = _encode(condition, metadata, self.planet_cfg, downsketch, downland_sketch, downtemp_sketch, ang)
        if random.random() < self.planet_cfg.encoder_dropout:
            to_return['metadata']['embedding'] = np.zeros_like(to_return['metadata']['embedding'])
        
        # print(f"Time to encode image: {time() - start_time:.2f}")
        # to_display = self.planet_cfg.output_display(
        #     to_return['cond_image'].unsqueeze(0), 
        #     to_return['target_image'].unsqueeze(0),
        #     to_return['target_image'].unsqueeze(0)) #[:, :, 1024:-768]
        # to_display = ((to_display.cpu().numpy().transpose(1, 2, 0)+1)/2*255).astype(np.uint8)
        return to_return

class DiskDataset(Dataset):
    def __init__(self, 
                 planet_cfg: PlanetConfig,
                 target_image_channels: int=None,
                 cond_image_channels: int=None,
                 normalise: bool=True,
                 conditioning_dropout: float=0.0,
                 tile_size: int=256,
                 train_mode: bool=True) -> None:
        super().__init__()
        # What we want to do here is to simply load images from the transformation folders
        # Then spin, hflip and vflip will be transformations that can be applied to the dataset
        # We can use get_k_val to get the k value for a given image name

        # TODO: Consider doing encoder stuff here or add the downsketch and downland_sketch to the return

        self.planet_cfg = planet_cfg
        setup(planet_cfg)
        self.seed = planet_cfg.planet_seed
        self.tile_size = tile_size
        self.normalise = normalise
        self.target_image_channels = target_image_channels if target_image_channels is not None else planet_cfg.output_channels()
        self.cond_image_channels = cond_image_channels if cond_image_channels is not None else planet_cfg.input_channels()
        self.generator = torch.Generator(device='cpu')
        if self.seed is not None:
            self.generator.manual_seed(self.seed)
        self.valid_tiles = planet_cfg.valid_tiles()
        random.Random(self.seed).shuffle(self.valid_tiles)
        if train_mode:
            self.valid_tiles = self.valid_tiles[:int(len(self.valid_tiles)*0.8)]

        cond_transforms = [
            # Convert Image or numpy array to tensor
            transforms.ToTensor()
        ]

        target_transforms = [
            # Convert Image or numpy array to tensor
            transforms.ToTensor()
        ]

        if normalise:
            cond_transforms.append(NormaliseTransform())
            target_transforms.append(NormaliseTransform())

        if self.planet_cfg.inpainting_channels > 0:
            cond_transforms.append(RandomMaskTransform(self.planet_cfg.inpainting_channels, self.generator))


        self.cond_transform = transforms.Compose(cond_transforms)
        self.target_transform = transforms.Compose(target_transforms)

    def __len__(self):
        return len(self.valid_tiles)
    
    @timing
    @profile
    def __getitem__(self, idx):
        start_time = time()
        y, x, mask, ref, ang = self.valid_tiles[idx]
        delta = 2**self.planet_cfg.downscale_offset
        mask_name = f"{mask}_{ref}_{ang}/{y}_{x}.{self.planet_cfg.image_extension}"
        down_mask_name = f"{mask}_{ref}_{ang}/{y//delta//256*256}_{x//delta//256*256}.{self.planet_cfg.image_extension}"
        cfg = get_mask_config(0, self.planet_cfg)
        cfg['hflip'] = random.randint(0, 1)
        cfg['vflip'] = random.randint(0, 1)
        # Quick and dirty fix. I'd like to figure out how to do this properly
        x += random.randint(0, 256//self.planet_cfg.spin-1)*self.planet_cfg.spin 
        vflip = cfg['vflip']
        hflip = cfg['hflip']
        # spin = cfg['spin']
        ry, rx = get_translated_coords(y, x, vflip, hflip, 0, self.planet_cfg.size, self.tile_size)
        rry, rrx = get_translated_coords(ry, rx, vflip, hflip, 0, self.planet_cfg.size, self.tile_size)
        assert rry == y and rrx == x, f"Translated coordinates are not the same as the original coordinates: {rry, rrx} != {y, x}"
        cfg['mask_refs'][mask] = ref
        cfg['mask_angs'][mask] = ang
        cfg['mask_rems'][mask] = False
        k = get_k_val(cfg, self.planet_cfg)

        to_return = {}
        down_tile_width = self.tile_size // delta

        mask_dirs = self.planet_cfg.mask_dirs
        down_mask_dirs = self.planet_cfg.downscale_cfg.mask_dirs
        condition = []
        for mask_type in self.planet_cfg.input_types:
            if 'mask' == mask_type:
                # TODO: Set masking width
                condition.append(create_mask(np.zeros((self.tile_size, self.tile_size))))
            elif mask_type == 'downland_sketch':
                mask_path = os.path.join(down_mask_dirs['land'], down_mask_name)
                mask = open_image_array(mask_path, self.tile_size)
                mask = landcover_paint(mask, self.planet_cfg)
                mask = get_tile(mask, y//delta-y//delta//256*256, x//delta-x//delta//256*256, down_tile_width)
                mask = resize(mask, (self.tile_size, self.tile_size), interpolation=INTER_NEAREST)
                condition.append(mask)
            elif mask_type == 'downsketch':
                mask_path = os.path.join(down_mask_dirs['dem'], down_mask_name)
                mask = open_image_array(mask_path, self.tile_size)
                mask = dilate_paint(mask, self.planet_cfg)
                mask = get_tile(mask, y//delta-y//delta//256*256, x//delta-x//delta//256*256, down_tile_width)
                mask = resize(mask, (self.tile_size, self.tile_size), interpolation=INTER_NEAREST)
                condition.append(mask)
            else:
                mask_path = os.path.join(mask_dirs[mask_type], mask_name)
                mask = open_image_array(mask_path, self.tile_size)
                mask = get_tile(mask, y-y//256*256, x-x//256*256, self.tile_size)
                condition.append(mask)

        


        target = []
        for mask_type in self.planet_cfg.output_types:
            mask_path = os.path.join(mask_dirs[mask_type], mask_name)
            mask = open_image_array(mask_path, self.tile_size)
            mask = get_tile(mask, y-y//256*256, x-x//256*256, self.tile_size)
            target.append(mask)
        
        condition = np.dstack(condition)
        target = np.dstack(target)
        if hflip:
            condition = np.fliplr(condition).copy()
            target = np.fliplr(target).copy()
        if vflip:
            condition = np.flipud(condition).copy()
            target = np.flipud(target).copy()

        assert len(condition.shape) == 3
        assert len(target.shape) == 3
        assert condition.shape[2] == self.cond_image_channels
        assert target.shape[2] == self.target_image_channels

        down_sketch_index = self.planet_cfg.input_index('downsketch')
        downland_sketch_index = self.planet_cfg.input_index('downland_sketch')

        if down_sketch_index > -1 and downland_sketch_index > -1:
            condition[:, :, downland_sketch_index][condition[:, :, down_sketch_index] == 0] = 0
        

        # TODO: Colour dropout

        metadata = {
            'range': np.max(target) - np.min(target),
            'zoom': 2**self.planet_cfg.size,
            'tile_y': ry,
            'tile_x': rx,
            'k': k,
            'factor': self.planet_cfg.size,
            'resolution': 0, #84375.0/(2**self.planet_cfg.size),
            'idx': idx,
        }
        to_return['target_image'] = self.target_transform(target.astype(np.float32)/255.0)
        to_return['cond_image'] = self.cond_transform(condition.astype(np.float32)/255.0)
        to_return['metadata'] = metadata

        # print(f"Time to load image: {time() - start_time:.2f}")
        start_time = time()
        to_return['metadata']['embedding'] = _encode(condition, metadata, self.planet_cfg)
        # print(f"Time to encode image: {time() - start_time:.2f}")
        return to_return

@profile
def _encode(cond, metadata, planet_cfg: PlanetConfig, full_sketch=None, full_land_sketch=None, full_temp_sketch=None, angle: int=0,
            width: int=None, height:int=None):
    '''
    Encode the conditional image
    '''
    embedding = metadata.get('embedding', None)
    if embedding is not None:
        return embedding
    im_conf = get_mask_config(metadata['k'], planet_cfg)
    if full_sketch is None:
        full_sketch = _gen_full_image({}, im_conf, 'sketch', planet_cfg.downscale_cfg, 'max')
    if full_land_sketch is None:
        full_land_sketch = _gen_full_image({}, im_conf, 'land_sketch', planet_cfg.downscale_cfg, 'max')
    if full_temp_sketch is None:
        full_temp_sketch = _gen_full_image({}, im_conf, 'temp_sketch', planet_cfg.downscale_cfg, 'max')
    
    all_mask = np.dstack([full_sketch > 0, full_land_sketch > 0, full_temp_sketch > 0]).astype(np.uint8)*255

    y = metadata['tile_y']
    x = metadata['tile_x']
    hflip = metadata['hflip']
    vflip = metadata['vflip']
    tile_size = metadata['tile_size']
    W = 512*2**planet_cfg.size
    H = W // 2
    if hflip:
        x = W - x - tile_size
    if vflip:
        y = H - y - tile_size
    delta = 2**planet_cfg.downscale_offset
    width = int(512*2**(planet_cfg.size-planet_cfg.downscale_offset)) if width is None else width
    
    # For debugging
    # downsketch = _gen_image(cond, im_conf, 'downsketch', planet_cfg, 'max', y, x)
    # sketch = _gen_image(cond, im_conf, 'sketch', planet_cfg, 'max', y, x)
    tile_w = tile_size // delta
    start_y = int(y + tile_size // 2) // delta
    start_x = int(x + tile_size // 2) // delta % width
    small_y = int(y) // delta
    small_x = int(x) // delta

    shift_x = int(im_conf['spin']) // delta
    if im_conf['hflip']:
        shift_x = -shift_x
    center = (2.5*tile_w, 2.5*tile_w)
    sketch_surrounds = get_tile(full_sketch, small_y - 2*tile_w, small_x + shift_x - 2*tile_w, 5*tile_w)
    sketch_surrounds = nearest_rotate(sketch_surrounds, angle, center)
    sketch_surrounds = get_tile(sketch_surrounds, tile_w, tile_w, 3*tile_w)
    # downsketch = resize(downsketch, (tile_w, tile_w), interpolation=INTER_NEAREST)
    # to_display = np.dstack([sketch_surrounds, np.zeros_like(sketch_surrounds), np.zeros_like(sketch_surrounds)])
    # to_display[tile_w:2*tile_w, tile_w:2*tile_w, 1] = downsketch
    # to_display = img.fromarray(to_display.astype(np.uint8)).resize((256, 256), resample=img.NEAREST)


    land_sketch_surrounds = get_tile(full_land_sketch, small_y - 2*tile_w, small_x + shift_x - 2*tile_w, 5*tile_w)
    land_sketch_surrounds = nearest_rotate(land_sketch_surrounds, angle, center)
    land_sketch_surrounds = get_tile(land_sketch_surrounds, tile_w, tile_w, 3*tile_w)

    temp_sketch_surrounds = get_tile(full_temp_sketch, small_y - 2*tile_w, small_x + shift_x - 2*tile_w, 5*tile_w)
    temp_sketch_surrounds = nearest_rotate(temp_sketch_surrounds, angle, center)
    temp_sketch_surrounds = get_tile(temp_sketch_surrounds, tile_w, tile_w, 3*tile_w)
    
    
    all_surrounds = np.dstack([sketch_surrounds > 0, land_sketch_surrounds > 0, temp_sketch_surrounds > 0]).astype(np.uint8)*255
    
    # We want to encode the following features:
    # Data from surrounding 8 sketch tiles
    # - Percentage of each elevation colour 9 x ~4
    # - Percentage of each landcover class 9 x 8
    # - Distance to ocean in all 9 directions 9 x 1 from edges of tile or middle 

    # Maybe: 
    # - Continent size 1

    # Rediscretize rotated sketches:
    planet_colours = planet_cfg.colours
    colour_step = (256 // planet_colours)
    sketch_surrounds = sketch_surrounds.astype(np.uint16) + 1
    sketch_surrounds //= colour_step
    sketch_surrounds *= colour_step
    sketch_surrounds[sketch_surrounds > 0] -= 1
    sketch_surrounds = sketch_surrounds.astype(np.uint8)

    planet_classes = planet_cfg.landcover_classes
    class_step = (256 // planet_classes)
    land_sketch_surrounds = land_sketch_surrounds.astype(np.uint16) + 1
    land_sketch_surrounds //= class_step
    land_sketch_surrounds *= class_step
    land_sketch_surrounds[land_sketch_surrounds > 0] -= 1
    land_sketch_surrounds = land_sketch_surrounds.astype(np.uint8)

    planet_temps = planet_cfg.temp_classes
    temp_step = (255 // planet_temps)
    temp_sketch_surrounds = temp_sketch_surrounds.astype(np.uint16) + 1
    temp_sketch_surrounds //= temp_step
    temp_sketch_surrounds *= temp_step
    temp_sketch_surrounds[temp_sketch_surrounds > 0] -= 1
    temp_sketch_surrounds = temp_sketch_surrounds.astype(np.uint8)

    # For debugging TODO
    sketch_tile = get_tile(sketch_surrounds, tile_w, tile_w, tile_w)
    land_sketch_tile = get_tile(land_sketch_surrounds, tile_w, tile_w, tile_w)
    all_surrounds = np.dstack([sketch_surrounds > 0, land_sketch_surrounds > 0, temp_sketch_surrounds > 0]).astype(np.uint8)*255

    # We add one here for the ocean which isn't included in the colour or class count
    sketch_colours = np.zeros((9, planet_colours + 1))
    landcover_classes = np.zeros((9, planet_classes + 1))
    temp_classes = np.zeros((9, planet_temps + 1))
    distances = np.zeros(9)
    inland_embedding = np.zeros(9) # 0.0 for ocean, 0.5 for coastal, 1.0 for inland
    oceanmask = (full_sketch == 0)
    # This is the maximum possible distance
    height = width // 2 if height is None else height
    max_dist = (height * np.sqrt(5))
    distances[4] = max_dist
    # debug_full_sketch = full_sketch.copy()
    for i, k in enumerate(range(9)):
        # TODO: Potential issues here. 
        # 1. Surrounding tiles are unrotated. Requires either rotating full sketch or sketch surrounds.
        #   - Full sketch:
        #       - Rotating the full sketch requires rotating about center coordinate of tile.
        #       - It also has issues with ocean direction since antarctica shouldn't be possible to rotate.
        #       - We can just use the rotated dx/dy for ocean with the unrotated full sketch
        #   - Surrounds:
        #       - We can easily rotate about the center of the tile. 
        #       - Will require getting a slightly larger tile.
        #   - Interpolation might cause issues, but if we rediscretize it should fix the colour issues.
        #   - Resolution issues aren't as easy to solve.
        #       - Will cause a slight accuracy issue.
        # 2. The full sketch will not be the same as the mosaic of tiles due to local dilation.
        #   - Not a huge issue since it will be very similar.

        # Solutions:
        # - Use rotated sketch surrounds for tile distributions
        # - Use rotated dx/dy for nearest ocean
        left = (k % 3) * tile_w
        top = (k // 3) * tile_w
        colours, col_counts = np.unique(sketch_surrounds[top:top+tile_w, left:left+tile_w], return_counts=True)
        colours = colours.astype(np.uint16)
        colours[colours > 0] += 1
        colour_step = (256 // planet_colours)
        assert np.array_equal(colours // colour_step * colour_step, colours), f"Colour values: {colours}, Colour step: {colour_step}"
        colours //= colour_step
        classes, class_counts = np.unique(land_sketch_surrounds[top:top+tile_w, left:left+tile_w], return_counts=True)
        classes = classes.astype(np.uint16)
        classes[classes > 0] += 1
        class_step = (256 // planet_classes)
        assert np.array_equal(classes // class_step * class_step, classes), f"Class values: {classes}, Class step: {class_step}"
        classes //= class_step
        temp_values, temp_counts = np.unique(temp_sketch_surrounds[top:top+tile_w, left:left+tile_w], return_counts=True)
        temp_values = temp_values.astype(np.uint16)
        temp_values[temp_values > 0] += 1
        temp_step = (255 // planet_temps)
        assert np.array_equal(temp_values // temp_step * temp_step, temp_values), f"Temp values: {temp_values}, Temp step: {temp_step}"
        temp_values //= temp_step

        sketch_colours[i, colours] = col_counts / (tile_w*tile_w)
        landcover_classes[i, classes] = class_counts / (tile_w*tile_w)
        temp_classes[i, temp_values] = temp_counts / (tile_w*tile_w)
        if sketch_colours[i][0] == 1.0:
            inland_embedding[i] = 0.0 # All ocean
        elif sketch_colours[i][0] > 0:
            inland_embedding[i] = 0.5 # Coastal
        else:
            inland_embedding[i] = 1.0 # Inland
        if k == 4:
            continue
        dy = (k // 3) - 1
        dx = (k % 3) - 1
        start_y = int(y + tile_size // 2) // delta
        start_x = int(x + tile_size // 2) // delta % width
        s = np.sin(angle/180*np.pi)
        c = np.cos(angle/180*np.pi)
        new_dy = dx * s + dy * c
        new_dx = dx * c - dy * s
        corner_x = start_x + (tile_w // 2) * dx
        corner_y = start_y + (tile_w // 2) * dy
        # debug_full_sketch[int(corner_y) % 256, int(corner_x) % 512] = 128
        assert (new_dy) ** 2 + (new_dx) ** 2 - dy ** 2 - dx ** 2 < 1e-6, f"Rotation failed: {new_dy} {new_dx} {dy} {dx}"
        dy, dx = new_dy, new_dx
        corner_x = start_x + (tile_w // 2) * dx
        corner_y = start_y + (tile_w // 2) * dy
        # debug_full_sketch[int(corner_y) % 256, int(corner_x) % 512] = 255
        

        oy = start_y
        ox = start_x

        # debug_full_sketch[start_y, start_x] = 255
        
        found = False
        # Maybe change to distance from each corner rather than center
        # I would probably then add an additional element for the center
        #
        # Done this now. Not sure if it the best choice.
        tries = 0
        while oy >= 0 and oy < height and ox:
            tries += 1
            sqrdist = (oy - start_y)**2 + (ox - start_x)**2
            distances[i] = np.sqrt(sqrdist) / max_dist
            if oceanmask[int(oy), int(ox) % width]:
                if distances[i] < distances[4]:
                    # TODO Consider whether we want to do this
                    # Currently setting the middle distance to the nearest ocean in any direction
                    # Model may not be able to distinguish this
                    distances[4] = distances[i]
                if (oy - start_y)**2 + (ox - start_x)**2 >= (corner_y - start_y) ** 2 + (corner_x - start_x) ** 2:
                    found = True
                    break
            oy += dy
            ox += dx
            if distances[i] > 1:
                break
            if int(ox) % width == start_x and int(ox) != start_x and dx != 0:
                break 
            if dx == 0 and dy == 0:
                break
            if tries % 100000 == 0:
                print(f"Warning: Ocean not found after {tries} tries")
                break
        sqrdist = (oy - corner_y)**2 + (ox - corner_x)**2
        distances[i] = np.sqrt(sqrdist) / max_dist
        if not found:
            # If we haven't found ocean in a direction, 
            # it means that we are in antarctica and we have found the edge.
            # Since antartica is roughly round, 
            # we can multiply the current distance by two to roughly find where it would be on a globe
            distances[i] *= 2
    # TODO Check encoding is correct
    to_return = np.concatenate((sketch_colours.flatten(), 
                                landcover_classes.flatten(), 
                                temp_classes.flatten(),
                                inland_embedding))
    return to_return.astype(np.float32) 

class PlanetDataset(Dataset):
    def __init__(self, 
                 planet_cfg: Union[PlanetConfig, List[PlanetConfig]],
                 target_image_channels: int=None,
                 cond_image_channels: int=None, 
                 normalise: bool=True,
                 conditioning_dropout: float=0.0,
                 tile_size: int=256,
                 rgb_mode: bool=False,
                 random_oceans: bool=False,
                 inverted_target: bool=False,
                 num_configurations: int=1):
        
        gen_lists = planet_cfg.gen_lists

        if isinstance(planet_cfg, list):
            self.planet_cfg = planet_cfg[0]
            self.extra_planet_cfgs = planet_cfg[1:]
            if gen_lists:
                self.extra_gen_lists = [gen_list(cfg, True) for cfg in self.extra_planet_cfgs]
        else:
            self.planet_cfg = planet_cfg
            self.extra_planet_cfgs = [replace(
                planet_cfg, planet_seed=planet_cfg.planet_seed + i if planet_cfg.planet_seed is not None else None
                ) for i in range(1, num_configurations)]
            if gen_lists:
                self.extra_gen_lists = [gen_list(cfg, True) for cfg in self.extra_planet_cfgs]
        planet_cfg = self.planet_cfg
        setup(planet_cfg)
        self.length = planet_cfg.iters
        self.size = planet_cfg.size
        self.normalise = normalise
        self.seed = planet_cfg.planet_seed
        self.tile_size = tile_size
        self.rgb_mode = rgb_mode
        self.random_oceans = random_oceans
        self.inverted_target = inverted_target
        self.generator = torch.Generator(device='cpu')
        if self.seed is not None:
            self.generator.manual_seed(self.seed)
        self.mask_count = planet_cfg.mask_count
        self.steps = 0
        self.target_image_channels = target_image_channels if target_image_channels is not None else planet_cfg.output_channels()
        self.cond_image_channels = cond_image_channels if cond_image_channels is not None else planet_cfg.input_channels()
        self.river_dropout = planet_cfg.river_dropout
        self.conditioning_dropout = conditioning_dropout
        if planet_cfg.bucketing_mode == 'global-max' and 'dem' in self.planet_cfg.output_types:
            self.max_pixel = planet_cfg.dem_max_pixel
        else:
            self.max_pixel = None
        self.gen_lists = gen_lists
        if gen_lists:
            self.gen_list = gen_list(planet_cfg, True)
            random.Random(self.seed).shuffle(self.gen_list)
            for i, extra_cfg in enumerate(self.extra_planet_cfgs):
                random.Random(extra_cfg.planet_seed).shuffle(self.extra_gen_lists[i])
        # Create transformation that will be applied to each image
        cond_transforms = [
            # Convert Image or numpy array to tensor
            transforms.ToTensor()
        ]

        target_transforms = [
            # Convert Image or numpy array to tensor
            transforms.ToTensor()
        ]

        if self.max_pixel is not None:
            if not self.rgb_mode and not self.random_oceans:
                target_transforms.append(ScalingTransform(255/self.max_pixel))
            if 'dem' in self.planet_cfg.input_types:
                cond_transforms.append(ScalingTransform(255/self.max_pixel))

        if normalise:
            cond_transforms.append(NormaliseTransform())
            target_transforms.append(NormaliseTransform())
        if self.planet_cfg.inpainting_channels > 0:
            cond_transforms.append(RandomMaskTransform(self.planet_cfg.inpainting_channels, self.generator))
        if self.planet_cfg.image_mode == 'none':
            cond_transforms.append(ZeroTensorTransform())
        if self.planet_cfg.image_mode == 'noise':
            cond_transforms.append(NoiseTransform(self.generator))
        if self.planet_cfg.image_mode == 'normal-noise':
            cond_transforms.append(NormalisedNoiseTransform(self.generator))
        
        if self.inverted_target:
            target_transforms.append(InvertTransform())

        self.cond_transform = transforms.Compose(cond_transforms)
        self.target_transform = transforms.Compose(target_transforms)


    def __len__(self):
        return self.length
    
    # @profile
    def __getitem__(self, idx):
        
        # This is fairly slow but it is faster than generating a new image instead
        # We are limited by the fact that we can't store all the images in memory
        if self.planet_cfg.randomize_steps > 0 and self.steps % self.planet_cfg.randomize_steps == 0:
            self.planet_cfg.randomize_parameters()
            if self.gen_lists:
                self.gen_list = gen_list(self.planet_cfg, True)
                random.Random(self.seed).shuffle(self.gen_list)
            if not self.planet_cfg.on_fly_conditioning:
                setup(self.planet_cfg)
            for i, extra_cfg in enumerate(self.extra_planet_cfgs):
                extra_cfg.randomize_parameters()
                if self.gen_lists:
                    self.extra_gen_lists[i] = gen_list(extra_cfg, True)
                    random.Random(extra_cfg.planet_seed).shuffle(self.extra_gen_lists[i])
                if not extra_cfg.on_fly_conditioning:
                    setup(extra_cfg)
        if self.gen_lists:
            k, y, x = self.gen_list[idx]    
            condition, target = gen_tile_pair(k, 256*y, 256*x, self.planet_cfg)
        else:
            seed = self.planet_cfg.planet_seed + idx if self.seed is not None else None
            condition, target, k, y, x = get_next_tile_pair(seed, self.planet_cfg)
        self.steps += 1
        stacked = False
        for i, extra_cfg in enumerate(self.extra_planet_cfgs):
            if self.gen_lists:
                xk, xy, xx  = self.extra_gen_lists[i][idx]
                extra_condition, extra_target = gen_tile_pair(xk, 256*xy, 256*xx, extra_cfg)
            else:
                seed = extra_cfg.planet_seed + idx if self.seed is not None else None
                extra_condition, extra_target, xk, xy, xx = get_next_tile_pair(seed, extra_cfg, y)
            if not overlap(extra_target, target) and xy == y:
                target = target + extra_target
                condition = np.maximum(condition, extra_condition)
                stacked = True
        if condition.shape[2] < self.cond_image_channels:
            condition = change_channels(condition, self.cond_image_channels)
        if target.shape[2] < self.target_image_channels:
            target = change_channels(target, self.target_image_channels)
        conditioning_seed = idx if self.seed is not None else None
        # TODO Maybe do encoder dropout instead of channel dropout
        # for c in range(1, self.cond_image_channels):
        #     condition[:, :, c] = dropout(condition[:, :, c], p=self.conditioning_dropout, 
        #                                  seed=conditioning_seed+c if conditioning_seed is not None else None)

        # Colour dropout TODO
        # I might need to do some stuff with connected components to make this work properly
        # For now I will just use fewer colours
        si = self.planet_cfg.input_index('sketch')
        if si > -1 and self.planet_cfg.colour_dropout > 0:
            sketch = condition[:, :, si]
            new_sketch = sketch.copy()
            colour_list = np.unique(sketch).tolist()
            if 0 in colour_list:
                colour_list.remove(0)
            colour_list.reverse()
            for i, c in enumerate(colour_list[1: -1], start=1):
                seed = conditioning_seed + c if conditioning_seed is not None else None
                if random.Random(seed).random() < self.planet_cfg.colour_dropout:
                    new_col = colour_list[i+1]
                    new_sketch[sketch == c] = new_col # Set to lower colour to create gradient 
            condition[:, :, si] = new_sketch



        assert len(condition.shape) == 3
        assert len(target.shape) == 3
        
        if self.max_pixel is not None:
            target = np.clip(target, 0, self.max_pixel)
        if self.rgb_mode:
            target = np_rgb(target[:, :, 0], 'terrain', self.max_pixel if self.max_pixel is not None else 255)
        if self.random_oceans:
            assert self.target_image_channels >= 2
            normal_channel = target[:, :, 0]
            mask_channel = target[:, :, 1]
            max_val = self.max_pixel if self.max_pixel is not None else 255
            mask_channel[normal_channel == 0] = max_val
            mask_channel[normal_channel != 0] = 0
            normal_channel[normal_channel == 0] = np.random.randint(0, max_val, size=normal_channel[normal_channel == 0].shape)

        condition = condition[:, :, :self.cond_image_channels]
        target = target[:, :, :self.target_image_channels]
        if self.tile_size < 256:
            water = 1.0
            tries = 0
            while water > 0.75 and tries < 100:
                tile_x = random.randint(0, 256-self.tile_size)
                tile_y = random.randint(0, 256-self.tile_size)
                condition_tile = get_tile(condition, tile_y, tile_x, self.tile_size)
                target_tile = get_tile(target, tile_y, tile_x, self.tile_size)
                water = np.sum(condition_tile[:, :, 0] == 0) / (self.tile_size**2)
                tries += 1
            condition = condition_tile
            target = target_tile
        elif self.tile_size > 256:
            condition = resize(condition, (self.tile_size, self.tile_size), interpolation=INTER_NEAREST)
            target = resize(target, (self.tile_size, self.tile_size), interpolation=INTER_NEAREST)

        assert condition.shape[2] == self.cond_image_channels
        assert target.shape[2] == self.target_image_channels
        # TODO: Update these
        metadata = {
            'range': np.max(target) - np.min(target),
            'zoom': 2**self.size,
            'tile_y': y,
            'tile_x': x,
            'k': k,
            'factor': self.size,
            'resolution': 84375.0/(2**self.size),
            'idx': idx,
            'stacked': stacked,
        }
        to_return = {
            'target_image': self.target_transform(target.astype(np.float32)/255.0),
            'cond_image': self.cond_transform(condition.astype(np.float32)/255.0),
            'metadata': metadata
        }
        return to_return
@dataclass
class DataLoaderArgs:
    num_workers: int = field(
        default=0,
        metadata={
            "help": "Number of worker threads"
        }
    )
    batch_size: int = field(
        default=1,
        metadata={
            "help": "Batch size"
        }
    )

if __name__ == '__main__':
    parser = CustomArgumentParser(
        (
            PlanetConfig, 
            DataLoaderArgs
        ),
        description="Benchmark dataset loading speed" 
    )
    test_cfg, dataloader_args = parser.parse_args_into_dataclasses()
    # setup(test_cfg, force=False)
    dataset = RAMDataset(test_cfg, target_image_channels=test_cfg.output_channels(), 
                            cond_image_channels=test_cfg.input_channels(), 

                            normalise=True)
    dataloader = DataLoader(dataset, shuffle=False, **dataloader_args.__dict__)
    iters = test_cfg.iters
    iters //= dataloader_args.batch_size
    if len(dataloader) < iters:
        iters = len(dataloader)
    t1 = time()
    with tqdm(total=iters) as pbar:
        for i, batch in enumerate(dataloader):
            # im = test_cfg.output_display(batch['cond_image'], batch['target_image'], batch['target_image'])
            # im = tensor_to_np(im)
            if i >= iters:
                break
            pbar.update(1)
            
    print(f'batch_size: {dataloader_args.batch_size} | num_workers: {dataloader_args.num_workers} | image_mode: {test_cfg.image_mode}', end='')
    print(f' | {dataloader_args.batch_size*iters/(time() - t1):.2f} it/s')