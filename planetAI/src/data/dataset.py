import itertools
from torch.utils.data import Dataset, DataLoader
from time import time
from torchvision import transforms
import torch
import numpy as np
import random
from cv2 import resize, INTER_NEAREST, dilate, INTER_LANCZOS4
import cv2
from PIL import Image as img
from scipy.ndimage import label
from tqdm import tqdm
from dataclasses import dataclass, field, replace
import os
from math import ceil
from diffusers.models import AutoencoderKL
import math
from skimage.morphology import skeletonize
import heapq

try:
    from line_profiler import profile
except ImportError:
    from .utils import profile


from .utils import (
    PlanetConfig,
    np_hex_to_rgb,
    np_rgb_to_hex,
    masking,
    open_image_array,
    image_grid,
    continuous_to_spread,
    modal_resize,
    tensor_to_np,
    get_data_image,
)

from .landcover_utils import LandcoverClasses, gray_to_land, randomize_land, translate_land
from .sketch_gen import landcover_paint, dilate_paint, temperature_paint, get_buckets, get_start_points
from .dataclass_argparser import CustomArgumentParser
from .modal_sketch import ModalSketch
from .sphere_mapping import SphereMapping, QuadSphere
from .river_modal_sketch import RiverModalSketch
from .river_processing import apply_filters, get_stacked_rivers, filter_components
from .line_dataset import line_resizer, line_rotater
from .atlas_loader import AtlasLoader, quad_data_loader, QuadAtlasLoader


class NormaliseTransform(torch.nn.Module):
    """Normalise (0, 1) to (-1, 1)"""

    def forward(self, img):
        return (img * 2) - 1


class RandomMaskTransform(torch.nn.Module):
    def __init__(self, channels: int = 1, generator: torch.Generator = None):
        super().__init__()
        self.channels = channels
        self.generator = generator

    def forward(self, img):
        img[: self.channels] = masking(img[: self.channels], self.generator)
        return img


def segment_and_skeletonize(mask: np.ndarray, num_segments: int = 25, overlap: float = 0.2) -> np.ndarray:
    lines = np.zeros_like(mask, dtype=bool)
    mask_max = mask.max()
    mask_min = mask.min()
    for i in range(num_segments):
        segment_width = (mask_max - mask_min) / num_segments
        min_val = segment_width * i * (1 - overlap / 2) + mask_min
        max_val = segment_width * (i + 1) * (1 + overlap / 2) + mask_min
        current_mask = (mask > min_val) & (mask <= max_val)
        current_mask = cv2.dilate(current_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        current_lines = skeletonize(current_mask > 0)
        lines |= current_lines
    return lines


def increasing_djikstra(
    river_upa: np.ndarray, start: tuple[int, int], end: tuple[int, int]
) -> list[tuple[int, int]] | None:
    stack: list[list[tuple[int, int, int]]] = []
    heapq.heapify(stack)
    heapq.heappush(stack, [(1, *start)])
    h, w = river_upa.shape[:2]
    visited = np.zeros_like(river_upa, dtype=bool)
    ey, ex = end
    while stack:
        prev_points = heapq.heappop(stack)
        dist, y, x = prev_points[-1]
        visited[y, x] = True
        if y == ey and x == ex:
            return [(y, x) for _, y, x in prev_points]
        for dy, dx in itertools.product([-1, 0, 1], [-1, 0, 1]):
            if dy == 0 and dx == 0:
                continue
            ny = y + dy
            nx = x + dx
            if not (0 <= ny < h) or not (0 <= nx < w):
                continue
            if visited[ny, nx]:
                continue
            if river_upa[y, x] <= river_upa[ny, nx] and river_upa[y, x] > 0:
                heapq.heappush(stack, prev_points + [(dist + 1, ny, nx)])
    return None


def djikstra_extract_river_segment(river_upa: np.ndarray) -> np.ndarray:
    start_points = get_start_points(river_upa, False, False)
    min_value = river_upa[river_upa > 0].min()
    sy, sx = np.where((start_points) | (river_upa == min_value))
    ey, ex = np.where(river_upa == river_upa.max())
    river_ys = []
    river_xs = []
    for start, end in itertools.product(zip(sy, sx), zip(ey, ex)):
        path = increasing_djikstra(river_upa, start, end)
        if path is None:
            continue
        river_ys.extend([y for y, _ in path])
        river_xs.extend([x for _, x in path])

    mask = np.zeros_like(river_upa, dtype=bool)
    mask[river_ys, river_xs] = True
    return mask


def djikstra_extract_rivers(river_upa: np.ndarray) -> np.ndarray:
    n, river_segments = cv2.connectedComponents((river_upa > 0).astype(np.uint8) * 255, connectivity=8)
    mask = np.zeros_like(river_upa, dtype=bool)
    for i in range(n):
        current_river_upa = river_upa.copy()
        current_river_upa[river_segments != i] = 0
        if current_river_upa.any():
            mask |= djikstra_extract_river_segment(current_river_upa)
    mask = skeletonize(mask)
    # TODO Remove cycles
    return mask


def get_river_upa_mask(river_upa: np.ndarray, river_upa_max: float, planet_cfg: PlanetConfig) -> np.ndarray:
    # Random threshold used to randomly remove some rivers, including the big ones.
    # This will still allow the model to generate rivers without an explicit condition
    # Akin to dropout
    random_th = random.random() ** 2  # Square to overrepresent lower values
    apply_th = random.random() < planet_cfg.river_upa_dropout_chance

    # Random offset to add some variation to the output and allow the model to generate
    # more rivers when the input data is different to the training data
    random_offset = random.random() * planet_cfg.river_upa_variance
    apply_offset = random.random() < planet_cfg.river_upa_variance_chance

    mask = river_upa.copy()

    mask[mask < planet_cfg.river_upa_min] = 0

    # Normalise by the global river UPA maximum
    mask = mask / river_upa_max

    if apply_th:
        mask[mask < random_th] = 0
    if apply_offset:
        mask = np.maximum(mask - random_offset, 0)

    # thin_rivers = segment_and_skeletonize(mask)
    # thin_rivers = djikstra_extract_rivers(river_upa)
    # river_upa[~thin_rivers] = 0
    # thin_rivers = djikstra_extract_rivers(river_upa)

    # We apply a sqrt here to overrepresent high values, giving us a more uniform distribution of values
    mask = (np.sqrt(mask) * 255)
    if planet_cfg.discrete_rivers:
        # We skeletonize here to avoid interpolation errors from the rotation and also so we have a uniform width
        normal_thin_rivers = skeletonize(mask > 0)
        mask[~normal_thin_rivers] = 0
    return mask


def array_variance(arr: np.ndarray, max_variance: int = 10) -> np.ndarray:
    return (arr + (np.random.random() - 0.5) * 2 * max_variance).clip(0, 255).astype(np.uint8)


def fill_missing(landcover: np.ndarray, water_mask: np.ndarray) -> np.ndarray:
    missing = ((landcover == 0) & (~water_mask))
    while missing.any():
        next_landcover = cv2.dilate(landcover, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        landcover[missing] = next_landcover[missing]
        next_missing = ((landcover == 0) & (~water_mask))
        if np.array_equal(next_missing, missing):
            # If nothing changes then we can assume that the missing
            # stuff is in the middle of the ocean so we don't care
            break
        missing = next_missing
    return landcover


def fill_land(landcover: np.ndarray, sat: np.ndarray, temp: np.ndarray, dem: np.ndarray) -> np.ndarray:
    landcover = landcover.copy()
    water_mask = (sat[:, :, 0] < 10) & (sat[:, :, 1] < 30) & (sat[:, :, 2] < 40)
    deep_ocean_mask = (sat[:, :, 0] == 2) & (sat[:, :, 1] == 5) & (sat[:, :, 2] == 20)
    snow_mask = (sat[:, :, 0] > 240) & (sat[:, :, 1] > 240) & (sat[:, :, 2] > 240)
    missing = ((landcover == 0) & (~water_mask))
    inland_water = ((landcover == 0) | (water_mask)) & (dem > 10)  # This fills in a lot of stuff that we don't want
    if not missing.any():
        return landcover
    # Prioritise adding missing snow first since it is a minority class
    landcover[missing & snow_mask & (temp <= 50)] = LandcoverClasses.SNOW_AND_ICE.gray_colour
    landcover[inland_water] = LandcoverClasses.WATER.gray_colour
    landcover = fill_missing(landcover, water_mask)

    # Remove coastal water

    ocean = (landcover == 0).astype(np.uint8)
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    coastline = (cv2.dilate(ocean, kernel) - ocean) > 0
    dilated_landcover = cv2.dilate(landcover, kernel)
    coastal_water = coastline & (landcover == LandcoverClasses.WATER.gray_colour)
    landcover[coastal_water] = dilated_landcover[coastal_water]

    # Make sure ocean is definitely labelled as ocean
    landcover[deep_ocean_mask] = LandcoverClasses.OPEN_WATER.gray_colour

    return landcover


def upscale_regularization(target: np.ndarray, planet_cfg: PlanetConfig, tile_size: int, delta: int) -> np.ndarray:
    """
    Remove detail from the target and resize correctly to regularize the low resolution input
    """
    # TODO Tweak these values
    blur_size = planet_cfg.max_blur_radius * 2 + 1
    blur_amount = planet_cfg.max_blur_amount
    if planet_cfg.upscale_reg_strat == "mean":
        mask = cv2.blur(target, (blur_size, blur_size))
    if planet_cfg.upscale_reg_strat == "median+gauss" or planet_cfg.upscale_reg_strat == "median":
        blur_size = max(blur_size, 3)
        if blur_size > 5:
            target = target.clip(0, 255).round().astype(np.uint8)
        target = cv2.medianBlur(target, blur_size)
    if planet_cfg.upscale_reg_strat == "median+gauss" or planet_cfg.upscale_reg_strat == "gauss":
        target = cv2.GaussianBlur(target, (blur_size, blur_size), blur_amount)

    mask = resize(
        target,
        (tile_size // delta, tile_size // delta),
        interpolation=cv2.INTER_LANCZOS4,
    )
    mask = resize(
        mask,
        (tile_size, tile_size),
        interpolation=INTER_NEAREST,
    )
    return mask


shared_sat_dem_dict = {}
shared_sat_dem_s_dict = {}
shared_land_temp_dict = {}
shared_land_temp_s_dict = {}
shared_rivers_dict = {}
shared_mars_sat_dem_dict = {}
shared_mars_land_temp_dict = {}
shared_bathy_dict = {}
shared_quad_boundary_sketch_dict = {}
shared_river_upa_dict = {}

shared_atlas_loader_dict = {}
shared_quad_atlas_loader_dict = {}


class RAMDataset(Dataset):
    @profile
    def __init__(
        self,
        planet_cfg: PlanetConfig,
        target_image_channels: int = None,
        cond_image_channels: int = None,
        normalise: bool = True,
        conditioning_dropout: float = 0.0,
        tile_size: int = 256,
        mode: str = "train",
        auto_encoder: AutoencoderKL | None = None,
        sat_dem: np.ndarray | None = None,
        land_temp: np.ndarray | None = None,
        do_transforms: bool = True,
        shuffle_landcover: bool = False,
    ) -> None:
        super().__init__()
        # Here we want to load the full untranslated images into memory
        # We can then take a random crop from each image (twice as large as the tile size)
        # This can be rotated and flipped then cropped to the tile size
        self.min_angle = planet_cfg.min_angle
        self.max_angle = planet_cfg.max_angle  # TODO Maybe increase this
        self.planet_cfg = planet_cfg
        self.shuffle_landcover = shuffle_landcover
        global shared_atlas_loader_dict
        global shared_quad_atlas_loader_dict

        shape = planet_cfg.H, planet_cfg.W

        shared_atlas_loader = shared_atlas_loader_dict.get(shape)
        shared_quad_atlas_loader = shared_quad_atlas_loader_dict.get(shape)

        if shared_atlas_loader is None:
            shared_atlas_loader = AtlasLoader(planet_cfg)
            shared_atlas_loader_dict[shape] = shared_atlas_loader
        self.atlas_loader = shared_atlas_loader
        if shared_quad_atlas_loader is None:
            shared_quad_atlas_loader = QuadAtlasLoader(planet_cfg, self.atlas_loader)
            shared_quad_atlas_loader_dict[shape] = shared_quad_atlas_loader
        self.quad_atlas_loader = shared_quad_atlas_loader
        self.do_transforms = do_transforms
        # setup(planet_cfg)
        self.seed = planet_cfg.planet_seed
        self.use_summer = planet_cfg.use_summer
        self.use_bathy = 'bathy' in planet_cfg.output_types
        self.tile_size = tile_size
        self.normalise = normalise
        self.target_image_channels = (
            target_image_channels or planet_cfg.output_channels()
        )
        self.cond_image_channels = cond_image_channels or planet_cfg.input_channels()
        self.generator = torch.Generator(device="cpu")
        if self.seed is not None:
            self.generator.manual_seed(self.seed)
        self.auto_encoder = auto_encoder
        # Always use quad when using bathy
        self.use_quad = planet_cfg.use_quad_data or self.use_bathy
        self.use_mars = planet_cfg.use_mars_data

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
            cond_transforms.append(
                RandomMaskTransform(self.planet_cfg.inpainting_channels, self.generator)
            )

        self.cond_transform = transforms.Compose(cond_transforms)
        self.target_transform = transforms.Compose(target_transforms)

        H = 256 * 2**planet_cfg.size
        W = 512 * 2**planet_cfg.size

        h = 256 * 2 ** (planet_cfg.size - planet_cfg.downscale_offset)
        w = 512 * 2 ** (planet_cfg.size - planet_cfg.downscale_offset)
        h = int(h)
        w = int(w)
        self.H = H
        self.W = W
        self.h = h
        self.w = w
        self.sphere_mapping = SphereMapping(shape=(H, W))
        self.down_sphere_mapping = SphereMapping(shape=(h, w))
        self.quad_sphere_mapping = QuadSphere(shape=(H, W))
        self.down_quad_sphere_mapping = QuadSphere(shape=(h, w))

        self.size = planet_cfg.size
        self.downscale_offset = planet_cfg.downscale_offset
        self.delta = 2**self.downscale_offset
        self.down_tile_width = self.tile_size // self.delta
        # This should make a shared memory array for all subprocesses
        global shared_sat_dem_dict
        global shared_sat_dem_s_dict
        global shared_land_temp_dict
        global shared_land_temp_s_dict
        global shared_rivers_dict
        global shared_mars_sat_dem_dict
        global shared_mars_land_temp_dict
        global shared_bathy_dict
        global shared_quad_boundary_sketch_dict
        global shared_river_upa_dict
        # FOR TESTING
        # mars_shape = (mars_shape[0] // 2, mars_shape[1] // 2)
        mars_shape = (self.atlas_loader.mars_H, self.atlas_loader.mars_W)

        if self.use_mars and mars_shape not in shared_mars_sat_dem_dict:
            mars_sat = self.atlas_loader.mars_sat
            mars_dem = self.atlas_loader.mars_dem
            shared_mars_sat_dem_dict[mars_shape] = np.dstack([mars_sat, mars_dem]).astype(np.uint8)

        if self.use_mars and mars_shape not in shared_mars_land_temp_dict:
            # mars_temp = planet_cfg.open_image_array("Mars_Temp_1024x512.png")
            # mars_temp = cv2.resize(
            #     mars_temp, (mars_W, mars_H), interpolation=cv2.INTER_LANCZOS4
            # )
            mars_temp = self.atlas_loader.mars_temp
            shared_mars_land_temp_dict[mars_shape] = np.dstack([np.ones_like(mars_temp) * 255, mars_temp])

        if self.use_mars:
            self.mars_quadsphere = QuadSphere(shape=mars_shape)
            self.mars_sphere = SphereMapping(shape=mars_shape)

        shape = (H, W)
        if shape not in shared_sat_dem_dict:
            if sat_dem is None:
                # dem = planet_cfg.open_image_array(f"World_DEM_{W}x{H}.png")
                # sat = planet_cfg.open_image_array(f"world.satellite.{W}x{H}.png")
                dem = self.atlas_loader.float_dem
                sat = self.atlas_loader.sat
                shared_sat_dem_dict[shape] = np.dstack([sat, dem])
            else:
                shared_sat_dem_dict[shape] = sat_dem
        if shape not in shared_land_temp_dict:
            if land_temp is None:
                # land = planet_cfg.open_image_array(f"World_LandCover_{W}x{H}.png")
                # temp = planet_cfg.open_image_array(f"World_Temp_{W}x{H}.png")
                land = self.atlas_loader.land
                temp = self.atlas_loader.temp
                shared_land_temp_dict[shape] = np.dstack([land, temp])
            else:
                shared_land_temp_dict[shape] = land_temp
        sat_dem_data = shared_sat_dem_dict[shape]
        dem = sat_dem_data[:, :, 3]
        sat = sat_dem_data[:, :, :3]
        land_temp_data = shared_land_temp_dict[shape]
        land = land_temp_data[:, :, 0]
        temp = land_temp_data[:, :, 1]

        if self.use_summer:
            if shape not in shared_sat_dem_s_dict:
                shared_sat_dem_s_dict[shape] = sat_dem_data.copy()
                sat_summer = self.atlas_loader.sat_summer
                shared_sat_dem_s_dict[shape][:, :, :3] = sat_summer

            if shape not in shared_land_temp_s_dict:
                shared_land_temp_s_dict[shape] = land_temp_data.copy()
                temp_summer = self.atlas_loader.temp_summer
                shared_land_temp_s_dict[shape][:, :, 1] = temp_summer

        if self.use_bathy:
            # TODO Deal with land better
            shape = (H, W)
            shared_bathy_dict[shape] = get_data_image(
                planet_cfg.data_dir,
                (H, W),
                "gebco_bathy.WxH.jpg",
                default_shape=(10801, 21601),
                interpolation=cv2.INTER_LANCZOS4,
            )
            shared_quad_boundary_sketch_dict[shape] = get_data_image(
                planet_cfg.data_dir,
                self.quad_sphere_mapping.quad_shape,
                "quad_boundary_line_sketch_WxH.png",
                default_shape=(652, 3912),
                interpolation=cv2.INTER_NEAREST,
                custom_resizer=line_resizer,
            )

        if self.use_mars:
            mars_loader = quad_data_loader(planet_cfg.data_dir, mars_shape, self.mars_quadsphere)
            mars_shape = (self.atlas_loader.mars_H, self.atlas_loader.mars_W)
            quad_mars_sat_dem = mars_loader("mars_sat_dem", shared_mars_sat_dem_dict[mars_shape])
            quad_mars_land_temp = mars_loader("mars_land_temp", shared_mars_land_temp_dict[mars_shape], discrete=True)

        self.use_river_upa = "river_upa" in self.planet_cfg.input_types or self.planet_cfg.add_river_upa_to_dem

        quad_loader = quad_data_loader(planet_cfg.data_dir, (H, W), self.quad_sphere_mapping)
        shape = (H, W)
        quad_sat_dem = quad_loader("sat_dem", shared_sat_dem_dict[shape])
        quad_land_temp = quad_loader("land_temp", shared_land_temp_dict[shape], discrete=True)
        if self.use_summer:
            quad_sat_dem_s = quad_loader("sat_dem_summer", shared_sat_dem_s_dict[shape])
            quad_land_temp_s = quad_loader("land_temp_summer", shared_land_temp_s_dict[shape], discrete=True)
        if self.use_bathy:
            quad_bathy = quad_loader("bathy", shared_bathy_dict[shape], discrete=False, use_cached=True)

        self.river_upa_max = np.inf
        if self.use_river_upa:
            if self.use_quad:
                shared_river_upa = self.quad_atlas_loader.quad_river_upa
            else:
                shared_river_upa = self.atlas_loader.river_upa
            if self.planet_cfg.discrete_rivers:
                shared_river_upa = cv2.dilate(
                    shared_river_upa, np.ones((3, 3), dtype=np.uint8), iterations=1
                )
            self.river_upa_max = shared_river_upa.max()
            shared_river_upa_dict[shape] = shared_river_upa

        self.downdem = resize(dem, (w, h), interpolation=INTER_LANCZOS4)
        downland_dir = os.path.join(planet_cfg.data_dir, f"World_LandCover_{w}x{h}.png")
        self.downland = open_image_array(downland_dir)
        if self.downland is None:
            self.downland = modal_resize(land, self.delta)
            img.fromarray(self.downland).save(downland_dir)
        self.downsat = resize(sat, (w, h), interpolation=INTER_LANCZOS4)
        self.downtemp = resize(temp, (w, h), interpolation=INTER_LANCZOS4)

        def convert_to_quad(atlas: np.ndarray) -> np.ndarray:
            return QuadSphere(atlas).quad_sphere_atlas

        if self.use_mars:
            self.down_mars_dem = self.atlas_loader.down_mars_dem
            # TODO Use quad sphere for get_buckets
            self.down_mars_buckets = (
                get_buckets(self.down_mars_dem, planet_cfg)
                if planet_cfg.bucketing_mode == "uniform"
                else None
            )

        self.buckets = (
            # For the buckets, we should technically be using the quad atlas for better area correctness
            # but the normal atlas gives a better sketch
            get_buckets(self.downdem, self.planet_cfg)
            if self.planet_cfg.bucketing_mode == "uniform"
            else None
        )
        self.downsketch = dilate_paint(
            self.downdem, planet_cfg.downscale_cfg, buckets=self.buckets
        )
        self.downland_sketch = landcover_paint(self.downland, planet_cfg.downscale_cfg)
        self.downland_sketch[self.downsketch == 0] = 0

        if (
            "modal" in planet_cfg.input_types + planet_cfg.output_types
            or "downmodal" in planet_cfg.input_types + planet_cfg.output_types
        ):
            self.modal_sketch = ModalSketch(planet_cfg, temp=temp, land=land, sat=sat, use_quad=True)
            # matrix = self.modal_sketch.create_colour_matrix()

        # self.quad_sphere_mapping.quad_sphere_atlas = quad_sat_dem[:, :, :3]
        # Do some extra processing on the temp image to make up for the resizing
        # causing the edges to be colder than they are supposed to be
        self.downtemp[self.downtemp < 1] = 1
        self.downtemp_sketch = temperature_paint(self.downtemp, planet_cfg)

        tile_mask = dilate(self.downsketch, kernel=np.ones((3, 3)), iterations=1)
        if self.use_bathy:
            tile_mask = (tile_mask == 0).astype(np.uint8) * 255
        # This prevents tiles (and neighbouring embeddings) that go off the bottom/top of the image
        # Also there is a mismatch between satellite imagery and others from y 0 to y 455
        tile_mask[: self.down_tile_width // 2 + ceil(455 / (16384 / self.planet_cfg.w)), :] = 0

        tile_mask_H = 256 * 2 ** self.planet_cfg.tile_mask_size
        tile_mask_W = 2 * tile_mask_H
        full_tile_mask = cv2.resize(tile_mask, (tile_mask_W, tile_mask_H), interpolation=cv2.INTER_NEAREST)

        lats, longs = self.sphere_mapping.get_distributed_points(full_tile_mask)

        self.valid_tiles = list(zip(lats, longs))
        if self.use_mars:
            lats, longs = self.mars_quadsphere.get_distributed_points(
                np.ones_like(full_tile_mask)
            )
            self.mars_valid_tiles = list(zip(lats, longs))
            random.Random(0).shuffle(self.mars_valid_tiles)
        random.Random(0).shuffle(self.valid_tiles)
        split_size = min(1000, ceil(0.1 * len(self.valid_tiles)))
        if mode == "train":
            self.valid_tiles = self.valid_tiles[:-split_size]
        elif mode == "val":
            self.valid_tiles = self.valid_tiles[-split_size:]
        elif mode == "test":
            self.valid_tiles = self.valid_tiles[-split_size:]
        self.split = mode

        # data_mask = np.dstack([shared_sat_dem_dict[shape][:, :, 3]] * 3)
        # for lat, long in self.valid_tiles:
        #     y = int((90 - lat) / 180 * H)
        #     x = int((180 + long) / 360 * W)
        #     # data_mask[y, x, 0] = 255
        #     cv2.circle(data_mask, (x, y), 7, (0, 0, 255), -1)
        # img.fromarray(data_mask.astype(np.uint8)).save(os.path.join(planet_cfg.test_dir, f"data_mask_{mode}.png"))
        self.use_modal_rivers = planet_cfg.use_modal_rivers and not self.use_bathy
        self.min_river_size = planet_cfg.min_size
        if self.use_modal_rivers:
            self.river_modal_sketch = RiverModalSketch(planet_cfg)
            shape = (H, W)
            if shape not in shared_rivers_dict:
                rivers = get_stacked_rivers(planet_cfg.data_dir, H, W)
                shared_rivers_dict[shape] = apply_filters(
                    rivers[:, :, 0],
                    rivers[:, :, 1],
                    rivers[:, :, 2],
                )

            quad_rivers_path = os.path.join(
                planet_cfg.data_dir, f"quad_rivers_{W}x{H}.npy"
            )
            if os.path.exists(quad_rivers_path):
                quad_rivers = np.load(quad_rivers_path)
            else:
                self.quad_sphere_mapping.discrete = True
                quad_rivers = self.quad_sphere_mapping.get_quad_sphere_atlas(
                    shared_rivers_dict[shape]
                )
                np.save(quad_rivers_path, quad_rivers)
                self.quad_sphere_mapping.discrete = False

        # TODO Change to just always using quad
        if self.use_quad:
            shape = (H, W)
            mars_shape = (self.atlas_loader.mars_H, self.atlas_loader.mars_W)
            shared_sat_dem_dict[shape] = quad_sat_dem
            shared_land_temp_dict[shape] = quad_land_temp
            if self.use_modal_rivers:
                shared_rivers_dict[shape] = quad_rivers
            if self.use_summer:
                shared_sat_dem_s_dict[shape] = quad_sat_dem_s
                shared_land_temp_s_dict[shape] = quad_land_temp_s
            if self.use_mars:
                shared_mars_sat_dem_dict[mars_shape] = quad_mars_sat_dem
                shared_mars_land_temp_dict[mars_shape] = quad_mars_land_temp
            if self.use_bathy:
                shared_bathy_dict[shape] = quad_bathy
        _land = cv2.resize(self.downland_sketch, (W, H), interpolation=INTER_NEAREST)
        _temp = cv2.resize(self.downtemp_sketch, (W, H), interpolation=INTER_NEAREST)
        if self.use_modal_rivers:
            full_river_sketch = self.river_modal_sketch.get_sketch(
                _land, _temp, shared_rivers_dict[shape]
            )

        # Save all downsketches
        img.fromarray(self.downsketch.astype(np.uint8)).save(
            os.path.join(planet_cfg.test_dir, "dem_sketch.png")
        )
        img.fromarray(translate_land(self.downland_sketch)).save(
            os.path.join(planet_cfg.test_dir, "landcover_sketch.png")
        )
        img.fromarray(gray_to_land(translate_land(self.downland_sketch))).save(
            os.path.join(planet_cfg.test_dir, "rgb_landcover_sketch.png")
        )
        img.fromarray(self.downtemp_sketch).save(
            os.path.join(planet_cfg.test_dir, "temperature_sketch.png")
        )

        if self.use_river_upa:
            river_upa_sketch = get_river_upa_mask(
                shared_river_upa_dict[shape],
                self.river_upa_max,
                replace(
                    self.planet_cfg,
                    river_upa_dropout_chance=0.0,
                    river_upa_variance_chance=0.0,
                )
            )

            img.fromarray(river_upa_sketch).save(
                os.path.join(planet_cfg.test_dir, "river_upa.tif")
            )

        if hasattr(self, "modal_sketch"):
            self.downmodal_sketch = self.modal_sketch.get_sketch(
                self.downland, self.downtemp_sketch
            )
            img.fromarray(self.downmodal_sketch).save(
                os.path.join(planet_cfg.test_dir, "downmodal_sketch.png")
            )
            full_modal_sketch = cv2.resize(
                self.downmodal_sketch, (W, H), interpolation=INTER_NEAREST
            )
            if self.use_modal_rivers:
                current_shape = (self.H, self.W)
                full_modal_sketch[
                    shared_rivers_dict[current_shape] > 0
                ] = full_river_sketch[shared_rivers_dict[current_shape] > 0]
                img.fromarray(full_modal_sketch).save(
                    os.path.join(planet_cfg.test_dir, "full_river_sketch.png")
                )

        # quad_sat = quad_sat_dem[:, :, :3]
        # quad_dem = quad_sat_dem[:, :, 3]
        # quad_land = quad_land_temp[:, :, 0]
        # quad_temp = quad_land_temp[:, :, 1]
        # img.fromarray(quad_sat).save(os.path.join(planet_cfg.data_dir, 'quad_sat.png'))
        # img.fromarray(gray_to_land(quad_land)).save(os.path.join(planet_cfg.data_dir, 'quad_land.png'))
        # img.fromarray(quad_temp).save(os.path.join(planet_cfg.data_dir, 'quad_temp.png'))
        # img.fromarray(quad_dem).save(os.path.join(planet_cfg.data_dir, 'quad_dem.png'))
        # if self.use_modal_rivers:
        #     img.fromarray(quad_rivers).save(os.path.join(planet_cfg.data_dir, 'quad_rivers.png'))

    def __len__(self):
        return len(self.valid_tiles)

    def __getitem__(self, idx):
        return self.get_item(idx)

    @profile
    def get_item(self, idx) -> dict:
        # print memory address of shared_sat_dem and current pid
        # print(
        #   f"shared_sat_dem: {id(shared_sat_dem)} "
        #   f"| pid: {os.getpid()} "
        #   f"| ram: {psutil.Process(os.getpid()).memory_info().rss/1024**3}GB"
        # )
        random_int = random.randint(0, 100)
        is_mars_iter = self.use_mars and random_int % 10 == 0

        lat, long = (
            self.valid_tiles[idx]
            if not is_mars_iter
            else self.mars_valid_tiles[idx % len(self.mars_valid_tiles)]
        )
        is_summer_iter = random_int % 2 == 0

        min_angle = self.min_angle
        max_angle = self.max_angle

        ang = ((2 * np.random.random() - 1.0) * (max_angle - min_angle) + min_angle) or 1
        hflip = random.randint(0, 1)

        if self.planet_cfg.vflip_in_training:
            vflip = random.randint(0, 1)
        else:
            # Always vflip for val/test if not in training
            vflip = self.split != "train"

        return self.get_item_at_coords(
            lat,
            long,
            is_mars_iter,
            is_summer_iter,
            ang,
            hflip,
            vflip,
            idx
        )

    def get_item_at_coords(
        self,
        lat: float,
        long: float,
        is_mars_iter: bool,
        is_summer_iter: bool,
        ang: float,
        hflip: bool,
        vflip: bool,
        idx: int = 0
    ):
        quad_sphere = self.quad_sphere_mapping
        sphere = self.sphere_mapping
        if self.use_quad:
            get_tile_mapping = quad_sphere.get_quad_tile_mapping
            get_tile = quad_sphere.get_quad_tile
            if is_mars_iter:
                get_tile_mapping = self.mars_quadsphere.get_quad_tile_mapping
                get_tile = self.mars_quadsphere.get_quad_tile
        else:
            get_tile_mapping = sphere.get_tile_mapping
            get_tile = sphere.get_tile
            if is_mars_iter:
                get_tile_mapping = self.mars_sphere.get_tile_mapping
                get_tile = self.mars_sphere.get_tile
        tile_w = self.tile_size
        _tile_size = math.ceil(tile_w * math.sqrt(2))
        big_indices = get_tile_mapping(
            coord=(lat, long),
            tile_size=_tile_size,
            **({"round": False} if self.use_quad else {"round_result": False})
        )
        big_indices = tuple([np.nan_to_num(x) for x in big_indices])
        if self.use_quad:
            x, y, z = quad_sphere.quad_coords_to_surface_coords(*big_indices)
            lats, longs = quad_sphere.surface_coords_to_coords(x, y, z)
        else:
            lats, longs = sphere.tile_mapping_to_coords(*big_indices)
        use_rivers = self.use_modal_rivers and not is_mars_iter
        use_river_upa = self.use_river_upa and not is_mars_iter
        mask_names = ["sat_dem", "land_temp"]
        shape = (self.H, self.W)
        masks = (
            [shared_sat_dem_dict[shape], shared_land_temp_dict[shape]]
            if not is_summer_iter or not self.use_summer
            else [shared_sat_dem_s_dict[shape], shared_land_temp_s_dict[shape]]
        )
        if is_mars_iter:
            mars_shape = (self.atlas_loader.mars_H, self.atlas_loader.mars_W)
            masks = [shared_mars_sat_dem_dict[mars_shape], shared_mars_land_temp_dict[mars_shape]]
        # Non-discrete uses bilinear sampling which is supposed to be better quality but slower
        # TODO Consider removing or optimising
        # We have change to bicubic interpolation in the rotation so I am re-enabling it
        discrete = [False, True]
        custom_rotater = [None, None]
        # discrete = [True, True, True, True]

        if use_rivers:
            mask_names.append("rivers")
            masks.append(shared_rivers_dict[shape])
            # For speed purposes we just use the discrete version
            # The small rivers will be broken up a bit but that is okay
            discrete.append(True)
            # custom_rotater.append(lambda x, rotation: line_rotater(x, rotation, min_val=0))
            custom_rotater.append(None)

        if use_river_upa:
            mask_names.append("river_upa")
            masks.append(shared_river_upa_dict[shape])
            discrete.append(self.planet_cfg.discrete_rivers)
            # custom_rotater.append(
            #     lambda x, rotation: line_rotater(x, rotation, min_val=self.planet_cfg.river_upa_min)
            # )
            custom_rotater.append(None)

        if self.use_bathy:
            mask_names = ["bathy", "dem", "boundary_line_sketch"]
            quad_shape = self.quad_sphere_mapping.quad_shape
            masks = [
                shared_bathy_dict[shape],
                shared_sat_dem_dict[shape][:, :, 3],
                shared_quad_boundary_sketch_dict[quad_shape]
            ]
            discrete = [False, True, True]
            custom_rotater = [None, None, lambda x, rotation: line_rotater(x, rotation, min_val=0)]
        masks = {
            name: get_tile((lat, long), ang, tile_w, mask, big_indices, disc, custom_rotate_func=cr)
            for name, mask, disc, cr in zip(mask_names, masks, discrete, custom_rotater)
        }

        if hflip:
            masks = {name: np.fliplr(mask).copy() for name, mask in masks.items()}
        if vflip:
            masks = {name: np.flipud(mask).copy() for name, mask in masks.items()}

        rivers = np.zeros((tile_w, tile_w), dtype=np.uint8)
        river_upa = np.zeros((tile_w, tile_w), dtype=np.float32)

        sat_dem = masks.get("sat_dem", np.zeros((tile_w, tile_w, 4), dtype=np.uint8))
        land_temp = masks.get("land_temp", np.zeros((tile_w, tile_w, 2), dtype=np.uint8))
        sat = sat_dem[:, :, :3]
        dem = sat_dem[:, :, 3]
        dem = masks.get("dem", dem)
        land = land_temp[:, :, 0]
        temp = land_temp[:, :, 1]
        land = translate_land(land, self.planet_cfg.single_water_class)
        land = fill_land(land, sat, temp, dem)

        if self.shuffle_landcover:
            land = randomize_land(land)

        temp = array_variance(temp, self.planet_cfg.max_temp_variance).clip(1, 255)
        # dem = array_variance(dem, 10).clip(0, 255)
        bathy = masks.get("bathy", np.zeros((tile_w, tile_w), dtype=np.float32))
        boundary_line_sketch = masks.get(
            "boundary_line_sketch", np.zeros((tile_w, tile_w), dtype=np.uint8)
        )
        rivers = masks.get("rivers", rivers)
        river_upa = masks.get("river_upa", river_upa)

        mask_ratio = 1.0
        condition = []
        combined_sketch = np.zeros((self.tile_size, self.tile_size))

        delta = self.delta

        lod_levels = abs(self.planet_cfg.sketch_lod_levels)
        lower_lods = self.planet_cfg.sketch_lod_levels < 0

        for _ in range(lod_levels):
            n = random.randint(0, 9)
            if n == 0:
                if lower_lods:
                    delta *= 2
                else:
                    delta //= 2
            else:
                break

        down_shape = (self.tile_size // delta, self.tile_size // delta)
        water = (sat[:, :, 0] < 10) & (sat[:, :, 1] < 30) & (sat[:, :, 2] < 40)

        mask = cv2.resize(dem, down_shape, interpolation=INTER_LANCZOS4)
        mask = dilate_paint(
            mask,
            self.planet_cfg.downscale_cfg,
            buckets=self.down_mars_buckets if is_mars_iter else self.buckets,
        )
        mask = resize(
            mask, (self.tile_size, self.tile_size), interpolation=INTER_NEAREST
        )
        dem_sketch = mask
        mask = modal_resize(land, delta)
        mask = landcover_paint(mask, self.planet_cfg.downscale_cfg)
        if is_mars_iter:
            mask[:, :] = 255
        mask = resize(
            mask, (self.tile_size, self.tile_size), interpolation=INTER_NEAREST
        )
        land_sketch = mask
        _downtemp = cv2.resize(temp, down_shape, interpolation=INTER_LANCZOS4)
        _downtemp[_downtemp < 1] = 1
        _downtemp = temperature_paint(_downtemp, self.planet_cfg.downscale_cfg)
        mask = resize(
            _downtemp,
            (self.tile_size, self.tile_size),
            interpolation=INTER_NEAREST,
        )
        temp_sketch = mask
        mask = get_river_upa_mask(river_upa, self.river_upa_max, self.planet_cfg)
        rivers = (mask > 0).astype(np.uint8) * 255
        has_boundary_data = False

        for mask_type in self.planet_cfg.input_types:
            if "mask" == mask_type:
                # TODO: Add mask size and coastal info
                # mask = create_mask(np.zeros((self.tile_size, self.tile_size)), iw, iw)
                # mask = create_overlap_mask((self.tile_size, self.tile_size))
                # if random.random() < self.planet_cfg.context_dropout:
                #     mask[:, :] = 255
                mask = (~water).astype(np.uint8) * 255
                mask_ratio = (mask > 0).mean()
            elif "oceanmask" == mask_type:
                mask = cv2.resize(water.astype(np.uint8) * 255, down_shape, interpolation=cv2.INTER_NEAREST)
                mask = cv2.resize(mask, (self.tile_size, self.tile_size), interpolation=cv2.INTER_NEAREST)
            elif mask_type == "downland_sketch":
                mask = modal_resize(land, delta)
                mask = landcover_paint(mask, self.planet_cfg.downscale_cfg)
                if is_mars_iter:
                    mask[:, :] = 255
                mask = resize(
                    mask, (self.tile_size, self.tile_size), interpolation=INTER_NEAREST
                )
                land_sketch = mask
                lc = self.planet_cfg.landcover_classes
                if self.planet_cfg.rgb_landcover:
                    mask = gray_to_land(mask)
                elif self.planet_cfg.spread_landcover:
                    mask = continuous_to_spread(mask, lc)
                combined_sketch *= lc + 1
                combined_sketch += (land_sketch + 1) // (255 // lc)
            elif mask_type == "downsketch":
                mask = cv2.resize(dem, down_shape, interpolation=INTER_LANCZOS4)
                mask = dilate_paint(
                    mask,
                    self.planet_cfg.downscale_cfg,
                    buckets=self.down_mars_buckets if is_mars_iter else self.buckets,
                )
                mask = resize(
                    mask, (self.tile_size, self.tile_size), interpolation=INTER_NEAREST
                )
                if self.planet_cfg.add_river_upa_to_dem:
                    processed_upa = get_river_upa_mask(river_upa, self.river_upa_max, self.planet_cfg)
                    mask = mask.astype(np.float32)
                    mask[processed_upa > 0] = processed_upa[processed_upa > 0]
                dem_sketch = mask.astype(np.uint8)
                dc = self.planet_cfg.colours
                combined_sketch *= dc + 1
                combined_sketch += ((mask.astype(np.uint16) + 1) // (256 // dc)).astype(
                    np.uint8
                )
            elif mask_type == "downtemp_sketch":
                _downtemp = cv2.resize(temp, down_shape, interpolation=INTER_LANCZOS4)
                _downtemp[_downtemp < 1] = 1
                _downtemp = temperature_paint(_downtemp, self.planet_cfg.downscale_cfg)
                mask = resize(
                    _downtemp,
                    (self.tile_size, self.tile_size),
                    interpolation=INTER_NEAREST,
                )
                temp_sketch = mask
                combined_sketch *= self.planet_cfg.temp_classes + 1
                combined_sketch += mask
            elif mask_type == "modal":
                raise NotImplementedError("I don't think this is correct")
                mask = modal_resize(temp, delta)
                mask = resize(
                    mask, (self.tile_size, self.tile_size), interpolation=INTER_NEAREST
                )
                _downtemp_sketch = mask
                mask = self.modal_sketch.get_sketch(
                    land, _downtemp_sketch, is_mars_iter
                )
            elif mask_type == "downmodal":
                _downland = modal_resize(land, delta)
                _downtemp = cv2.resize(temp, down_shape, interpolation=INTER_LANCZOS4)
                _downtemp[_downtemp < 1] = 1
                _downtemp = temperature_paint(_downtemp, self.planet_cfg.downscale_cfg)
                missing = (_downland > 0) & (_downtemp == 0)
                _temps, _temp_counts = np.unique(_downtemp, return_counts=True)
                if _temps.shape[0] > 1 and _temps[0] == 0:
                    _temps = _temps[1:]
                    _temp_counts = _temp_counts[1:]
                    _downtemp[missing] = _temps[np.argmax(_temp_counts)]
                mask = self.modal_sketch.get_sketch(_downland, _downtemp, is_mars_iter)
                mask = resize(
                    mask, (self.tile_size, self.tile_size), interpolation=INTER_NEAREST
                )
                _downland = resize(
                    _downland,
                    (self.tile_size, self.tile_size),
                    interpolation=INTER_NEAREST,
                )
                _downtemp = resize(
                    _downtemp,
                    (self.tile_size, self.tile_size),
                    interpolation=INTER_NEAREST,
                )
                land_sketch = _downland
                temp_sketch = _downtemp
                rivers[_downland == 0] = 0
                if (
                    use_rivers and rivers.sum() > self.min_river_size
                ):  # Always use rivers if they are available
                    rivers = filter_components(
                        rivers, min_component_size=self.min_river_size
                    )
                    river_sketch = self.river_modal_sketch.get_sketch(
                        _downland, _downtemp, rivers
                    )
                    mask[rivers > 0] = river_sketch[rivers > 0]

                tc = self.planet_cfg.temp_classes
                combined_sketch *= tc + 1
                combined_sketch += (
                    resize(
                        _downtemp,
                        (self.tile_size, self.tile_size),
                        interpolation=INTER_NEAREST,
                    )
                    + 1
                ) // (256 // tc)
                lc = self.planet_cfg.landcover_classes
                combined_sketch *= lc + 1
                combined_sketch += resize(
                    _downland,
                    (self.tile_size, self.tile_size),
                    interpolation=INTER_NEAREST,
                ) // (256 // lc)

            elif mask_type == "downsat_sketch":
                sat_hex = np_rgb_to_hex(sat)
                mask = modal_resize(sat_hex, 32, exclude_zeros=True, is_hex=True)
                mask = np_hex_to_rgb(mask)
                mask = resize(
                    mask, (self.tile_size, self.tile_size), interpolation=INTER_NEAREST
                )
            elif mask_type == "downbathy_sketch":
                # TODO Use DEM for this
                _land_mask = dem > 0
                mask = bathy.copy()
                mask[mask == 0] = 255
                mask = dilate_paint(
                    mask, self.planet_cfg.downscale_cfg
                )
                mask = modal_resize(mask, delta)
                mask = resize(
                    mask, (self.tile_size, self.tile_size), interpolation=INTER_NEAREST
                )
                if boundary_line_sketch.any():
                    boundary_line_sketch = line_resizer(
                        boundary_line_sketch,
                        (self.tile_size, self.tile_size),
                        do_skeletonize=True,
                        thickness=3,
                    )

                    mask[boundary_line_sketch > 0] = boundary_line_sketch[boundary_line_sketch > 0]
                    has_boundary_data = True
                mask[_land_mask] = 0
            elif mask_type == "river_mask":
                rivers = filter_components(
                    rivers, min_component_size=self.min_river_size
                )
                mask = rivers
            elif mask_type == "river_upa":
                mask = get_river_upa_mask(river_upa, self.river_upa_max, self.planet_cfg)
                river_sketch = (mask > 0).astype(np.uint8) * 255
                if "rivers" not in masks:
                    rivers = (mask > 0).astype(np.uint8) * 255
            else:
                down = False
                discrete = False
                if "down" in mask_type:
                    mask_type = mask_type[4:]
                    down = True

                if mask_type == "dem":
                    mask = dem
                elif mask_type == "land":
                    mask = land
                    discrete = True
                    if self.planet_cfg.rgb_landcover:
                        mask = gray_to_land(mask)
                    elif self.planet_cfg.spread_landcover:
                        mask = continuous_to_spread(
                            mask, self.planet_cfg.landcover_classes
                        )
                elif mask_type == "sat":
                    mask = sat
                elif mask_type == "temp":
                    mask = temp
                else:
                    raise ValueError(f"Unknown output mask type {mask_type}")
                if down:
                    if discrete:
                        mask = modal_resize(mask, delta)
                        mask = resize(
                            mask,
                            (self.tile_size, self.tile_size),
                            interpolation=INTER_NEAREST,
                        )
                    else:
                        mask = upscale_regularization(mask, self.planet_cfg, self.tile_size, delta)
                if mask_type == "dem" and self.planet_cfg.add_river_upa_to_dem:
                    processed_upa = get_river_upa_mask(river_upa, self.river_upa_max, self.planet_cfg)
                    mask = mask.astype(np.float32)
                    mask[processed_upa > 0] = processed_upa[processed_upa > 0]
            condition.append(mask)
        combined_sketch += 1

        target = []
        for mask_type in self.planet_cfg.output_types:
            if mask_type == "dem":
                mask = dem
            elif mask_type == "land":
                mask = land
                if self.planet_cfg.rgb_landcover:
                    mask = gray_to_land(mask)
                elif self.planet_cfg.spread_landcover:
                    mask = continuous_to_spread(mask, self.planet_cfg.landcover_classes)
            elif mask_type == "sat":
                mask = sat
            elif mask_type == "temp":
                mask = temp
            elif mask_type == "bathy":
                mask = resize(
                    bathy, (self.tile_size, self.tile_size), interpolation=INTER_NEAREST
                )
            else:
                raise ValueError(f"Unknown output mask type {mask_type}")
            target.append(mask)

        def resize_masks(mask_types: list[str], arr: list[np.ndarray]):
            for i, (mask_type, mask) in enumerate(zip(mask_types, arr)):
                discrete = "land" in mask_type or "sketch" in mask_type
                _h, _w = mask.shape[:2]
                arr[i] = (
                    modal_resize(mask, 2)
                    if discrete
                    else resize(mask, (_w // 2, _h // 2), interpolation=cv2.INTER_LANCZOS4)
                )

        # resize_masks(self.planet_cfg.input_types, condition)
        # resize_masks(self.planet_cfg.output_types, target)

        condition = np.dstack(condition)
        target = np.dstack(target)

        # TODO: Colour dropout
        to_return = {}
        metadata = {
            "range": np.max(target) - np.min(target),
            "zoom": 2**self.planet_cfg.size,
            "tile_y": lat,
            "tile_x": long,
            "lat": lat,
            "long": long,
            "vflip": vflip,
            "hflip": hflip,
            "k": 0,
            "factor": self.planet_cfg.size,
            "delta": delta,
            "angle": ang,
            "resolution": 0,  # 84375.0/(2**self.planet_cfg.size),
            "idx": idx,
            "tile_size": self.tile_size,
            "mask_ratio": mask_ratio,
            "is_mars": is_mars_iter,
            "is_summer": is_summer_iter,
        }
        if self.planet_cfg.input_index("mask"):
            metadata["mask_channel"] = self.planet_cfg.input_index("mask")
        if self.do_transforms:
            to_return["target_image"] = self.target_transform(
                target.astype(np.float32) / 255.0
            )
            to_return["cond_image"] = self.cond_transform(
                condition.astype(np.float32) / 255.0
            )
        else:
            to_return["target_image"] = target
            to_return["cond_image"] = condition
        to_return["metadata"] = metadata
        to_return["combined_sketch"] = combined_sketch.astype(np.int16)

        if self.use_bathy or self.planet_cfg.embedding_type == "disabled":
            embedding = np.zeros(encoder_size(self.planet_cfg), dtype=np.float32)
            if has_boundary_data:
                # For now only use the boundary line sketch in encoding
                embedding[:] = 1.0
        else:
            try:
                embedding = _simple_encode(
                    dem_sketch,
                    land_sketch,
                    temp_sketch,
                    self.planet_cfg,
                    delta,
                    rivers=rivers,
                    dem=dem,
                    land=land,
                    sat=sat,
                )
            except Exception:
                print("Unable to get embedding, using zeros")
                embedding = np.zeros(encoder_size(self.planet_cfg), dtype=np.float32)
        to_return["metadata"]["embedding"] = embedding

        if random.random() < self.planet_cfg.encoder_dropout:
            to_return["metadata"]["embedding"] = np.zeros_like(
                to_return["metadata"]["embedding"]
            )

        return to_return

    @property
    def sat(self):
        shape = (self.H, self.W)
        return shared_sat_dem_dict[shape][:, :, :3]

    @property
    def dem(self):
        shape = (self.H, self.W)
        return shared_sat_dem_dict[shape][:, :, 3]

    @property
    def land(self):
        shape = (self.H, self.W)
        return shared_land_temp_dict[shape][:, :, 0]

    @property
    def temp(self):
        shape = (self.H, self.W)
        return shared_land_temp_dict[shape][:, :, 1]


def mode(x: np.ndarray, exclude_zeros: bool = False):
    values, counts = np.unique(x, return_counts=True)
    if len(values) > 1 and exclude_zeros:
        counts = counts[values != 0]
        values = values[values != 0]
    return values[np.argmax(counts)]


def distribution(x: np.ndarray, max_value: int) -> np.ndarray:
    values, counts = np.unique(x.astype(np.int32), return_counts=True)
    res = np.zeros(max_value + 1)
    res[values] = counts / counts.sum()
    return res


def onehot(i: int, n: int) -> np.ndarray:
    onehot = np.zeros(n)
    onehot[i] = 1
    return onehot.astype(np.uint8)


def has_coastline(dem: np.ndarray) -> bool:
    """
    Check if there is a coastline in the tile
    """
    water: np.ndarray = dem == 0
    return water.mean() > 0.1


def has_mountains(dem: np.ndarray) -> bool:
    """
    Check if there are mountains in the tile
    """
    return dem.max() > 150


def has_snow_cover(land_mode: int) -> bool:
    """
    Check if there is snow cover in the tile
    """
    return (
        land_mode == LandcoverClasses.ANTARCTICA.index
        or land_mode == LandcoverClasses.SNOW_AND_ICE.index
    )


def has_rivers(rivers: np.ndarray) -> bool:
    """
    Check if there are rivers in the tile
    """
    # Check that there is a river that is at least half the length of the tile
    return (rivers > 0).mean() > 0.5 * 1 / 256


def has_islands(dem: np.ndarray) -> bool:
    """
    Check if there are islands in the tile
    """
    # We can use opening to remove small islands and check the difference
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    landmasses = (dem > 0).astype(np.uint8) * 255
    opening = cv2.morphologyEx(landmasses, cv2.MORPH_OPEN, kernel, iterations=2)
    # Make a frame around the image to remove to prevent lakes from creating false positives
    opening[0, :] = 255
    opening[-1, :] = 255
    opening[:, 0] = 255
    opening[:, -1] = 255
    labeled, ncomponents = label(opening)
    # TODO Consider making this a float rather than a bool to differentiate between a few and many islands
    return ncomponents > 5


def has_lakes(sat: np.ndarray, dem: np.ndarray) -> bool:
    """
    Check if there are lakes in the tile
    """
    # We can use opening to remove small lakes and check the difference
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    water = (sat[:, :, 0] < 10) & (sat[:, :, 1] < 30) & (sat[:, :, 2] < 40)
    water[dem == 0] = 0
    water = cv2.morphologyEx(
        water.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel, iterations=2
    )
    return water.any()


class EncoderOverride:
    def __init__(
        self,
        coastline: bool | None = None,
        mountains: bool | None = None,
        snow_cover: bool | None = None,
        rivers: bool | None = None,
        islands: bool | None = None,
        icebergs: bool | None = None,
        lakes: bool | None = None,
    ):
        self.coastline = coastline
        self.mountains = mountains
        self.snow_cover = snow_cover
        self.rivers = rivers
        self.islands = islands
        self.icebergs = icebergs
        self.lakes = lakes


def encoder_size(planet_cfg: PlanetConfig) -> int:
    # TODO Change to 512
    return 9 * (
        planet_cfg.colours + planet_cfg.landcover_classes + planet_cfg.temp_classes + 4
    )


@profile
def _simple_encode(
    dem_sketch: np.ndarray,
    land_sketch: np.ndarray,
    temp_sketch: np.ndarray,
    planet_cfg: PlanetConfig,
    delta: int,
    rivers: np.ndarray = None,
    dem: np.ndarray = None,
    land: np.ndarray = None,
    sat: np.ndarray = None,
    is_mars: bool = False,
    encoder_override: EncoderOverride = EncoderOverride(),
    # TODO Add feature vec as a parameter for inference
) -> np.ndarray:
    """
    Get the embedding for the
    """
    if planet_cfg.embedding_type == "disabled":
        return np.zeros(encoder_size(planet_cfg), dtype=np.float32)

    if rivers is None:
        # TODO Actually derive the river here if it is not provided
        rivers = np.zeros_like(dem_sketch) + 255
    if dem is None:
        dem = dem_sketch.copy()
    if land is None:
        land = land_sketch.copy()
    if sat is None:
        sat = ModalSketch(planet_cfg, use_quad=True).get_sketch(land, temp_sketch)

    # This is to keep compatibility with the old code
    encoding_size = encoder_size(planet_cfg)

    shape = dem_sketch.shape
    resize_shape = (shape[1] // delta, shape[0] // delta)
    dem_int = planet_cfg.dem_to_int(
        cv2.resize(dem_sketch, resize_shape, interpolation=cv2.INTER_NEAREST)
    )
    land_int = planet_cfg.landcover_to_int(
        cv2.resize(land_sketch, resize_shape, interpolation=cv2.INTER_NEAREST)
    )
    temp_int = planet_cfg.temp_to_int(
        cv2.resize(temp_sketch, resize_shape, interpolation=cv2.INTER_NEAREST)
    )

    # We want to encode the mode class of each sketch type for the current sketch to easily distinguish between them
    # dem_mode = mode(dem_int)
    land_mode = mode(land_int)
    # temp_mode = mode(temp_int)

    if is_mars:
        land_mode = LandcoverClasses.BUILT_UP.index
        # Just use this one because it is the most uncommon.
        # TODO Actually merge built up into a different landcover class

    dem_encoding = distribution(dem_int, planet_cfg.colours)
    land_encoding = distribution(land_int, planet_cfg.landcover_classes + 1)
    temp_encoding = distribution(temp_int, planet_cfg.temp_classes)
    if planet_cfg.embedding_type == "one-hot":
        dem_encoding = onehot(dem_encoding.argmax(), dem_encoding.size)
        land_encoding = onehot(land_encoding.argmax(), land_encoding.size)
        temp_encoding = onehot(temp_encoding.argmax(), temp_encoding.size)

    # TODO We can then append a multi-hot encoding of the other features
    # e.g. Coastline, mountains, snow cover, rivers, islands, icebergs, lakes, etc
    # This will be a bit more complex but should be doable
    _coastline = encoder_override.coastline or (
        encoder_override.coastline is None and not is_mars and has_coastline(dem)
    )
    _snow_cover = encoder_override.snow_cover or (
        encoder_override.snow_cover is None and has_snow_cover(land_mode)
    )
    _mountains = encoder_override.mountains or (
        encoder_override.mountains is None
        and has_mountains(dem)
        and not land_mode == LandcoverClasses.ANTARCTICA.index
    )
    _rivers = encoder_override.rivers or (
        encoder_override.rivers is None and not is_mars and has_rivers(rivers)
    )
    _islands = encoder_override.islands or (
        encoder_override.islands is None and not is_mars and has_islands(dem)
    )
    _icebergs = encoder_override.icebergs or (
        encoder_override.icebergs is None and _islands and _snow_cover
    )
    _lakes = encoder_override.lakes or (
        encoder_override.lakes is None and not is_mars and has_lakes(sat, dem)
    )

    other_features = np.array(
        [
            _coastline,
            _mountains,
            _snow_cover,
            _rivers,
            _islands,
            _icebergs,
            _lakes,
            is_mars,
        ]
    ).astype(np.uint8)

    zero_vec = np.zeros(encoding_size)
    feature_vec = np.concatenate(
        (dem_encoding, land_encoding, temp_encoding, other_features)
    )
    zero_vec[: feature_vec.shape[0]] = feature_vec
    return zero_vec.astype(np.float32)


@profile
def _encode(
    sphere: SphereMapping | QuadSphere,
    metadata: dict,
    planet_cfg: PlanetConfig,
    full_sketch: np.ndarray,
    full_land_sketch: np.ndarray,
    full_temp_sketch: np.ndarray,
    normal_full_sketch: np.ndarray | None = None,
    normal_full_land_sketch: np.ndarray | None = None,
    normal_full_temp_sketch: np.ndarray | None = None,
):
    """
    Encode the conditional image
    """
    if "embedding" in metadata:
        return metadata["embedding"]
    tile_size = metadata["tile_size"]
    delta = metadata["delta"]
    angle = metadata["angle"]
    long = metadata["tile_x"]
    lat = metadata["tile_y"]
    hflip = metadata["hflip"]
    vflip = metadata["vflip"]
    tile_w = tile_size // delta
    surround_w = 3 * tile_w

    masks = [full_sketch, full_land_sketch, full_temp_sketch]
    failed = False
    if type(sphere) is QuadSphere:
        try:
            big_indices = sphere.get_quad_tile_mapping(
                coord=(lat, long), tile_size=math.ceil(surround_w * math.sqrt(2))
            )
            masks = map(
                lambda x: sphere.get_quad_tile(
                    (lat, long), angle, tile_w, x, big_indices, True
                ),
                masks,
            )
        except NotImplementedError:
            failed = True
    if failed or type(sphere) is SphereMapping:
        if type(sphere) is QuadSphere:
            masks = [
                normal_full_sketch,
                normal_full_land_sketch,
                normal_full_temp_sketch,
            ]
            missing = [mask is None for mask in masks]
            if True in missing:
                masks = map(lambda x: sphere.get_normal_atlas(x), masks)
        big_indices = sphere.get_tile_mapping(
            coord=(lat, long), tile_size=math.ceil(surround_w * math.sqrt(2))
        )
        masks = map(
            lambda x: sphere.get_tile((lat, long), angle, tile_w, x, big_indices, True),
            masks,
        )
    if hflip:
        masks = map(lambda x: np.fliplr(x).copy(), masks)
    if vflip:
        masks = map(lambda x: np.flipud(x).copy(), masks)
    sketch_surrounds, land_sketch_surrounds, temp_sketch_surrounds = masks

    # We want to encode the following features:
    # Data from surrounding 8 sketch tiles
    # - Percentage of each elevation colour 9 x ~4
    # - Percentage of each landcover class 9 x 8
    # - Distance to ocean in all 9 directions 9 x 1 from edges of tile or middle

    # Maybe:
    # - Continent size 1

    planet_colours = planet_cfg.colours
    planet_classes = planet_cfg.landcover_classes
    planet_temps = planet_cfg.temp_classes

    # We add one here for the ocean which isn't included in the colour or class count
    sketch_colours = np.zeros((9, planet_colours + 1))
    landcover_classes = np.zeros((9, planet_classes + 1))
    temp_classes = np.zeros((9, planet_temps + 1))
    # distances = np.zeros(9)  # TODO Do this again well with sphere mapping
    inland_embedding = np.zeros(9)  # 0.0 for ocean, 0.5 for coastal, 1.0 for inland
    for i, k in enumerate(range(9)):
        left = (k % 3) * tile_w
        top = (k // 3) * tile_w
        colours, col_counts = np.unique(
            sketch_surrounds[top: top + tile_w, left: left + tile_w],
            return_counts=True,
        )
        colours = colours.astype(np.uint16)
        colours[colours > 0] += 1
        colour_step = 256 // planet_colours
        assert np.array_equal(
            colours // colour_step * colour_step, colours
        ), f"Colour values: {colours}, Colour step: {colour_step}"
        colours //= colour_step
        classes, class_counts = np.unique(
            land_sketch_surrounds[top: top + tile_w, left: left + tile_w],
            return_counts=True,
        )
        class_step = 256 // planet_classes
        assert np.array_equal(
            classes // class_step * class_step, classes
        ), f"Class values: {classes}, Class step: {class_step}"
        classes //= class_step
        temp_values, temp_counts = np.unique(
            temp_sketch_surrounds[top: top + tile_w, left: left + tile_w],
            return_counts=True,
        )
        temp_values = temp_values.astype(np.uint16)
        temp_values[temp_values > 0] += 1
        temp_step = 255 // planet_temps
        assert np.array_equal(
            temp_values // temp_step * temp_step, temp_values
        ), f"Temp values: {temp_values}, Temp step: {temp_step}"
        temp_values //= temp_step

        sketch_colours[i, colours] = col_counts / (tile_w * tile_w)
        landcover_classes[i, classes] = class_counts / (tile_w * tile_w)
        temp_classes[i, temp_values] = temp_counts / (tile_w * tile_w)
        if sketch_colours[i][0] == 1.0:
            inland_embedding[i] = 0.0  # All ocean
        elif sketch_colours[i][0] > 0:
            inland_embedding[i] = 0.5  # Coastal
        else:
            inland_embedding[i] = 1.0  # Inland
    to_return = np.concatenate(
        (
            sketch_colours.flatten(),
            landcover_classes.flatten(),
            temp_classes.flatten(),
            inland_embedding,
        )
    )
    return to_return.astype(np.float32)


@dataclass
class DataLoaderArgs:
    num_workers: int = field(default=0, metadata={"help": "Number of worker threads"})
    batch_size: int = field(default=1, metadata={"help": "Batch size"})
    tile_size: int = field(default=256, metadata={"help": "Tile size"})
    mode: str = field(default="train", metadata={'choices': ["train", "val", "test"]})


if __name__ == "__main__":
    parser = CustomArgumentParser(
        (PlanetConfig, DataLoaderArgs), description="Benchmark dataset loading speed"
    )
    args_dataclasses: tuple[PlanetConfig, DataLoaderArgs] = parser.parse_args_into_dataclasses()
    test_cfg: PlanetConfig = args_dataclasses[0]
    dataloader_args: DataLoaderArgs = args_dataclasses[1]
    # setup(test_cfg, force=False)
    dataset = RAMDataset(
        test_cfg,
        target_image_channels=test_cfg.output_channels(),
        cond_image_channels=test_cfg.input_channels(),
        normalise=True,
        tile_size=dataloader_args.tile_size,
        mode=dataloader_args.mode,
    )
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        num_workers=dataloader_args.num_workers,
        batch_size=dataloader_args.batch_size
    )
    iters = test_cfg.iters
    iters //= dataloader_args.batch_size
    if len(dataloader) < iters:
        iters = len(dataloader)
    t1 = time()
    im_grid = []
    cols = 10
    with tqdm(total=iters) as pbar:
        for i, batch in enumerate(dataloader):
            if test_cfg.image_mode == 'bathy':
                try:
                    sketch_ims = batch["cond_image"][:, 0]
                    bathy_ims = batch["target_image"][:, 0]
                except Exception:
                    iters += 1
                    continue
                for bathy_im, sketch_im in zip(bathy_ims, sketch_ims):
                    bathy_im = tensor_to_np(bathy_im)
                    sketch_im = tensor_to_np(sketch_im)
                    combined = np.concatenate((sketch_im, bathy_im), axis=1)
                    im_grid.append(img.fromarray(combined))
            else:
                display_tensor = test_cfg.output_display(
                    batch["cond_image"],
                    batch["target_image"],
                    batch["target_image"],
                    show_outputs=False
                )
                c, h, w = display_tensor.shape
                im_grid.append(img.fromarray(tensor_to_np(display_tensor)).resize((512 * w // h, 512), img.NEAREST))

            if i >= iters:
                break
            pbar.update(1)

    print(
        f"batch_size: {dataloader_args.batch_size} "
        f"| num_workers: {dataloader_args.num_workers} "
        f"| image_mode: {test_cfg.image_mode}",
        end="",
    )
    print(f" | {dataloader_args.batch_size*iters/(time() - t1):.2f} it/s")
    im_grid = image_grid(im_grid, int(math.ceil(len(im_grid) / cols)), cols)
    im_grid.save(os.path.join(test_cfg.test_dir, "benchmark.png"))
