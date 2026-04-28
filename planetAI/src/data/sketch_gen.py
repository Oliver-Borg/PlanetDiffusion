try:
    from utils import call_depth, brush_mask, PlanetConfig, get_brush_deltas, image_grid, profile, np_rgb, timing
    from landcover_utils import translate_land
except:
    from .utils import call_depth, brush_mask, PlanetConfig, get_brush_deltas, image_grid, profile, np_rgb, timing
    from .landcover_utils import translate_land


import itertools
from PIL import Image as img
from PIL.Image import Image
import numpy as np
import os
from tqdm import tqdm
from math import sqrt, log2
import json
from scipy.ndimage import label
from scipy.spatial.distance import cdist
from time import time
from skimage.morphology import skeletonize
from pysheds.view import Raster, ViewFinder
from pysheds.grid import Grid
from affine import Affine
import pyproj
import cv2
from cv2 import GaussianBlur, erode, dilate
import warnings
from dataclasses import replace

from planetAI.src.data.landcover_utils import gray_to_land
from planetAI.src.data.utils import timing

img.MAX_IMAGE_PIXELS = 933120000
# 512	256
# 1024	512
# 2048	1024
# 4096	2048
# 8192	4096
# 16384	8192

global_sizes = [
    (512, 256),
    (1024, 512),
    (2048, 1024),
    (4096, 2048),
    (8192, 4096),
    (16384, 8192),
]

MAX_LOGGING_DEPTH = call_depth() + 2


def quantize(x: int, colours: int=8, max_value: int=255) -> int:
    '''
    DEPRECATED
    Given an input x, quantize it to it's bucket in the colour list.
    Args:
        x (int): The pixel colour to quantize
    Returns:
        int: The quantized pixel colour
    '''
    bucket_size = (max_value+1)//colours
    colour_interval = 256//colours
    return int(min((x//bucket_size+1) * colour_interval - 1, colour_interval * colours - 1))

def quantize_list(pixels: list, colours: int=8, uniform=False, use_max_pixel=False) -> dict:
    '''
    DEPRECATED
    Quantize a list of pixel colours to the given number of colours.
    Args:
        pixels (list): The list of pixel colours with coordinates to quantize
        colours (int, optional): The number of colours to quantize to. Defaults to 8.
        uniform (bool, optional): Whether to use uniform bucket sizes. Defaults to False.
        use_max_pixel (bool, optional): Whether to use the maximum pixel value as the maximum colour value. Defaults to False.
    Returns:
        dict: A dictionary of uniformly distributed colours and associated pixel coordinates
    '''
    warnings.warn("This function is deprecated. Use numpy_get_pixel_lists instead.", DeprecationWarning)
    pixel_lists = {}
    max_colour = max(pixels, key=lambda x: x[0])[0] if use_max_pixel else 255

    if uniform:
        # Uniform colours
        # Uniform bucket sizes
        pixels.sort(key=lambda x: x[0])
        bucket_size = len(pixels) // colours
        step_size = 256//colours
        for i, (colour, (y, x)) in enumerate(pixels):
            col = min((i // bucket_size + 1) * step_size - 1, step_size * colours - 1) 
            if col not in pixel_lists:
                pixel_lists[col] = []
            pixel_lists[col].append((int(y), int(x)))
    else:
        # Uniform colours
        # Non-uniform bucket sizes
        for colour, (y, x) in pixels:
            col = quantize(colour, colours, max_colour)
            if col not in pixel_lists:
                pixel_lists[col] = []
            pixel_lists[col].append((int(y), int(x)))
    
    return pixel_lists


def get_outline_pixels(oceanmask: np.ndarray) -> np.ndarray:
    '''
    Get a list of pixels that border the ocean in the oceanmask.
    This is used to create the outline of a mask once it has been painted without having to paint the whole mask.

    Args:
        oceanmask (np.ndarray): The oceanmask to get the outline of
    Returns:
        np.ndarray: A list of pixel tuples that border the ocean in the oceanmask
    '''
    outline_pixels = []
    h, w = oceanmask.shape
    with tqdm(total=1, desc=f"Getting outline pixels for {w}x{h}...", disable=(MAX_LOGGING_DEPTH<call_depth())) as pbar:
        borders = border_mask(oceanmask > 30)
        outline_pixels = np.argwhere(borders)
        pbar.update(1)
    return outline_pixels

def create_mask(oceanmask: np.ndarray, brush_size: int, force: bool=False, data_dir=os.path.join(os.getcwd(), "data")) -> None:
    '''
    Create a mask for the oceanmask outline with the given brush size.
    It also sketches the original outline in the mask to allow the original mask to be seen.
    Args:
        oceanmask (np.ndarray): The oceanmask to create the mask for
        brush_size (int): The size of the brush to use to create the mask
        force (bool, optional): Whether to force the creation of the mask even if it already exists. Defaults to False.
        data_dir (str, optional): The directory to save the mask to. Defaults to os.path.join(os.getcwd(), "data").
    Returns:
        None (Saves the mask to the data/masks folder)
    '''
    h, w = oceanmask.shape
    output_folder = os.path.join(data_dir, "masks", f"{w}x{h}", f"brush_{brush_size}")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_dir = os.path.join(output_folder, "mask.png")
    if os.path.exists(output_dir) and not force:
        return
    borders = border_mask(oceanmask > 30)
    bm = brush_mask(borders, brush_size)

    with tqdm(total=1, desc=f"Outlining mask {w}x{h}...", disable=(MAX_LOGGING_DEPTH<call_depth())) as pbar:
        oceanmask[bm] = 255
        pbar.update(1)
    with tqdm(total=1, desc=f"Sketching mask {w}x{h} outline...", disable=(MAX_LOGGING_DEPTH<call_depth())) as pbar:
        oceanmask[borders] = 128
        pbar.update(1)   
    
    img.fromarray(oceanmask).save(output_dir)

def create_masks(brush_size: int, sizes: list=[(512, 256)], force: bool=False, data_dir=os.path.join(os.getcwd(), "data")) -> None:
    '''
    Create masks for all the oceanmasks in the data folder with the given brush size.
    Args:
        brush_size (int): The size of the brush to use to create the masks
        sizes (list, optional): A list of size tuples (width, height) to create masks for. Defaults to [(512, 256)].
        force (bool, optional): Whether to force the creation of the masks even if they already exist. Defaults to False.
        data_dir (str, optional): The directory to save the masks to. Defaults to os.path.join(os.getcwd(), "data").    
    Returns:
        None (Saves the masks to the data/masks folder)
    '''
    for w, h in sizes:
        oceanmask = np.array(img.open(os.path.join(data_dir, f'world.oceanmask.{w}x{h}.png')))
        create_mask(oceanmask, brush_size, force, data_dir)

def change_sea_level(dem: np.ndarray, sea_level: int, rescale: bool=True) -> np.ndarray:
    '''
    Change the sea level of the DEM.
    Args:
        dem (np.ndarray): The DEM to change the sea level of
        sea_level (int): The sea level to change the DEM to
        rescale (bool, optional): Whether to rescale the DEM to the maximum land value. Defaults to True.
    Returns:
        np.ndarray: The DEM with the new sea level
    '''
    land = dem.copy()
    land[land <= sea_level] = 0
    max_land = land.max()
    land[land > sea_level] -= sea_level
    if rescale:
        land = land.astype(np.float32) / (max_land - sea_level) * max_land
        land = land.astype(np.uint8)
        assert land.max() == max_land, f"Max land value {land.max()} is not equal to max land {max_land}"
    return land

def rgb_to_h(r: int, g: int, b: int) -> int:
    '''
    Get the hue of an RGB colour.
    '''
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    diff = cmax-cmin
    if cmax == cmin: 
        return 0
    elif cmax == r: 
        return (60 * ((g - b) / diff) + 360) % 360
    elif cmax == g:
        return (60 * ((b - r) / diff) + 120) % 360
    elif cmax == b:
        return (60 * ((r - g) / diff) + 240) % 360

def rgb_to_h_img(img: np.ndarray) -> np.ndarray:
    '''
    Convert an RGB image to an H image.
    Args:
        img (np.ndarray): The RGB image to convert
    Returns:
        np.ndarray: The H image
    '''
    h_img = np.zeros_like(img[:, :, 0])
    r_channel = img[:,:,0]/255
    g_channel = img[:,:,1]/255
    b_channel = img[:,:,2]/255
    cmax = np.max(img, axis=2)/255
    cmin = np.min(img, axis=2)/255
    diff = cmax-cmin
    mask = (cmax == cmin)
    h_img[mask] = 0
    mask = (cmax == r_channel)
    h_img[mask] = (60 * ((g_channel - b_channel) / diff) + 360)[mask] % 360
    mask = (cmax == g_channel)
    h_img[mask] = (60 * ((b_channel - r_channel) / diff) + 120)[mask] % 360
    mask = (cmax == b_channel)
    h_img[mask] = (60 * ((r_channel - g_channel) / diff) + 240)[mask] % 360

    return h_img

def get_satellite_pixel_lists(sat_img: np.ndarray, dem: np.ndarray, planet_cfg: PlanetConfig, h_dist: bool=False) -> dict:
    '''
    Gets the lists of pixels associated with each colour in the colour list 
    Args:
        sat_img (np.ndarray): The satellite image to get the pixel lists for
        dem (np.ndarray): The DEM to use to get the land pixels
        planet_cfg (PlanetConfig): The planet config to use to get the bucketing mode and colours
        colour_list (list, optional): The list of colours to get the pixel lists for. Defaults to the default satellite colours.
        h_dist (bool, optional): Whether to use hue distance instead of euclidean distance. Defaults to False.
    Returns:
        dict: A dictionary of colour buckets to lists of pixels
    '''
    pixel_lists = {}
    dists = []
    # TODO: Optimize by using two arrays for storing min distance and associated colour
    colour_list = planet_cfg.satellite_colours
    np_img = sat_img
    np_img[dem <= planet_cfg.sea_level] = [0, 0, 0]
    
    land_mask = (dem != 0).astype(np.uint8)*255
    land_mask = cv2.erode(land_mask, 
                        np.ones((planet_cfg.erode_size, planet_cfg.erode_size), np.uint8), 
                        iterations=planet_cfg.erode_iters)
    np_img[land_mask == 0] = [0, 0, 0]
    for colour in colour_list:
        r = (colour & 0xff0000) >> 16
        g = (colour & 0x00ff00) >> 8
        b = (colour & 0x0000ff)
        if h_dist:
            dist = np.abs(rgb_to_h_img(np_img)-rgb_to_h(r, g, b))
        else:
            dist_r = (np_img[:,:,0].astype(np.int32) - r) ** 2
            dist_g = (np_img[:,:,1].astype(np.int32) - g) ** 2
            dist_b = (np_img[:,:,2].astype(np.int32) - b) ** 2
            rmean = (np_img[:,:,0].astype(np.int32) + r) / 2
            dist = (2+rmean>128)*dist_r + 4*dist_g + (2+rmean<=128)*dist_b
            dist = dist_r + dist_g + dist_b
        dists.append(dist)
    dists = np.stack(dists, axis=2)
    colours = np.argmin(dists, axis=2)
    for i, colour in enumerate(colour_list):
        pixel_lists[colour] = np.argwhere(colours == i)
    return pixel_lists

def paint_satellite_image(np_img: np.ndarray, dem: np.ndarray, planet_cfg: PlanetConfig) -> np.ndarray:
    '''
    Paints the image with the given colours
    Args:
        np_img (np.ndarray): The image to paint
        landmask (np.ndarray): The dem to use to get the land pixels
        planet_cfg (PlanetConfig): The planet config to use to get the bucketing mode and colours
    Returns:
        np.ndarray: The painted image
    '''
    brush_size = planet_cfg.brush_size
    brush_deltas = get_brush_deltas(brush_size)
    sketch = np.zeros_like(np_img)
    gaussian_blur = planet_cfg.gaussian_blur
    blur_size = planet_cfg.blur_size
    np_img = np_img.copy()
    np_img[dem == 0] = [0, 0, 0]
    pixel_lists = get_satellite_pixel_lists(GaussianBlur(np_img, (blur_size, blur_size), gaussian_blur), dem, planet_cfg)
    for colour, pixels in pixel_lists.items():
        r = (colour & 0xff0000) >> 16
        g = (colour & 0x00ff00) >> 8
        b = (colour & 0x0000ff)
        mask = np.zeros((np_img.shape[0], np_img.shape[1]), dtype=bool)
        if len(pixels) == 0:
                continue
        mask[tuple(np.array(pixels).T)] = True
        sketch[mask] = [r, g, b]
        shifted_mask = np.zeros_like(mask, dtype=bool)
        for dx, dy in brush_deltas:
            roll = np.roll(mask, (dy, dx), axis=(0, 1))
            if dy > 0:
                roll[:dy,:] = False
            elif dy < 0:
                roll[dy:,:] = False
            shifted_mask |= roll

        sketch[shifted_mask] = [r, g, b]
    if planet_cfg.preserve_edges:
        sketch[dem == 0] = [0, 0, 0]
    return sketch


def numpy_get_pixel_lists(np_img: np.ndarray, landmask: np.ndarray, planet_cfg: PlanetConfig) -> dict:
    '''
    Get the lists of pixels as a dictionary for each colour bucket in the img.
    Args:
        np_img (np.ndarray): The image to get the pixel lists for
        landmask (np.ndarray): The landmask to use to filter out land pixels
        planet_cfg (PlanetConfig): The planet config to use to get the bucketing mode and colours
    Returns:
        dict: A dictionary of colour buckets to lists of pixels
    '''        

    colours = planet_cfg.colours
    uniform = planet_cfg.bucketing_mode == 'uniform'
    use_global_max = planet_cfg.bucketing_mode == 'global-max'
    use_local_max = planet_cfg.bucketing_mode == 'local-max'
    max_colour = planet_cfg.get_max_pixel(np_img)
    sea_level = planet_cfg.sea_level
    rescale = planet_cfg.sea_level_rescale

    assert max_colour >= np_img.max(), f"Max colour {max_colour} is less than max pixel value {np_img.max()}"
    pixel_lists = {}
    mask = (np_img > 0) & (landmask > 200)

    land = np_img.copy()
    if sea_level != 0:
        land = change_sea_level(land, sea_level, rescale)
    land[~mask] = 0
    # Apply gaussian blur here
    if planet_cfg.gaussian_blur > 0:
        blur_size = planet_cfg.blur_size
        land = GaussianBlur(land, (blur_size, blur_size), planet_cfg.gaussian_blur)

    # pixels = np.argwhere(mask)
    
    step_size = (max_colour+1)//colours
    step_list = [i for i in range(step_size-1, max_colour+1, step_size)] 
    colour_size = 256 // colours
    colour_list = [i*colour_size-1 for i in range(1, colours+1)]
    for colour in colour_list:
        pixel_lists[colour] = []
    if uniform:
        # Uniform colours
        # Uniform bucket sizes
        sorted = np.argsort(land.ravel())
        sorted = np.unravel_index(sorted, land.shape)
        l = np.count_nonzero(land)
        zeros = land.size - l
        bucket_size = l // colours
        for i, col in enumerate(colour_list):
            pixel_lists[col] = list(zip(sorted[0][zeros + i*bucket_size:zeros + (i+1)*bucket_size],
                                        sorted[1][zeros + i*bucket_size:zeros + (i+1)*bucket_size]))
        pixel_lists[colour_list[-1]] += list(zip(sorted[0][zeros + colours*bucket_size:],
                                        sorted[1][zeros + colours*bucket_size:]))

        
        
    else:
        # Uniform colours
        # Non-uniform bucket sizes
        step_list = [0] + step_list
        step_list[-1] = max_colour
        for i, step in enumerate(step_list[:-1]):
            next_step = step_list[i+1]
            pixels = np.where((land > step) & (land <= next_step))
            pixel_lists[colour_list[i]] = list(zip(pixels[0], pixels[1]))

        
    return pixel_lists

def get_pixel_lists(np_img: np.ndarray, landmask: np.ndarray, json_dir: str, force: bool=False, colours: int=8, save=True, uniform=False, use_max_pixel=False) -> dict:
    '''
    DEPRECATED
    Get the lists of pixels as a dictionary for each colour bucket in the img.
    Args:
        np_img (np.ndarray): The image to get the pixel lists for
        landmask (np.ndarray): The landmask to use to filter out land pixels
        json_dir (str): The directory to save the pixel lists to or read from to improve performance of repeated calls
        force (bool, optional): Whether to force the creation of the pixel lists even if they already exist. Defaults to False.
        colours (int, optional): The number of colours to use to bucket the pixels. Defaults to 8.
        save (bool, optional): Whether to save the updated pixel list to the json file. Defaults to True.
        uniform (bool, optional): Whether to use uniform bucket sizes. Defaults to False.
        use_max_pixel (bool, optional): Whether to use the maximum pixel value as the maximum colour value. Defaults to False.
    Returns:
        dict: A dictionary of pixel lists for each colour bucket in the image
    '''
    warnings.warn("get_pixel_lists is deprecated, use numpy_get_pixel_lists instead", DeprecationWarning)
    pixel_lists = {}
    if os.path.exists(json_dir) and not force:
        with open(json_dir, 'r') as entry:
            pixel_lists = json.loads(entry.read())
        if len(pixel_lists) == colours:
            return pixel_lists
        else:
            pixel_lists = {}
    h, w = np_img.shape
    pixels = []
    with tqdm(total=1, desc=f"Getting pixel lists for {w}x{h}...", disable=(MAX_LOGGING_DEPTH<call_depth())) as pbar:
        mask = (np_img > 0) & (landmask > 200)
        pixels = list(zip(np_img[mask], np.argwhere(mask)))
        pixel_lists = quantize_list(pixels, colours, uniform, use_max_pixel)
        pbar.update(1)
    if save:
        with open(json_dir, 'w') as out:
            out.write(json.dumps(pixel_lists))
    return pixel_lists

def landcover_paint(landcover: np.ndarray, planet_cfg: PlanetConfig) -> np.ndarray:
    '''
    Produce a sketch using the landcover image. 
    Args:
        landcover (np.ndarray): The landcover image to produce the sketch from
        planet_cfg (PlanetConfig): The planet config to use to produce the sketch
    Returns:
        np.ndarray: The sketch
    '''
    colours = planet_cfg.landcover_classes
    sketch = landcover // (255 // colours)
    sketch = dilate_paint(sketch, planet_cfg, discretized=True)
    return sketch * (255 // colours)

def get_buckets(dem: np.ndarray, planet_cfg: PlanetConfig) -> list[tuple[int, int]]:
    """
    Get the buckets for the given DEM.
    Args:
        dem (np.ndarray): The DEM to get the buckets for
        planet_cfg (PlanetConfig): The planet config to use to get the buckets
    Returns:
        list: The list of buckets
    """
    pixels = dem.flatten()
    pixels.sort()
    pixels = pixels[pixels > 0]
    num_buckets = planet_cfg.colours
    bucket_size = round(len(pixels) / num_buckets)
    buckets = [(0, 1)]
    for i in range(0, num_buckets):
        lower = pixels[0]
        upper = pixels[bucket_size]
        buckets.append((lower, upper))
        pixels = pixels[bucket_size:]
        if len(pixels) <= bucket_size:
            buckets.append((upper, 256))
            break
    return buckets


DEFAULT_BUCKETS = [(0, 1), (1.0091229e-05, 5.645488), (5.645488, 16.990366), (16.990366, 56.57612), (56.57612, 256)]


@profile
def dilate_paint(
    dem: np.ndarray,
    planet_cfg: PlanetConfig,
    discretized: bool = False,
    buckets: list | None = None
) -> np.ndarray:
    '''
    Produce a sketch using a combination of dilation and erosion operations.
    Args:
        dem (np.ndarray): The DEM to produce the sketch from
        planet_cfg (PlanetConfig): The planet config to use to produce the sketch
        discretized (bool, optional): Whether the DEM is already discretized. Defaults to False.
        buckets (list, optional): The list of buckets to use to produce the sketch. Sketch should not be discretized if using this option Defaults to None.
    Returns:
        np.ndarray: The sketch
    '''

    dem = dem.round().clip(0, 255).astype(np.uint8)
    if not discretized:

        if buckets is not None:
            sketch = np.zeros_like(dem)
            for i, bucket in enumerate(buckets):
                sketch[(dem >= bucket[0]) & (dem < bucket[1])] = i*(256//planet_cfg.colours)-1 if i > 0 else 0
        else:
            colours = planet_cfg.sketch_colours
            # sketch = paint_img(dem, replace(planet_cfg, brush_size=0))
            in_step = (planet_cfg.get_max_pixel(dem)+1) // colours
            out_step = 256 // colours
            
            if planet_cfg.gaussian_blur > 0:
                sketch = GaussianBlur(dem, (planet_cfg.blur_size, planet_cfg.blur_size), planet_cfg.gaussian_blur)
            else:
                sketch = dem.copy()
            sketch[dem == 0] = 0
            sketch = sketch.astype(np.float32) / in_step
            sketch = np.ceil(sketch) * out_step - 1
            sketch[sketch < 0] = 0
            sketch = sketch.astype(np.uint8)
    else:
        sketch = dem

    erode_filter = np.zeros((2*planet_cfg.erode_size, 2*planet_cfg.erode_size), dtype=bool)
    dilate_filter = np.zeros((2*planet_cfg.dilate_size, 2*planet_cfg.dilate_size), dtype=bool)
    
    erode_filter[planet_cfg.erode_size-1, planet_cfg.erode_size-1] = 1
    dilate_filter[planet_cfg.dilate_size-1, planet_cfg.dilate_size-1] = 1

    erode_filter = brush_mask(erode_filter, planet_cfg.erode_size).astype(np.uint8)
    dilate_filter = brush_mask(dilate_filter, planet_cfg.dilate_size).astype(np.uint8)
    
    if planet_cfg.dilate_first:
        sketch = dilate(sketch, dilate_filter, iterations=planet_cfg.dilate_iters)
    sketch = erode(sketch, erode_filter, iterations=planet_cfg.erode_iters)
    sketch = dilate(sketch, dilate_filter, iterations=planet_cfg.dilate_iters)
    if planet_cfg.preserve_edges:
        sketch[dem == 0] = 0
    return sketch


def paint_img(dem: np.ndarray, planet_cfg: PlanetConfig) -> np.ndarray:
    '''
    Paint the image with the given pixel lists and brush deltas.
    Paints from the lowest colour to the highest colour to ensure the highest altitudes are painted last.
    Args:
        pixel_lists (dict): The dictionary of pixel lists for each colour to paint
        dem (np.ndarray): The DEM to paint on
        brush_deltas (list): The list of brush deltas to use to paint
        planet_cfg (PlanetConfig): The planet config to use to paint
    Returns:
        np.ndarray: The sketch
    '''
    sketch = np.zeros_like(dem)
    
    use_colour_map = planet_cfg.use_colour_map
    pixel_lists = numpy_get_pixel_lists(dem, (dem > 0).astype(np.uint8)*255, planet_cfg)
    brush_deltas = get_brush_deltas(planet_cfg.brush_size)


    total_pixels = sum([len(pixel_lists[col]) for col in pixel_lists])
    colour_list = list(pixel_lists.keys())
    colour_list.sort(key=lambda x: int(x))
    h, w = dem.shape
    with tqdm(total=len(colour_list), desc=f"Painting sketch {w}x{h}...", disable=True) as pbar:
        for colour in colour_list:
            mask = np.zeros_like(dem, dtype=bool)
            pixels = pixel_lists[colour]
            if len(pixels) == 0:
                pbar.update(1)
                continue
            mask[tuple(np.array(pixels).T)] = True
            sketch[mask] = colour
            shifted_mask = np.zeros_like(mask, dtype=bool)
            for dx, dy in brush_deltas:
                roll = np.roll(mask, (dy, dx), axis=(0, 1))
                if dy > 0:
                    roll[:dy,:] = False
                elif dy < 0:
                    roll[dy:,:] = False
                if h <= 256 and w <= 256:
                    # We are generating a tile so don't wrap around
                    if dx > 0:
                        roll[:,:dx] = False
                    elif dx < 0:
                        roll[:,dx:] = False
                shifted_mask |= roll

            sketch[shifted_mask] = colour
            pbar.update(1)
    if planet_cfg.preserve_edges:
        sketch[dem == 0] = 0
    return sketch

@profile
def temperature_paint(temp: np.ndarray, planet_cfg: PlanetConfig) -> np.ndarray:
    """
    Create a temperature sketch from the grayscale temperature image.
    Args:
        temp (np.ndarray): The temperature image to create the sketch from
        planet_cfg (PlanetConfig): The planet config to use to create the sketch
    Returns:
        np.ndarray: The temperature sketch
    """
    colours = planet_cfg.temp_classes
    # sketch = GaussianBlur(temp, (planet_cfg.blur_size, planet_cfg.blur_size), planet_cfg.gaussian_blur)
    sketch = temp.copy()
    sketch[temp == 0] = 0
    in_step = 255 // colours
    out_step = 256 // colours
    sketch = sketch.astype(np.float32) / in_step
    sketch = np.ceil(sketch) * out_step - 1
    sketch[sketch < 0] = 0
    sketch = sketch.astype(np.uint8)
    sketch = dilate_paint(sketch, planet_cfg, discretized=True)
    return sketch

class ConnectedComponent:
    size: int
    pixels: list
    border_pixels: list
    bridge: list
    bounds: list
    center: tuple
    def add(self, pixel: tuple, img: np.ndarray = np.ndarray(0)) -> None:
        '''
        Add a pixel to the connected component.
        Args:
            pixel (tuple): The pixel to add
            img (np.ndarray): The image to get the border pixels from
        Returns:
            None
        '''
        self.pixels.append(pixel)
        self.size += 1
        h, w = img.shape
        if img.shape != (0,):
            y, x = pixel
            for dx, dy in get_brush_deltas(1, (0, 0)):
                if y+dy < 0 or y+dy >= h:
                    continue
                if img[y+dy, (x+dx)%w] == 0:
                    self.border_pixels.append((y, x))
                    break
        # Create square bounds
        if pixel[0] < self.bounds[0]:
            self.bounds[0] = pixel[0]
        if pixel[0] > self.bounds[1]:
            self.bounds[1] = pixel[0]
        if pixel[1] < self.bounds[2]:
            self.bounds[2] = pixel[1]
        if pixel[1] > self.bounds[3]:
            self.bounds[3] = pixel[1]
        width = self.bounds[3] - self.bounds[2]
        height = self.bounds[1] - self.bounds[0]
        if width > height:
            self.bounds[0] -= (width - height) // 2
            self.bounds[1] += (width - height) // 2
        elif height > width:
            self.bounds[2] -= (height - width) // 2
            self.bounds[3] += (height - width) // 2
        # Update center
        self.center = (self.bounds[0] + (self.bounds[1] - self.bounds[0]) // 2, self.bounds[2] + (self.bounds[3] - self.bounds[2]) // 2)
        
    
    def merge(self, other: 'ConnectedComponent', bridge: list) -> None:
        '''
        Merge another connected component into this one.
        Args:
            other (ConnectedComponent): The other connected component to merge
        Returns:
            None
        '''
        self.pixels += other.pixels + bridge
        self.size += other.size + len(bridge)
        self.border_pixels += other.border_pixels
        self.bridge = bridge

    def __init__(self) -> None:
        '''
        Initialise a new connected component.
        '''
        self.size = 0
        self.pixels = []
        self.border_pixels = []
        self.bridge = []
        self.bounds = [float('inf'), 0, float('inf'), 0]

def strahler_paint_river(dem: np.ndarray, ocean: np.ndarray, water: np.ndarray, planet_cfg: PlanetConfig) -> np.ndarray:
    '''
    Get painted river map based on the strahler order of the rivers.
    This strahler order is derived from the DEM so it may not perfectly match the river mask.
    Args:
        dem (np.ndarray): The DEM to use
        oceanmask (np.ndarray): The int ocean mask to use
        watermask (np.ndarray): The int water mask to use
        planet_cfg (PlanetConfig): The planet config to use
    Returns:
        np.ndarray: The painted river mask
    '''
    rivers = get_strahler_orders(dem, ocean, water, planet_cfg)
    return process_strahler_rivers(rivers, planet_cfg)
    
# @timing
def get_strahler_orders(
    dem: np.ndarray,
    ocean: np.ndarray,
    water: np.ndarray,
    planet_cfg: PlanetConfig,
    threshold: int = None,
    gaussian_blur: int = None
) -> np.ndarray:
    '''
    Get the strahler orders of the rivers.
    '''
    assert dem.dtype == np.uint8, "DEM must be of type np.uint8"
    assert ocean.dtype == np.uint8, "Ocean mask must be of type np.uint8"
    assert water.dtype == np.uint8, "Water mask must be of type np.uint8"

    assert dem.shape == ocean.shape, "DEM and ocean mask must have the same shape"
    assert dem.shape == water.shape, "DEM and water mask must have the same shape"

    oceanmask = ocean == 255
    threshold = threshold or planet_cfg.threshold
    gaussian_blur = gaussian_blur or planet_cfg.gaussian_blur

    if dem.max() == 0:
        return np.zeros_like(dem)
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)

    vf = ViewFinder(
        affine=Affine(1, 0, 0, 0, 1, 0),
        shape=dem.shape,
        crs=pyproj.Proj('WGS84', preserve_units=False),
        mask=oceanmask
    )
    blur_size = planet_cfg.blur_size
    assert blur_size > 0 and blur_size % 2 == 1, f"Gaussian blur size must be odd and greater than 0 ({blur_size})"
    dem_raster = Raster(
        GaussianBlur(dem, (blur_size, blur_size), gaussian_blur) if gaussian_blur > 0 else dem,
        viewfinder=vf
    )
    if dem_raster.max() == 0:
        return np.zeros_like(dem)
    grid = Grid(vf)
    # https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
    ymin, ymax = np.where(np.any(dem_raster, axis=1))[0][[0, -1]]
    xmin, xmax = np.where(np.any(dem_raster, axis=0))[0][[0, -1]]
    grid.from_raster(dem_raster)
    grid.clip_to(dem_raster)

    # Fill pits in DEM
    pit_filled_dem = grid.fill_pits(dem_raster)

    # Fill depressions in DEM
    flooded_dem = grid.fill_depressions(pit_filled_dem)
        
    # Resolve flats in DEM
    inflated_dem = grid.resolve_flats(flooded_dem)

    # Get flow direction
    fdir = grid.flowdir(inflated_dem, dirmap=dirmap)

    # Get flow accumulation
    acc = grid.accumulation(fdir, dirmap=dirmap)

    # Get strahler order
    order = grid.stream_order(fdir, acc > threshold)

    rivers = np.zeros_like(dem, dtype=np.uint8)
    rivers[ymin: ymax+1, xmin: xmax+1] = order.astype(np.uint8)
    rivers[oceanmask] = 0
    return rivers


deltas = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


def get_neighbour_stack(arr: np.ndarray, v_wrap: bool = False, h_wrap: bool = True) -> np.ndarray:
    xpad = 0 if h_wrap else 1
    ypad = 0 if v_wrap else 1
    arr = np.pad(arr, np.array(((ypad, ypad), (xpad, xpad))), mode='constant')
    neighbour_arrays = []
    for dy, dx in deltas:
        if dy == 0 and dx == 0:
            continue
        rolled_arr = np.roll(arr, (dy, dx), (0, 1))
        neighbour_arrays.append(rolled_arr)
    neighbour_stack = np.dstack(neighbour_arrays)
    if not v_wrap:
        neighbour_stack = neighbour_stack[1:-1, :]
    if not h_wrap:
        neighbour_stack = neighbour_stack[:, 1:-1]
    return neighbour_stack


def get_num_neighbours(arr: np.ndarray, v_wrap: bool = False, h_wrap: bool = True) -> np.ndarray:
    stack = get_neighbour_stack(arr, v_wrap, h_wrap)
    return np.count_nonzero(stack > 0, axis=2).astype(np.uint8)


def get_endpoints(arr: np.ndarray, v_wrap: bool = False, h_wrap: bool = True) -> np.ndarray:
    stack = get_neighbour_stack(arr, v_wrap, h_wrap)
    arr_stack = np.dstack([arr] * 8)
    neighbour_smaller = stack < arr_stack
    return neighbour_smaller.all(axis=-1)


def get_start_points(arr: np.ndarray, v_wrap: bool = False, h_wrap: bool = True) -> np.ndarray:
    stack = get_neighbour_stack(arr, v_wrap, h_wrap)
    arr_stack = np.dstack([arr] * 8)
    neighbour_smaller = (stack >= arr_stack) | (stack == 0)
    majority_zero = np.count_nonzero(stack == 0, axis=-1) >= 5

    return neighbour_smaller.all(axis=-1) & majority_zero & (arr_stack > 0).all(axis=-1)


def hydro_erode_dem(
    dem: Raster,
    accumulation: Raster,
    fdir: Raster,
    acc_threshold_percentile: float = 0.95,
    max_erosion_pcnt: float = 0.01,
) -> Raster:

    acc_values = np.sort(accumulation.flatten())
    acc_values = acc_values[acc_values > 0]
    if acc_values.size == 0:
        return dem
    acc_threshold = acc_values[int(acc_values.size * acc_threshold_percentile)]

    # 1. Get the endpoints of all the rivers in the accumulation raster above some threshold
    accumulation[accumulation < acc_threshold] = 0
    acc_endpoints = get_endpoints(accumulation).astype(np.float32)
    acc_endpoints *= accumulation
    end_ys, end_xs = np.where(acc_endpoints > 0)

    # 2. Find the lowest of the three points that are in front of the endpoints using the fdir
    dem_neighbours = get_neighbour_stack(dem)
    fdir[fdir < 1] = 1
    int_fdir = np.log2(fdir).astype(np.uint8)
    c_clockwise_dir = (int_fdir - 1) % 8
    clockwise_dir = (int_fdir + 1) % 8

    c_clockwise_neighbour = np.take_along_axis(dem_neighbours, c_clockwise_dir[..., None], axis=-1)[..., 0]
    forward_neighbour = np.take_along_axis(dem_neighbours, int_fdir[..., None], axis=-1)[..., 0]
    clockwise_neighbour = np.take_along_axis(dem_neighbours, clockwise_dir[..., None], axis=-1)[..., 0]

    # c_clockwise_neighbour = dem_neighbours[:, :, c_clockwise_dir]
    # forward_neighbour = dem_neighbours[:, :, int_fdir]
    # clockwise_neighbour = dem_neighbours[:, :, clockwise_dir]

    directions = np.dstack([c_clockwise_dir, int_fdir, clockwise_dir])
    valid_neighbours = np.dstack([c_clockwise_neighbour, forward_neighbour, clockwise_neighbour])
    lowest_index = np.argmin(valid_neighbours, axis=2)
    # new_dir = directions[:, :, lowest_index]
    new_dir = np.take_along_axis(directions, lowest_index[..., None], axis=-1)[..., 0]

    delta_ys = np.array([d[0] for d in deltas])
    new_dir = new_dir[end_ys, end_xs]
    delta_ys = np.take_along_axis(delta_ys, new_dir, axis=-1)
    delta_xs = np.array([d[1] for d in deltas])
    delta_xs = np.take_along_axis(delta_xs, new_dir, axis=-1)

    h, w = dem.shape[:2]
    next_ys = (end_ys + delta_ys).clip(0, h - 1)
    next_xs = (end_xs + delta_xs) % w

    # 3. If it is within some height threshold of the current point,
    # lower the value by some percentage between 0 and max

    current_heights = dem[end_ys, end_xs]
    next_heights = dem[next_ys, next_xs]
    acc_vals = accumulation[end_ys, end_xs]
    scaler = 1.0 - acc_vals / acc_vals.max() * max_erosion_pcnt
    scaler[current_heights + 50 < next_heights] = 0
    scaler[next_heights < current_heights] = 0
    new_dem = dem.copy()
    new_dem[next_ys, next_xs] = current_heights * scaler
    return new_dem


@timing
def accumulation(
    dem: np.ndarray,
    weights: np.ndarray,
    efficiency: np.ndarray | None,
    planet_cfg: PlanetConfig,
    smoothing: int = 3,
    return_fdir: bool = False,
) -> np.ndarray | tuple[np.ndarray, Raster]:
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)

    vf = ViewFinder(
        affine=Affine(1, 0, 0, 0, 1, 0),
        shape=dem.shape,
        crs=pyproj.Proj('WGS84', preserve_units=False),
        nodata=0
    )

    if smoothing > 0:
        dem = cv2.GaussianBlur(dem, (2 * smoothing + 1, 2 * smoothing + 1), None)

    dem_raster = Raster(
        dem,
        viewfinder=vf
    )
    weights_raster = Raster(
        weights,
        viewfinder=vf
    )
    efficiency_raster = Raster(
        efficiency,
        viewfinder=vf
    ) if efficiency is not None else None
    if dem_raster.max() == 0:
        return np.zeros_like(dem)

    grid = Grid(vf)
    # https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
    ymin, ymax = np.where(np.any(dem_raster, axis=1))[0][[0, -1]]
    xmin, xmax = np.where(np.any(dem_raster, axis=0))[0][[0, -1]]
    grid.from_raster(dem_raster)
    grid.clip_to(dem_raster)

    # Fill pits in DEM
    pit_filled_dem = grid.fill_pits(dem_raster)

    # Fill depressions in DEM
    flooded_dem = grid.fill_depressions(pit_filled_dem)

    # Resolve flats in DEM
    inflated_dem = grid.resolve_flats(flooded_dem)

    # Get flow direction
    fdir = grid.flowdir(inflated_dem, dirmap=dirmap)

    # Get flow accumulation
    acc = grid.accumulation(fdir, weights_raster, dirmap=dirmap, efficiency=efficiency_raster)

    poly_acc = grid.polygonize(acc)

    rivers = np.zeros_like(weights)
    rivers[ymin: ymax+1, xmin: xmax+1] = np.sqrt(np.maximum(acc, 0))

    if return_fdir:
        return rivers, fdir

    return rivers


def accumulation_with_erosion(
    dem: np.ndarray,
    weights: np.ndarray,
    efficiency: np.ndarray,
    planet_cfg: PlanetConfig,
    smoothing: int = 0,
    iterations: int = 5
) -> np.ndarray:
    for i in tqdm(range(iterations)):
        acc, fdir = accumulation(dem, weights, efficiency, planet_cfg, smoothing, return_fdir=True)
        # h, w = acc.shape[:2]
        # img.fromarray(
        #     np_rgb(acc[h // 2: 5 * h // 8, w // 4: 3 * w // 8] / acc.max() * 255, cmap="viridis").astype(np.uint8)
        # ).save(os.path.join(planet_cfg.test_dir, f"{i}_river_upa_erode.png"))

        # img.fromarray(
        #     np_rgb(
        #         dem.clip(0, 5.0)[h // 2: 5 * h // 8, w // 4: 3 * w // 8] / 5.0 * 255,
        #         cmap="viridis"
        #     ).astype(np.uint8)
        # ).save(os.path.join(planet_cfg.test_dir, f"{i}_dem_erode.png"))
        if i == iterations - 1:
            break
        new_dem = hydro_erode_dem(dem, acc, fdir)
        delta = (dem - new_dem).sum()
        print(f"DEM lost a total of {delta:.3f} in iteration {i}")
        dem = new_dem
    return acc


def process_strahler_rivers(rivers: np.ndarray, planet_cfg: PlanetConfig) -> np.ndarray:
    '''
    Process the strahler order of the rivers to get a river mask.
    Args:
        rivers (np.ndarray): The strahler order of the rivers
        planet_cfg (PlanetConfig): The planet config to use
    Returns:
        np.ndarray: The river mask
    '''    
    # https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
    if rivers.max() == 0:
        return np.zeros_like(rivers)
    ymin, ymax = np.where(np.any(rivers, axis=1))[0][[0, -1]]
    xmin, xmax = np.where(np.any(rivers, axis=0))[0][[0, -1]]
    top_orders = planet_cfg.top_n_orders
    brush_size = planet_cfg.river_size
    variable_rivers = planet_cfg.variable_rivers
    orders, counts = np.unique(rivers, return_counts=True)
    i = len(orders) - 1
    while i >= 0:
        if counts[i] < planet_cfg.min_size:
            rivers[rivers == orders[i]] = 0
        i -= 1
    max_order = np.max(rivers)
    to_return = np.zeros_like(rivers)
    if variable_rivers:
        for delta_order in range(0, min(top_orders, max_order)):
            bs = max(0, brush_size - delta_order)
            order = max_order - delta_order
            to_return[brush_mask(rivers == order, bs, wrap_x=False)] = 255
    else:
        to_return[brush_mask(rivers > max(max_order-top_orders, 0), brush_size, wrap_x=False)] = 255

    labeled, ncomponents = label(to_return[ymin: ymax+1, xmin: xmax+1]) 
    # Remove small components
    for i in range(1, ncomponents+1):
        if np.sum(labeled == i) < planet_cfg.min_size:
            to_return[ymin: ymax+1, xmin: xmax+1][labeled == i] = 0
    return to_return

    

def skeletonize_paint_river(river_mask: np.ndarray, brush_size: int, min_size: int) -> np.ndarray:
    '''
    Get rivers based on their skeletonized shape and draw a thick line on them.
    Args:
        river_mask (np.ndarray): The river mask to paint
        brush_size (int): The size of the brush to use to paint
        min_size (int): The minimum size of the rivers to paint
    
    Returns:
        np.ndarray: The painted river mask
    '''
    warnings.warn("This function is deprecated. Use strahler_paint_river instead.", DeprecationWarning)
    # river_mask[river_mask < threshold] = 0
    # river_mask[river_mask >= threshold] = 255

    # img.fromarray(river_mask).show()


    # rivers = skeletonize(river_mask)
    
    
    # rivers = skeletonize(river_mask)
    
    # img.fromarray(rivers).show()
    assert river_mask.dtype == bool

    rivers = np.zeros_like(river_mask, dtype=np.uint8)
    rivers[skeletonize(river_mask.astype(np.uint8))] = 255

    labeled, ncomponents = label(~river_mask)

    # test = np.zeros_like(labeled, dtype=np.uint8)
    # test[labeled > 1] = 255
    # test[river_mask] = 128
    # test[labeled == 1] = 64
    # # test[labeled == 0] = 64

    # img.fromarray(test).show()
    # Fill in small components which represent bodies of water and not rivers
    counts = np.bincount(labeled.ravel())
    outside = np.argmax(counts)
    rivers[labeled != outside] = 255
    # Rivers are at index 0 since they have value 0 in the inverted river mask
    rivers[labeled == 0] = 255

    labeled, ncomponents = label(rivers, structure=np.ones((3, 3), dtype=np.uint8))
    counts = np.bincount(labeled.ravel())
    big_components = np.where(counts >= min_size)[0]
    small_components = np.where(counts < min_size)[0]
    with tqdm(total=min(len(big_components) - 1, len(small_components)), desc='Removing small components', disable=(MAX_LOGGING_DEPTH<call_depth())) as pbar:
        if len(big_components) < len(small_components):
            rivers = np.zeros_like(river_mask, dtype=np.uint8)
            for comp in list(big_components[1:]):
                rivers[labeled == comp] = 255
                pbar.update(1)
        else:
            for comp in list(small_components):
                rivers[labeled == comp] = 0
                pbar.update(1)

    rivers[brush_mask(rivers == 255, brush_size)] = 255

    return rivers
    

def paint_river(river_mask: np.ndarray, brush_size: int) -> np.ndarray:
    '''
    Paints rivers from the given river mask using the given brush size.
    The river mask should be a binary mask of the rivers to paint not including oceans.
    This should be used as a seperate channel to the sketch.
    Args:
        river_mask (np.ndarray): The river mask to paint
        brush_size (int): The size of the brush to use to paint the rivers
    Returns:
        np.ndarray: The painted river mask
    '''
    brush_deltas = get_brush_deltas(brush_size)
    output = np.zeros(shape=river_mask.shape, dtype=np.uint8)
    h, w = river_mask.shape
    mask = np.zeros_like(river_mask, dtype=bool)
    mask[river_mask > 0] = True
    with tqdm(total=1, desc=f"Painting river {w}x{h}...", disable=(MAX_LOGGING_DEPTH<call_depth())) as pbar:
        shifted_mask = np.zeros_like(mask, dtype=bool)
        for dx, dy in brush_deltas:
            roll = np.roll(mask, (dy, dx), axis=(0, 1))
            if dy > 0:
                roll[:dy,:] = False
            elif dy < 0:
                roll[dy:,:] = False
            shifted_mask |= roll
        output[shifted_mask] = 255
        pbar.update(1)
    return output

def get_line(start: tuple, end: tuple) -> list:
    '''
    Get a rasterized line between two points including both endpoints.
    Args:
        start (tuple): The start point
        end (tuple): The end point
    Returns:
        list: The list of points on the line
    '''
    y0, x0 = start
    y1, x1 = end
    if x1 == x0:
        return [(y, x0) for y in range(min(y0, y1), max(y0, y1)+1)]
    if y1 == y0:
        return [(y0, x) for x in range(min(x0, x1), max(x0, x1)+1)]
    
    y = y0
    x = x0
    points = []
     
    dx = x1 - x0
    dy = y1 - y0
    dist = sqrt(dx**2 + dy**2)
    m = dy/dx
    dx = dx/dist
    dy = dy/dist
    while round(x) != x1 or round(y) != y1:
        x += dx
        y += dy
        points.append((round(y), round(x)))
    
    return points

def is_border_pixel(pixel: tuple, img: np.ndarray) -> bool:
    '''
    Checks to see if the given pixel is a border pixel.
    Args:
        pixel (tuple): The pixel to check
        img (np.ndarray): The image to check
    Returns:
        bool: True if the pixel is a border pixel, False otherwise
    '''

    y, x = pixel
    h, w = img.shape
    for dx, dy in get_brush_deltas(1, (0, 0)):
        if y+dy < 0 or y+dy >= h:
            continue
        if img[y+dy, (x+dx)%w] == 0:
            return True
    return False

def connected(p0: tuple, p1: tuple, border_pixels: list, is_connected: np.ndarray, w: int) -> bool:
    '''
    Checks to see if two pixels are connected.
    Args:
        p0 (tuple): The first pixel
        p1 (tuple): The second pixel
        border_pixels (list): The list of border pixels
        is_connected (np.ndarray): The current array of connected pixels
        w (int): The width of the image
    Returns:
        bool: True if the pixels are connected, False otherwise
    '''
    y0, x0 = p0
    y1, x1 = p1
    visited = [0]*len(border_pixels)
    queue = []
    p0_found = False
    p1_found = False
    deltas = []
    for y in range(-1, 2):
        for x in range(-1, 2):
            if x == 0 and y == 0:
                continue
            deltas.append((y, x))
    for i, (y, x) in enumerate(border_pixels):
        if not visited[i]:
            queue.append((y, x))
            if p0_found or p1_found:
                return False
            p0_found = False
            p1_found = False
        while len(queue) > 0:
            y, x = queue.pop(0)
            if y == y0 and x == x0:
                p0_found = True
            if y == y1 and x == x1:
                p1_found = True
            if p0_found and p1_found:
                return True
            for dx, dy in deltas:
                i = -1
                try:
                    i = border_pixels.index((y+dy, x+dx))
                except:
                    continue
                if not visited[i]:
                    queue.append(border_pixels[i])
                visited[i] = 1
    return False



def border_mask(np_img: np.ndarray) -> np.ndarray:
    '''
    Get the border mask of the given image.
    Args:
        img (np.ndarray): The image to get the border mask from
    Returns:
        np.ndarray: The border mask
    '''
    mask = np.zeros_like(np_img, dtype=bool)
    mask[np_img > 0] = True
    shift_mask = brush_mask(mask, 1, (0, 0))
    mask = np.logical_xor(mask, shift_mask)
    mask = brush_mask(mask, 1, (0, 0))
    mask[np_img == 0] = False

    img_example = np.zeros_like(np_img)
    img_example[mask] = 255
    # img.fromarray(img_example).save('test.png')
    return mask


def get_connected_components(np_img: np.ndarray, radius: int=1) -> list:
    '''
    Get connected components within a radius
    Args:
        np_img (np.ndarray): The image to get the connected components from
        radius (int, optional): The radius to search for connected components. Defaults to 1.
    Returns:
        list: The list of connected components
    '''
    components = []
    h, w = np_img.shape
    radius_deltas = get_brush_deltas(radius, (0, 0))
    original = np_img.copy()
    with tqdm(total=h*w, desc=f"Getting components for {w}x{h}...", disable=(MAX_LOGGING_DEPTH<call_depth())) as pbar:
        for y in range(h):
            for x in range(w):
                neighbours = []
                if np_img[y, x] > 0:
                    components.append(ConnectedComponent())
                    neighbours.append((y, x))
                    np_img[y, x] = 0
                    while len(neighbours) > 0:
                        ny, nx = neighbours.pop()
                        components[-1].add((ny, nx), original)
                        for dx, dy in radius_deltas:
                            if ny+dy < 0 or ny+dy >= h:
                                continue
                            
                            if np_img[ny+dy, (nx+dx)%w] > 0:
                                neighbours.append((ny+dy, nx+dx))
                                np_img[ny+dy, (nx+dx)%w] = 0
                pbar.update(1)
    return components

# @profile
def numpy_remove_small_components(np_img: np.ndarray, min_size: int=25, radius: int=3, river_size: int=1) -> np.ndarray:
    '''
    Remove small components from an array.
    Only supports arrays with values of 0 or 255.
    Args:
        np_img (np.ndarray): The array to remove small components from
        min_size (int, optional): The minimum size of the component to keep. Defaults to 100.
        radius (int, optional): The max radius to connect components. Defaults to 1.
        river_size (int, optional): The size of the river brush. Defaults to 2.
    Returns:
        np.ndarray: The array with small components removed
    '''
    output = np.zeros(shape=np_img.shape, dtype=np.uint8)
    connect = radius > 0
    h, w = np_img.shape
    with tqdm(total=1, desc=f"Getting components for {w}x{h}...", disable=(MAX_LOGGING_DEPTH<call_depth())) as pbar:
        labeled, ncomponents = label(np_img)
        borders = border_mask(labeled)
        pbar.update(1)
        labeled_borders = labeled.copy()
        labeled_borders[~borders] = 0
        bridge_mask = np.zeros_like(output, dtype=bool)
        radius_sq = radius**2
        if connect:
            component_lengths = [np.sum(labeled == i) for i in range(1, ncomponents + 1)]    

    if connect:
        with tqdm(total=ncomponents-1, desc="Connecting components...", disable=(MAX_LOGGING_DEPTH<call_depth())) as pbar:
            # 1, 2 and 3 take up >75% of the time
            component_borders = [np.argwhere(labeled_borders == i) for i in range(1, ncomponents + 1)] #1
            
            for i in range(1, ncomponents):
                c0 = component_borders[i-1]
                for k in range(i+1, ncomponents+1):
                    c1 = component_borders[k-1]
                    # Connect small components to large ones
                    if component_lengths[i-1] + component_lengths[k-1] < min_size: 
                        continue
                    
                    dists = cdist(c0, c1, metric='sqeuclidean') #2
                    ip = np.argwhere(dists <= radius_sq) #3
                    if len(ip) == 0:
                        continue
                    
                    for (y0, x0), (y1, x1) in list(zip(c0[ip[:, 0]], c1[ip[:, 1]])):
                        points = get_line((y0, x0), (y1, x1))
                        bridge_mask[tuple(np.array(points).T)] = True
                    labeled[labeled == k] = i
                    component_lengths[i-1] += component_lengths[k-1]

                pbar.update(1) 

    for i in range(1, ncomponents + 1):
        if np.sum(labeled == i) < min_size:
            labeled[labeled == i] = 0
    bridge_mask = brush_mask(bridge_mask, river_size, (0.5, 0.5))
    output[bridge_mask] = 255   
    output[labeled > 0] = 255 
    return output




def remove_small_components(np_img: np.ndarray, min_size: int=25, radius: int=3, river_size: int=1) -> np.ndarray:
    '''
    Remove small components from an array.
    Only supports arrays with values of 0 or 255.
    Args:
        np_img (np.ndarray): The array to remove small components from
        min_size (int, optional): The minimum size of the component to keep. Defaults to 100.
        radius (int, optional): The max radius to connect components. Defaults to 1.
        river_size (int, optional): The size of the river brush. Defaults to 2.
    Returns:
        np.ndarray: The array with small components removed
    '''
    output = np.zeros(shape=np_img.shape, dtype=np.uint8)
    connect = radius > 0
    components = get_connected_components(np_img.copy(), radius)
    original = np_img.copy()
    river_deltas = get_brush_deltas(river_size, (0.5, 0.5))
    h, w = np_img.shape
    if connect:
        component_index = np.full(fill_value=-1, shape=np_img.shape, dtype=np.int32)
        small_components = get_connected_components(np_img.copy(), 1)
        with tqdm(total=len(small_components), desc="Getting component indices...", disable=(MAX_LOGGING_DEPTH<call_depth())) as pbar:
            for i, component in enumerate(small_components):
                for y, x in component.border_pixels:
                    component_index[y, x%w] = i
                pbar.update(1)
        with tqdm(total=len(components), desc="Connecting components...", disable=(MAX_LOGGING_DEPTH<call_depth())) as pbar:
            for component in components:
                for i, (y0, x0) in enumerate(component.border_pixels[:-1]):
                    for y1, x1 in component.border_pixels[i+1:]:
                        if (x1-x0)**2 + (y1-y0)**2 > (radius)**2:
                            continue
                        assert component_index[y0, x0%w] != -1
                        assert component_index[y1, x1%w] != -1
                        if component_index[y1, x1%w] != component_index[y0, x0%w]:
                            points = get_line((y0, x0), (y1, x1))
                            if len(points) > 2:
                                for y, x in points:
                                    for dx, dy in river_deltas:
                                        if y+dy < 0 or y+dy >= h:
                                            continue
                                        if original[y+dy, (x+dx)%w] == 0:
                                            component.bridge.append((y+dy, x+dx))
                                            component.size += 1
                pbar.update(1)

    for component in components:
        if component.size >= min_size:
            for y, x in component.pixels:
                output[y, x%w] = 255
            for y, x in component.bridge:
                output[y, x%w] = 255
    return output

def get_river_mask(oceanmask: np.ndarray, watermask: np.ndarray, threshold: int, down_scale: int) -> np.ndarray: 
    '''
    Get the river mask using the oceanmask and watermask.
    Args:
        oceanmask (np.ndarray): The oceanmask to use
        watermask (np.ndarray): The watermask to use
        threshold (int): The threshold to use on the masks
        down_scale (int): The downscale to use
    
    Returns:
        np.ndarray: The river mask
    '''
    assert oceanmask.shape == watermask.shape
    assert oceanmask.dtype == np.uint8
    assert watermask.dtype == np.uint8
    rivers = np.zeros_like(oceanmask, dtype=np.uint8)
    rivers[watermask <= 255-threshold] = 255
    rivers[oceanmask < 255] = 0
    h, w = rivers.shape
    h //= 2**down_scale
    w //= 2**down_scale
    h = max(256, int(h))
    w = max(512, int(w))
    if down_scale > 0:
        rivers = np.array(img.fromarray(rivers).resize((w, h), resample=img.Resampling.LANCZOS))
    # 20 here was found experimentally
    river_mask = np.zeros_like(rivers, dtype=bool)
    river_mask[rivers > 20] = True

    return river_mask

    


def test_river_params(sketch: np.ndarray, oceanmask: np.ndarray, watermask: np.ndarray, rivermask_offset: int) -> Image:
    '''
    Test different threshold and brush sizes for painting rivers and output a grid of combinations.
    The oceanmask and watermask must be the same size, but can be a different size to sketch.
    Args:
        sketch (np.ndarray): The sketch to add rivers to
        oceanmask (np.ndarray): The oceanmask to use to filter out oceans
        watermask (np.ndarray): The watermask to use to filter in water
        rivermask_offset (int): The size offset used for supersampling the rivermask
    Returns:
        Image: The grid of combinations
    '''
    thresholds = range(0, 255, 50)
    nth = len(thresholds)
    brush_sizes = range(0, 1, 1)
    nbs = len(brush_sizes)
    min_sizes = range(0, 25, 25)
    nms = len(min_sizes)
    radii = range(0, 1, 1)
    nr = len(radii)
    h, w = sketch.shape[:2]
    center = (h//2, w*11//10//2)
    xMin = center[1] - 128
    xMax = center[1] + 128
    yMin = center[0] - 128
    yMax = center[0] + 128
    imgs = []
    texts = []
    with tqdm(total=nth*nbs*nms*nr, desc="Testing river params...", disable=(MAX_LOGGING_DEPTH<call_depth())) as pbar:
        for r in radii:
            for th in thresholds:
                wm = watermask.copy()
                om = oceanmask.copy()
                rivers = np.zeros_like(sketch, dtype=np.uint8)
                rivers[get_river_mask(om, wm, th, rivermask_offset)] = 255
                rivers = rivers[yMin:yMax, xMin:xMax]
                for bs in brush_sizes:
                    for ms in min_sizes:
                        out = skeletonize_paint_river(rivers, th, ms)
                        # out = paint_river(rivers, bs)
                        # out = numpy_remove_small_components(out, ms, r, bs)
                        sk = sketch.copy()[yMin:yMax, xMin:xMax]
                        sk[:,:,2] = out
                        imgs.append(img.fromarray(sk))
                        texts.append(f"M{ms}|T{th}|B{bs}|R{r}")
                        pbar.update(1)
    return image_grid(imgs, nr*nth, nbs*nms, texts)

def river_mask_stats(rivers: np.ndarray, river_mask: np.ndarray) -> dict:
    '''
    Get stats to do with the usability and efficacy of the derived river mask.
    Args:
        rivers (np.ndarray): The derived rivers to use
        river_mask (np.ndarray): The original river mask to use
    
    Returns:
        dict: The stats (river coverage %, river wastage %, average component size, total components, total original rivers)
        River coverage: The percentage of the river mask that is covered by the derived rivers
        River wastage: The percentage of the derived rivers that is not covered by the river mask
        Average component size: The average size of the components in the derived rivers
        Total components: The total number of components in the derived rivers
        Total original rivers: The total number of rivers in the original river mask
    '''
    river_mask_pixels = np.sum(river_mask)
    river_pixels = np.sum(rivers == 255)

    river_coverage = np.sum(np.logical_and(rivers == 255, river_mask)) / river_mask_pixels if river_mask_pixels > 0 else 0
    river_wastage = np.sum(np.logical_and(rivers == 255, ~river_mask)) / river_pixels if river_pixels > 0 else 0

    labeled_rivers, num_components = label(rivers, structure=np.ones((3, 3)))
    labeled_river_mask, num_rivers = label(river_mask, structure=np.ones((3, 3)))
    avg_size = river_pixels / num_components if num_components > 0 else 0

    ret = {
        "river_coverage": river_coverage,
        "river_wastage": river_wastage,
        "avg_size": avg_size,
        "num_components": num_components,
        "num_rivers": num_rivers
    }
    return ret



def test_river_mask_params(sketch: np.ndarray, data_dir: str=os.path.join(os.getcwd(), 'data')) -> Image:
    '''
    Test different thresholds for painting rivers and output a grid of combinations.
    Args:
        sketch (np.ndarray): The sketch to add rivers to
        data_dir (str): The directory to read the masks from
    Returns:
        Image: The grid of combinations
    
    '''
    h, w = sketch.shape[:2]
    thresholds = range(50, 301, 50)
    nth = len(thresholds)

    offset = int(log2(16384//w))
    
    brush_sizes = range(2, 3, 1)
    nbs = len(brush_sizes)

    min_sizes = range(0, 70, 10)
    nms = len(min_sizes)
    
    center = (h//2, w*11//10//2)
    box_w = 512
    box_h = 256
    xMin = center[1] - box_w//2
    xMax = center[1] + box_w//2
    yMin = center[0] - box_h//2
    yMax = center[0] + box_h//2

    if xMax > w:
        xMax = w
        xMin = w - box_w
    imgs = []
    texts = []
    
    with tqdm(total=nms*nth*nbs, desc="Testing river params...") as pbar:
        W, H = min(w*2**offset, 16384), min(h*2**offset, 8192)
        watermask = np.array(img.open(os.path.join(data_dir, f'world.watermask.{W}x{H}.png')))
        oceanmask = np.array(img.open(os.path.join(data_dir, f'world.oceanmask.{W}x{H}.png')))
        water = np.array(img.open(os.path.join(data_dir, f'world.watermask.{w}x{h}.png')))
        ocean = np.array(img.open(os.path.join(data_dir, f'world.oceanmask.{w}x{h}.png')))
        dem = np.array(img.open(os.path.join(data_dir, f'World_DEM_{w}x{h}.png')))
        for bs in brush_sizes:
            for mi, ms in enumerate(min_sizes):
                for th in thresholds:
                    # sk = sketch.copy()[yMin:yMax, xMin:xMax]
                    # rivers = np.zeros(shape=(h, w), dtype=np.uint8)
                    river_mask = get_river_mask(oceanmask, watermask, th, offset)
                    # rivers[brush_mask(river_mask, bs)] = 255
                    # rivers = rivers[yMin:yMax, xMin:xMax]
                    # sk[:,:,2] = rivers
                    # stats = river_mask_stats(rivers, river_mask[yMin:yMax, xMin:xMax])
                    # imgs.append(img.fromarray(sk))
                    # txt_arr = [
                    #     f"T:{th}|B:{bs}|M:{ms}|R\n",
                    #     f"N:{int(stats['num_components'])}|R:{int(stats['num_rivers'])}|",
                    #     f"A:{int(stats['avg_size'])}\nC:{int(stats['river_coverage']*100)}|",
                    #     f"W:{int(stats['river_wastage']*100)}"
                    # ]
                    # txt = "".join(txt_arr)
                    # texts.append(txt)
                    

                    # sk = sketch.copy()[yMin:yMax, xMin:xMax]
                    # rivers = skeletonize_paint_river(river_mask, bs, ms)[yMin:yMax, xMin:xMax]
                    # sk[:,:,2] = rivers
                    # stats = river_mask_stats(rivers, river_mask[yMin:yMax, xMin:xMax])
                    # imgs.append(img.fromarray(sk))
                    # txt_arr = [
                    #     f"T:{th}|B:{bs}|M:{ms}|SK\n",
                    #     f"N:{int(stats['num_components'])}|R:{int(stats['num_rivers'])}|",
                    #     f"A:{int(stats['avg_size'])}\nC:{int(stats['river_coverage']*100)}|",
                    #     f"W:{int(stats['river_wastage']*100)}"
                    # ]
                    # txt = "".join(txt_arr)
                    # texts.append(txt)

                    sk = sketch.copy()[yMin:yMax, xMin:xMax]
                    rivers = strahler_paint_river(dem, ocean, water, PlanetConfig(top_n_orders=mi, brush_size=bs, threshold=th))[yMin:yMax, xMin:xMax]
                    sk[:,:,2] = rivers
                    stats = river_mask_stats(rivers, river_mask[yMin:yMax, xMin:xMax])
                    imgs.append(img.fromarray(sk))
                    txt_arr = [
                        f"T:{th}|B:{bs}|N:{mi}|ST\n",
                        f"N:{int(stats['num_components'])}|R:{int(stats['num_rivers'])}|",
                        f"A:{int(stats['avg_size'])}\nC:{int(stats['river_coverage']*100)}|",
                        f"W:{int(stats['river_wastage']*100)}"
                    ]
                    txt = "".join(txt_arr)
                    texts.append(txt)
                    pbar.update(1)
    # plt_grid(imgs, nbs, nms*nth*2, texts)
    return image_grid(imgs, nms*nbs, nth, texts)


def test_sketch_params(dem: np.ndarray, data_dir: str=os.path.join(os.getcwd(), 'data')) -> Image:
    '''
    Test different colour counts and brush sizes for painting sketches and output a grid of combinations.
    Args:
        dem (np.ndarray): The DEM to use to generate sketches
        data_dir (str): The directory to read the masks from
    Returns:
        Image: The grid of combinations
    '''
    h, w = dem.shape
    oceanmask = np.array(img.open(os.path.join(data_dir, f'world.oceanmask.{w}x{h}.png')))
    imgs = []
    texts = []
    center = (h//2, w*11//10//2)
    xMin = center[1] - 128
    xMax = center[1] + 128
    yMin = center[0] - 128
    yMax = center[0] + 128
    brush_sizes = range(0, 5, 1)
    nbs = len(brush_sizes)
    colour_counts = range(4, 9, 1)
    ncc = len(colour_counts)
    uniform = range(0, 2)
    nu = len(uniform)
    maxpixel = range(0, 2)
    nmp = len(maxpixel)
    with tqdm(total=nbs*ncc*nu*nmp, desc="Testing sketch params...", disable=(MAX_LOGGING_DEPTH<call_depth())) as pbar:
        for bs in brush_sizes:
            brush_deltas = get_brush_deltas(bs)
            for mp in maxpixel:
                for cc in colour_counts:
                    for u in uniform:
                        bucketing_mode = 'none'
                        if u == 1:
                            bucketing_mode = 'uniform'
                        if mp == 1:
                            bucketing_mode = 'global-max'
                        planet_cfg = PlanetConfig(bucketing_mode=bucketing_mode, colours=cc)
                        pixel_lists = numpy_get_pixel_lists(dem, oceanmask, )
                        sk = paint_img(pixel_lists, dem, brush_deltas, cc)
                        imgs.append(img.fromarray(sk[yMin:yMax, xMin:xMax]))
                        texts.append(f"U{u}|B{bs}|C{cc}|M{mp}")
                        pbar.update(1)
    
    return image_grid(imgs, nbs*nmp, ncc*nu, texts)

def draw_grid_lines(im: np.ndarray) -> np.ndarray:
    '''
    Draw red grid lines every 256 pixels on the given image.
    Args:
        im (np.ndarray): The image to draw grid lines on
    Returns:
        np.ndarray: The image with grid lines drawn on it
    '''
    
    assert len(im.shape) == 3, "Image must be 3D"

    h, w = im.shape[:2]
    to_return = im.copy()
    for i in range(0, h, 256):
        to_return[i, :, 0] = 255
        to_return[i, :, 1:] = 0
        to_return[i, :, -1] = 255
    for i in range(0, w, 256):
        to_return[:, i, 0] = 255
        to_return[:, i, 1:] = 0
        to_return[:, i, -1] = 255

    to_return[-1, :, 0] = 255
    to_return[-1, :, 1:] = 0
    to_return[-1, :, -1] = 255

    to_return[:, -1, 0] = 255
    to_return[:, -1, 1:] = 0
    to_return[:, -1, -1] = 255

    
    return to_return
    


def gen_sketches(sizes: list, planet_cfg: PlanetConfig, data_dir=os.path.join(os.getcwd(), "data"), 
                 rivermask_offset: int=6, force: bool=False, draw_grids: bool=False) -> None:
    '''
    Generate sketches for the given sizes and brush size.
    If river_size is greater than 0, rivers will be painted on the sketch in the blue channel.
    Args:
        sizes (list): The list of sizes to generate sketches for
        planet_cfg (PlanetConfig): The planet config to use for generating the sketches
        data_dir (str): The directory to save the sketches to
        rivermask_offset (int): The size offset used for supersampling the rivermask
        force (bool): Whether to force generation of sketches even if they already exist
        draw_grids (bool): Whether to draw grid lines on the sketches
    Returns:
        None (saves sketches to disk)
    '''
    brush_size = planet_cfg.brush_size
    colours = planet_cfg.colours
    river_size = planet_cfg.river_size
    threshold = planet_cfg.threshold
    min_size = planet_cfg.min_size
    top_n_orders = planet_cfg.top_n_orders

    brush_deltas = get_brush_deltas(brush_size)
    folder = planet_cfg.sketch_str()
    for i, (w, h) in enumerate(sizes):
        H = min(h*2**rivermask_offset, 8192)
        W = min(w*2**rivermask_offset, 16384)
        output_dir = os.path.join(data_dir, "sketches", f"{folder}", f"{w}x{h}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        json_dir = os.path.join(data_dir, "json", f"{folder}", f"{w}x{h}") 
        if not os.path.exists(json_dir):
            os.makedirs(json_dir)
        
        output_dir = os.path.join(output_dir, "sketch.png")
        json_dir = os.path.join(json_dir, f"pixel_lists.json")
        if os.path.exists(output_dir) and not force:
            continue
        DEM = np.array(img.open(os.path.join(data_dir, f'World_DEM_{w}x{h}.png')))
        oceanmask = np.array(img.open(os.path.join(data_dir, f'world.oceanmask.{w}x{h}.png')))
        test_sketch_params(DEM).save(output_dir.replace('sketch.png', f'{w}x{h}_sketch.png'))
        pixel_lists = numpy_get_pixel_lists(DEM, oceanmask, planet_cfg)
        SKETCH = paint_img(pixel_lists, DEM, brush_deltas, colours)
        if river_size:
            
            # test_river_params(SKETCH, oceanmask, watermask, rivermask_offset).save(output_dir.replace('sketch.png', f'{w}x{h}_{rivermask_offset}.png'))
            test_river_mask_params(SKETCH).save(output_dir.replace('sketch.png', f'{w}x{h}_{rivermask_offset}_mask.png'))
            watermask = np.array(img.open(os.path.join(data_dir, f'world.watermask.{W}x{H}.png')))
            oceanmask = np.array(img.open(os.path.join(data_dir, f'world.oceanmask.{W}x{H}.png')))
            downscale_factor = int(log2(W//w))
            river_mask = get_river_mask(oceanmask, watermask, threshold, downscale_factor)
            # rivers = skeletonize_paint_river(river_mask, river_size, min_size)
            rivers = strahler_paint_river(DEM, 
                                          np.array(img.open(os.path.join(data_dir, f'world.oceanmask.{w}x{h}.png'))), 
                                          np.array(img.open(os.path.join(data_dir, f'world.watermask.{w}x{h}.png'))), 
                                          planet_cfg)
            # rivers = paint_river(rivers, river_size)
            # remove_small_components(rivers, min_size=min_size, radius=radius, river_size=river_size, connect=True)
            # rivers = numpy_remove_small_components(rivers, min_size=min_size, radius=radius, river_size=river_size)
            SKETCH = np.dstack((SKETCH, SKETCH, rivers))
        # Save the outputs
        img.fromarray(DEM).save(output_dir.replace('sketch.png', 'dem.png'))
        if draw_grids:
            SKETCH = draw_grid_lines(SKETCH)
        img.fromarray(SKETCH).save(output_dir)



def create_binary_mask(mask: np.ndarray, threshold: int=127) -> np.ndarray:
    '''
    Create a binary mask from the given mask
    Args:
        mask (np.ndarray): The mask to convert to binary
        threshold (int): The threshold to use to convert to binary
    Returns:
        np.ndarray: The binary mask
    '''
    # Resize mask into single channel
    mask = np.min(mask, axis=2)
    mask[mask < threshold] = 0
    mask[mask >= threshold] = 255
    return mask


def create_binary_masks(count: int, data_dir=os.path.join(os.getcwd(), "data")) -> None:
    '''
    Create binary masks for the landmass masks extracted from the oceanmask in GIMP.
    Args:
        count (int): The number of masks to use
        data_dir (str): The directory to save the masks to
    Returns:
        None (saves binary masks to disk)
    '''
    for i in range(count):
        mask = img.open(os.path.join(data_dir, f"masks/data/00{i}.png"))
        mask = create_binary_mask(np.array(mask))
        img.fromarray(mask, mode='L').convert('1').save(os.path.join(data_dir, f"masks/{i}.png"))


def extract_binary_masks(count: int, brush_size: int=5, sizes: list=[(512, 256)], data_dir=os.path.join(os.getcwd(), "data")) -> None:
    '''
    Extract the binary masks from each size based on the 512x256 masks.
    Args:
        count (int): The number of masks to use
        brush_size (int): The size of the brush to use to paint the sketch
        sizes (list): The list of sizes to generate sketches for
        data_dir (str): The directory to save the masks to
    Returns:
        None (saves binary masks to disk) 
    '''
    masks = []
    for i in range(count):
        mask = img.open(os.path.join(data_dir, f"masks/{i}.png"))
        mask = np.array(mask)
        masks.append(mask)
    with tqdm(total=len(sizes)*len(masks), desc="Extracting binary masks...", disable=(MAX_LOGGING_DEPTH<call_depth())) as pbar:
        for w, h in sizes:
            for i, mask in enumerate(masks): 
                
                output_dir = os.path.join(data_dir, f"masks/{w}x{h}/brush_{brush_size}")
                if not os.path.exists(output_dir):
                    os.mkdir(output_dir)
                if os.path.exists(os.path.join(output_dir, f"{i}.png")):
                    pbar.update(1)
                    continue
                # Resize mask to the size of the big mask
                landmass = mask.repeat(h//256, axis=0).repeat(w//512, axis=1)
                bigmask = img.open(os.path.join(data_dir, f"masks/{w}x{h}/brush_{brush_size}/mask.png"))
                bigmask = np.array(bigmask)
            
                bigmask[bigmask < 127] = False
                bigmask[bigmask >= 127] = True
                # Binary and the masks together
                landmass = np.bitwise_and(bigmask, landmass)
                landmass *= 255
                # Save the landmass mask
                img.fromarray(landmass, mode='L').convert('1').save(os.path.join(output_dir, f"{i}.png"))
                pbar.update(1)


# create_outline(np.array(img.open(os.path.join(data_dir, f'world.oceanmask.1024x512.png'))), 5)
# create_outlines(5)
# create_binary_masks(7)
if __name__ == "__main__":
    # test_brushes()
    # size = 5
    # w, h = global_sizes[size]
    # dem = img.open(os.path.join(os.getcwd(), "data", f"World_DEM_{w}x{h}.png"))
    # dem = np.array(dem)

    # oceans = img.open(os.path.join(os.getcwd(), "data", "masks", f"{w}x{h}", "brush_3", "5.png"))
    # oceans = np.array(oceans).astype(np.uint8)*255

    # water = img.open(os.path.join(os.getcwd(), "data", f"world.watermask.{w}x{h}.png"))
    # water = np.array(water)

    # rivers = strahler_paint_river(dem, oceans, water, 7, 2, 200)

    # img.fromarray(rivers).save("Test.png")

    # exit(0)

    # create_masks(0, global_sizes, force=False)
    # extract_binary_masks(7, 0, global_sizes)
    planet_cfg = PlanetConfig(size=3, downscale_offset=3)
    # gen_sketches(sizes=global_sizes, planet_cfg=planet_cfg, data_dir=os.path.join(os.getcwd(), "data"), 
    #              rivermask_offset=6, force=True, draw_grids=True)
    # test = np.ndarray((256, 256), dtype=np.uint8)
    from .atlas_loader import AtlasLoader
    h = 2048
    w = 2 * h
    loader = AtlasLoader(planet_cfg)

    dem = loader.float_dem
    landcover = loader.land
    rivers = AtlasLoader(PlanetConfig()).river_upa
    temp = loader.temp_sketch

    landcover = translate_land(landcover, single_water_class=False)
    kernel_size = 5
    circular_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # dem = cv2.erode(dem, kernel=circular_kernel)
    # dem = cv2.GaussianBlur(dem, (kernel_size, kernel_size), sigmaX=5)
    dem = cv2.resize(dem, (w, h), interpolation=cv2.INTER_LANCZOS4)
    landcover = cv2.resize(landcover, (w, h), interpolation=cv2.INTER_NEAREST)
    temp = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

    # dem[(landcover > 0) & (dem == 0)] = 1
    # Avoid circular import
    from planetAI.src.data.mean_precip import PrecipSketch
    weights = PrecipSketch(planet_cfg).get_sketch(dem, landcover, temp) * 2 ** 5

    efficiency = np.ones_like(weights)

    weights[9 * h // 10:] = 0

    river_acc, fdir = accumulation(dem, weights, efficiency, planet_cfg, return_fdir=True)

    eroded_dem = hydro_erode_dem(dem, river_acc, fdir)

    def get_river_acc(dem, weights, efficiency, planet_cfg) -> img.Image:
        river_acc = accumulation_with_erosion(dem, weights, efficiency, planet_cfg)
        river_acc = cv2.normalize(
            river_acc, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        return img.fromarray(np_rgb(river_acc, "viridis"))

    weight_mults = [0, 1, 2, 3, 4][1:2]
    efficiency_mults = [1.0, 0.5, 0.25, 0.1, 0][:1]

    grid_images = []
    image_labels = []

    for wm, em in itertools.product(weight_mults, efficiency_mults):
        grid_images.append(get_river_acc(dem, weights ** wm, efficiency * em, planet_cfg))
        image_labels.append(f"Weight: {wm}, Eff: {em}")

    dem[dem > 4] = 4
    dem *= 60

    image_grid(
        grid_images, len(efficiency_mults), len(weight_mults), image_labels
    ).save(os.path.join(planet_cfg.test_dir, "river_acc.png"))

    rivers = cv2.normalize(rivers, None, 0, 255, cv2.NORM_MINMAX)

    img.fromarray(gray_to_land(landcover)).save(os.path.join(planet_cfg.test_dir, "river_acc_land.png"))
    img.fromarray(np_rgb(dem, "viridis")).save(os.path.join(planet_cfg.test_dir, "river_acc_dem.png"))
    img.fromarray(np_rgb(rivers, "viridis")).save(os.path.join(planet_cfg.test_dir, "river_acc_real_upa.png"))
