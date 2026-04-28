from math import ceil
import numpy as np
import torch
from PIL import Image as img
from PIL.Image import Image
import os
from PIL import ImageDraw 
from PIL import ImageFont
from PIL import UnidentifiedImageError
import inspect
from functools import wraps
from time import time
import matplotlib.pyplot as plt
from typing import Union, Optional, List, Tuple, Dict, Callable
from dataclasses import dataclass, field, fields, replace
import re
import random
import warnings
import psutil
from skimage.filters.rank import modal
from scipy import stats
from skimage.morphology import rectangle
from diffusers.utils.torch_utils import randn_tensor
import json
import cv2
from torchvision.transforms import ToPILImage
from enum import Enum
from .landcover_utils import gray_to_land
from functools import lru_cache
from datetime import datetime

try:
    import planet_defaults
except:
    from . import planet_defaults
import gc

try:
    from numba import njit, prange
except ImportError:
    prange = range

    def njit(*args, **kwargs):
        def wrapper(func):
            return func

        return wrapper

def format_memory(bytes):
    return f'{bytes/1024**3:.1f} GiB'

data_dir = os.path.join(os.getcwd(), "data")
class MaskStore:
    '''
    Central store for all masks including different planet configurations
    This can be used to quickly access as many masks as possible
    Members:
        mask_dict (dict): A dictionary of all of the masks
        removal_queue (list[str]): A priority queue of the next mask to remove when memory is limited
        memory_limit (int): The maximum number of bytes to store in memory
        current_memory (int): The current number of bytes stored in memory
    '''
    mask_dict: dict = {}
    removal_queue: List[Tuple[str, str, str]] = []
    memory_limit: int = 1e8 # 1e9: 1 GB, 1e10: 10 GB, 1e11: 100 GB
    current_memory: int = 0
    save_bounds: bool = False

    def __init__(self, memory_limit: int = 1000000000):
        '''
        Initialize the mask store
        Args:
            memory_limit (int): The maximum number of bytes to store in memory
        '''
        # print(f"Initializing mask store with memory limit {format_memory(memory_limit)}")
        self.memory_limit = memory_limit

    def remove_next(self) -> bool:
        '''
        Remove the next mask from the store
        Returns:
            bool: True if a mask was removed, False otherwise
        '''
        
        # TODO: Change to priority queue based on memory usage and age
        if len(self.removal_queue) <= 1:
            return False
        try:
            planet_str, mask_type, name = self.removal_queue.pop(0)
            
            data = self.mask_dict[planet_str][mask_type][name][0]
            self.current_memory -= data.nbytes
            self.mask_dict[planet_str][mask_type].pop(name)
            return True
        except:
            self.purge()
            return True
        # print(f"Removed {planet_str}/{mask_type}/{name} from mask store")
        # print(f"Total process memory: {format_memory(total_mem())}")

    def _load_mask(self, planet_str: str, mask_name: str, mask_dir: str, mask_type: str) -> None:
        '''
        Load a mask into the store
        Args:
            planet_str (str): The planet configuration
            mask_name (str): The name of the mask
            mask_dir (str): The directory of the mask
            mask_type (str): The type of mask
        '''
        if not planet_str in self.mask_dict:
            self.mask_dict[planet_str] = {}
        if not mask_type in self.mask_dict[planet_str]:
            self.mask_dict[planet_str][mask_type] = {}

        if mask_name in self.mask_dict[planet_str][mask_type]:
            return
        mask = open_image_array(os.path.join(mask_dir, mask_name))
        if mask is None:
            return
        self._add_mask(planet_str, mask_name, mask, mask_type)
        if self.save_bounds:
            self._save_bounds(planet_str, mask_name, mask_dir, mask_type)

    def _add_mask(self, planet_str: str, mask_name: str, mask: np.ndarray, mask_type: str, 
                  bounds: Tuple[int, int, int, int]=None, shape: Tuple[int, int]=None) -> None:
        if total_mem() > 16*1024**3:
            self.purge()
        if not planet_str in self.mask_dict:
            self.mask_dict[planet_str] = {}
        if not mask_type in self.mask_dict[planet_str]:
            self.mask_dict[planet_str][mask_type] = {}
        data = mask
        if bounds is None:
            bounds = get_bounds(mask)
            ymin, xmin, ymax, xmax = bounds
            data = mask[ymin:ymax+1, xmin:xmax+1]
        if shape is None:
            shape = mask.shape
        

        self.mask_dict[planet_str][mask_type][mask_name] = (data, bounds, shape)
        self.removal_queue.append((planet_str, mask_type, mask_name))
        self.current_memory += data.nbytes
        removed = True
        while self.current_memory > self.memory_limit and removed:
            removed = self.remove_next()
        if total_mem() > 18*1024**3:
            raise MemoryError('Memory usage is too high. Try reducing the memory limit or increasing the memory limit.')


    def _check_mask(self, planet_str: str, mask_name: str, mask_type: str) -> bool:
        '''
        Check if a mask is in the store
        Args:
            planet_str (str): The planet configuration
            mask_name (str): The name of the mask
            mask_type (str): The type of mask
        Returns:
            bool: True if the mask is in the store
        '''
        if not planet_str in self.mask_dict:
            return False
        if not mask_type in self.mask_dict[planet_str]:
            return False
        if not mask_name in self.mask_dict[planet_str][mask_type]:
            return False
        return True
    
    def _get_mask(self, planet_str: str, mask_name: str, mask_type: str) -> Union[np.ndarray, None]:
        '''
        Get a mask from the store
        Args:
            planet_str (str): The planet configuration
            mask_name (str): The name of the mask
            mask_type (str): The type of mask
        Returns:
            np.ndarray: The mask
        '''
        if not self._check_mask(planet_str, mask_name, mask_type):
            return None

        data = self.mask_dict[planet_str][mask_type][mask_name][0]
        bounds = self.mask_dict[planet_str][mask_type][mask_name][1]
        shape = self.mask_dict[planet_str][mask_type][mask_name][2]

        to_return = np.zeros(shape, dtype=np.uint8)

        ymin, xmin, ymax, xmax = bounds
        to_return[ymin:ymax+1, xmin:xmax+1] = data
        return to_return

    def _save_bounds(self, planet_str: str, mask_name: str, mask_dir: str, mask_type: str) -> None:
        '''
        Save the data part of a mask with it's bounds as a metadata json file.
        The mask must be in the store.
        Args:
            planet_str (str): The planet configuration
            mask_name (str): The name of the mask
            mask_dir (str): The directory of the mask
            mask_type (str): The type of mask
        '''
        if not self._check_mask(planet_str, mask_name, mask_type):
            return
        data = self.mask_dict[planet_str][mask_type][mask_name][0]
        bounds = self.mask_dict[planet_str][mask_type][mask_name][1]
        shape = self.mask_dict[planet_str][mask_type][mask_name][2]

        to_save_name = mask_name.replace('.', '_bounds.')
        to_save_metadata_name = mask_name.replace('.', '_bounds.json')
        with open(os.path.join(mask_dir, to_save_metadata_name), 'w') as f:
            json.dump({
                'bounds': [int(b) for b in list(bounds)], 
                'shape': [int(s) for s in list(shape)]}, f)
        save_image_array(os.path.join(mask_dir, to_save_name), data)

    def _load_bounds(self, planet_str: str, mask_name: str, mask_dir: str, mask_type: str) -> None:
        '''
        Load the data part of a mask with it's bounds from a metadata json file.
        Args:
            planet_str (str): The planet configuration
            mask_name (str): The name of the mask
            mask_dir (str): The directory of the mask
            mask_type (str): The type of mask
        '''
        if self._check_mask(planet_str, mask_name, mask_type):
            return
        to_load_name = 'bounds_' + mask_name
        to_load_metadata_name = 'bounds_' + mask_name.split('.')[0] + '.json'
        if not os.path.exists(os.path.join(mask_dir, to_load_name)):
            return

        with open(os.path.join(mask_dir, to_load_metadata_name), 'r') as f:
            metadata = json.load(f)
        bounds = metadata['bounds']
        shape = metadata['shape']

        data = open_image_array(os.path.join(mask_dir, to_load_name))
        if data is None:
            return
        self._add_mask(planet_str, mask_name, data, mask_type, bounds, shape)

    
    def get_mask(self, planet_str: str, mask_name: str, mask_type: str, mask_dir: str) -> Union[np.ndarray, None]:
        '''
        Get a mask from the store or load it into the store if not found
        Args:
            planet_str (str): The planet configuration
            mask_name (str): The name of the mask
            mask_type (str): The type of mask
            mask_dir (str): The directory of the mask
        '''

        if self.save_bounds and not self._check_mask(planet_str, mask_name, mask_type):
            self._load_bounds(planet_str, mask_name, mask_dir, mask_type)
        if not self._check_mask(planet_str, mask_name, mask_type):
            self._load_mask(planet_str, mask_name, mask_dir, mask_type)
        return self._get_mask(planet_str, mask_name, mask_type)

    def purge(self, purge_dems: bool=True):
        '''
        Purge the mask store
        '''
        print('Purging mask store')
        print('Before:', format_memory(self.current_memory), 'Total:', format_memory(total_mem()))
        self.mask_dict.clear()
        self.removal_queue.clear()
        gc.collect()
        print('After:', format_memory(self.current_memory), 'Total:', format_memory(total_mem()))
        
memory_info = psutil.virtual_memory()

def total_mem(): 
    return psutil.Process(os.getpid()).memory_info().rss

mask_store = MaskStore(memory_limit=1e8)  
import builtins

try:
    from line_profiler import profile
except:
    # If line_profiler is not available, timers will not be run.
    try:
        profile = builtins.__dict__['profile']
    except KeyError:
        # No line profiler, provide a pass-through version
        def profile(func): return func
        builtins.__dict__['profile'] = profile


def timing(f):
    '''
    Timing wrapper from https://stackoverflow.com/questions/1622943/timeit-versus-timing-decorator
    '''
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        if te - ts > 0.1:
            print('func:%r took: %2.4f sec' % \
            (f.__name__, te-ts))
        return result
    return wrap

@dataclass
class PlanetConfig:
    size: int = field(
        default=5,
        metadata={
            'help': 'Image size to generate. 0: 512x256, 1: 1024x512, 2: 2048x1024, 3: 4096x2048, 4: 8192x4096, 5: 16384x8192'
        }
    )
    brush_size: int = field(
        default=3,
        metadata={
            'help': 'Size of the brush'
        }
    )
    sketch_colours: int = field(
        default=4,
        metadata={
            'help': 'Number of colours to use'
        }  
    )
    bucketing_mode: str = field(
        default="uniform",
        metadata={
            'choices': ["none", "uniform", "local-max", "global-max"],
            'help': 'Bucketing mode for colour bucketing. "none": no bucketing, "uniform": uniform bucket sizes, "local-max": bucketing based on local maxima, "global-max": bucketing based on global maxima'
        }
    )
    min_size: int = field(
        default=500,
        metadata={
            'help': 'Minimum size of the connected river components'
        }
    )
    threshold: int = field(
        default=200,
        metadata={
            'help': 'Threshold for the river mask'
        }
    )
    top_n_orders: int = field(
        default=2,
        metadata={
            'help': 'Top n strahler orders to keep in the river mask'
        }
    )
    river_size: int = field(
        default=1,
        metadata={
            'help': 'Size of the river brush'
        }
    )
    iters: int = field(
        default=1000000,
        metadata={
            'help': 'Number of images to generate'
        }
    )
    mask_count: int = field(
        default=7,
        metadata={
            'help': 'Number of masks to use'
        }
    )
    planet_seed: int = field(
        default=None,
        metadata={
            'help': 'Seed to use for generating random planet list'
        }
    )
    offset: int = field(
        default=0,
        metadata={
            'help': 'The size offset used when generating transformation masks. Higher values help to prevent aliasing, but are also slower.'
        }
    )
    data_dir: str = field(
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data"),
        metadata={
            'help': 'Path to data directory'
        }
    )
    # TODO: Remove this
    use_colour_map: bool = field(
        default=False,
        metadata={
            'help': 'Use a colour map to colour the planet'
        }
    )
    # TODO: Remove this
    river_replace: bool = field(
        default=True,
        metadata={
            'help': 'Set channel 2 to the river mask or combine the river channel with the colour channel.\
                  The should only be set to false for visualisation purposes since the encoder will not work correctly.'
        }
    )
    image_mode: str = field(
        default="planet",
        metadata={
            'choices': [
                'planet', 'dem', 'sat', 'land', 'none', 'sketch-to-dem',
                'sketch-inpainting', 'dem-inpainting', 'sketch-upscaling', 'dem-upscaling',
                'noise', 'normal-noise', 'satellite', 'bathy', 'source', 'dem-to-satellite',
                'sketch-to-satellite', 'sat-upscaling', 'classifier',
            ],
            'help': 'The type of image pairs to generate.'
        }
    )
    inpainting_mode: str = field(
        default="both",
        metadata={
            'choices': ['blending', 'generation', 'both'],
            'help': 'The type of inpainting to use.'
        }
    )
    inpainting_width: float = field(
        default=0.0,
        metadata={
            'help': 'The width of the inpainting mask'
        }
    )
    inpainting_channels: int = field(
        default=None,
        metadata={
            'help': 'The number of channels to inpaint'
        }
    )
    sketch_mode: str = field(
        default="dilate",
        metadata={
            'choices': ['brush', 'dilate'],
            'help': 'The type of sketch to use. "brush": brush strokes, "dilate": dilation and erosion'
        }
    )
    dilate_size: int = field(
        default=1,
        metadata={
            'help': 'Size of the dilation brush'
        }
    )
    erode_size: int = field(
        default=1,
        metadata={
            'help': 'Size of the erosion brush'
        }
    )
    dilate_iters: int = field(
        default=0,
        metadata={
            'help': 'Number of dilation iterations'
        }
    )
    erode_iters: int = field(
        default=0,
        metadata={
            'help': 'Number of erosion iterations'
        }
    )
    dilate_first: bool = field(
        default=True,
        metadata={
            'help': 'Whether to do an extra dilate before erode'
        }
    )
    collision_mode: str = field(
        default="discard",
        metadata={
            'choices': ['discard', 'blend', 'translate'],
            'help': 'How to handle collisions between landmasses. "discard": discard the combination, "blend": blend the two landmasses, "translate": translate the landmasses to avoid collisions new location'
        }
    )
    discard_threshold: int = field(
        default=0,
        metadata={
            'help': 'Pixel overlap threshold for discarding landmasses. This is the number of pixels at the 512x256 and is scaled to the current size. Maximum: 300'
        }
    )
    gaussian_blur: int = field(
        default=0,
        metadata={
            'help': 'Gaussian blur size to apply to the dem before feature extraction to help with inprecise sketching'
        }
    )
    blur_size: int = field(
        default=7,
        metadata={
            'help': 'Size of the gaussian blur kernel to apply to the dem before feature extraction to help with inprecise sketching'
        }
    )
    variable_rivers: bool = field(
        default=False,
        metadata={
            'help': 'Use a different brush size for different river orders'
        }
    )
    river_dropout: float = field( # TODO Remove
        default=0.0,
        metadata={
            'help': 'Probability of dropping out the river channel in the dataloader'
        }
    )
    randomize_steps: int = field(
        default=0,
        metadata={
            'help': 'Number of steps before randomizing the planet. 0 means that randomization is disabled'
        }
    )
    purge_mask_store: bool = field(
        default=True,
        metadata={
            'help': 'Purge the mask store after each randomize'
        }
    )

    num_configurations: int = field(
        default=1,
        metadata={
            'help': 'Number of configurations to stack together'
        }
    )
    on_fly_conditioning: bool = field(
        default=True,
        metadata={
            'help': 'Generate conditioning on the fly if they are not already generated'
        }
    )
    on_fly_save: bool = field(
        default=False,
        metadata={
            'help': 'Save the generated conditioning on the fly'
        }
    )
    image_extension: str = field(
        default="png",
        metadata={
            'help': 'Image extension to use for saving images. ' + 
            'png: lossless (slow read, slow write), jpeg: lossy (fast read, fast write), tif: lossless (slow read, fast write)',
            'choices': ['png', 'jpeg', 'tif']
        }
    )
    use_mask_store: bool = field(
        default=True,
        metadata={
            'help': 'Use a mask store to speed up mask access times and reduce memory usage. Especially useful when using randomize_steps'
        }
    )
    use_parent_dem: bool = field( # TODO Remove
        default=False,
        metadata={
            'help': 'Use the parent dem as one of the input channels for all tasks. This should help with global coherence'
        }
    )
    use_surrounds: bool = field(
        default=False,
        metadata={
            'help': 'Use the surrounding tiles in the input sketch'
        }
    )
    use_modal_rivers: bool = field(
        default=False,
        metadata={
            'help': 'Inject rivers into the modal sketch sometimes. This should help with global coherence'
        }
    )
    parent_dem_padding: int = field(
        default=0,
        metadata={
            'help': 'The number of pixels to use around the parent dem. Maximum: 64'
        }
    )

    extra_rotations: bool = field(
        default=True,
        metadata={
            'help': 'Add extra rotations to the dataset'
        }
    )
    force_gen_list: bool = field(
        default=False,
        metadata={
            'help': 'Force generation of the planet list'
        }
    )
    add_size_to_colours: bool = field(
        default=False,
        metadata={
            'help': 'Add the size to the colours'
        }
    )
    use_defaults: bool = field(
        default=False,
        metadata={
            'help': 'Use the default values for the planet configuration'
        }
    )
    sea_level: int = field(
        default=0,
        metadata={
            'help': 'The sea level to use for the dem'
        }
    )
    sea_level_rescale: bool = field(
        default=False,
        metadata={
            'help': 'Rescale the dem after setting the sea level'
        }
    )
    use_sea_level: bool = field(
        default=False,
        metadata={
            'help': 'Change the sea level during randomization'
        }
    )
    gen_lists: bool = field(
        default=False,
        metadata={
            'help': 'Generate the planet lists'
        }
    )
    fast_tiles: bool = field(
        default=True,
        metadata={
            'help': 'Generate faster sketch and river tiles, but do not save them.'
        }
    )
    rotation_angle: int = field(
        default=1,
        metadata={
            'help': 'The angle step size for rotations'
        }
    )
    colour_dropout: float = field(
        default=0.0, # TODO Fine tune with this if models are good
        metadata={
            'help': 'Probability of dropping out colours in the sketch'
        }
    )
    downscale_offset: int = field(
        default=5,
        metadata={
            'help': 'The offset to subtract from the current zoom level for the downscale'
        }
    )
    downscaled: bool = field(
        default=False,
        metadata={
            'help': 'Whether the dataset is downscaled (don\'t change this)'
        }
    )
    preserve_edges: bool = field(
        default=True,
        metadata={
            'help': 'Preserve the edges of the DEM when deriving a sketch. Should only be used with upscaling'
        }
    )
    landcover_classes: int = field(
        default=10,
        metadata={
            'help': 'Number of landcover classes to use (excluding water)'
        }
    )
    spread_landcover: bool = field(
        default=False,
        metadata={
            'help': 'Spread the landcover classes to individual channels'
        }
    )
    rgb_landcover: bool = field(
        default=False,
        metadata={
            'help': 'Convert the landcover to rgb for the input.'
        }
    )
    temp_classes: int = field(
        default=5,
        metadata={
            'help': 'Number of temperature classes to use (excluding water)'
        }
    )
    replace_sketch: bool = field(
        default=False,
        metadata={
            'help':  '''Replace parts of the sketch unmasked outputs. 
                        This requires input and output channels to match semantically'''
        }
    )
    context_dropout: float = field(
        default=0.1,
        metadata={
            'help': 'Probability of dropping out the context channel in the dataloader'
        }
    )
    encoder_dropout: float = field(
        default=0.0,
        metadata={
            'help': 'Probability of dropping out the embedding in the dataloader'
        }
    )
    sphere_sampling: bool = field(
        default=True,
        metadata={
            'help': 'Use sphere sampling for the training tiles'
        }
    )
    sketch_lod_levels: int = field(
        default=2,
        metadata={
            'help': 'Use increased levels of detail for the sketch'
        }
    )
    use_quad_data: bool = field(
        default=False,
        metadata={
            'help': 'Use quad sphere data for the training tiles. Much faster but lower quality'
        }
    )
    use_summer: bool = field(
        default=True,
        metadata={
            'help': 'Include summer satellite/temperature data'
        }
    )
    use_mars_data: bool = field(
        default=True,
        metadata={
            'help': 'Use Mars data'
        }
    )
    river_upa_dropout_chance: float = field(
        default=0.25,
        metadata={
            'help': 'Chance to apply a random dropout threshold.'
        }
    )

    river_upa_variance: float = field(
        default=0.1,
        metadata={
            'help': 'Ratio of max value to vary river UPA by to regularize river inputs.'
        }
    )

    river_upa_min: float = field(
        default=5000,
        metadata={
            'help': 'Hand chosen threshold of 5000 km^2 to remove small rivers'
        }
    )

    discrete_rivers: bool = field(
        default=True,
        metadata={
            'help': 'Whether to treat rivers as distinct values or allow blurring'
        }
    )

    river_upa_variance_chance: float = field(
        default=0.25,
        metadata={
            'help': 'Chance to apply a random river variance.'
        }
    )
    min_angle: float = field(
        default=0.0,
        metadata={
            'help': 'Minimum angle in degrees to rotate tiles.'
        }
    )
    max_angle: float = field(
        default=15.0,
        metadata={
            'help': 'Maximum angle in degrees to rotate tiles.'
        }
    )
    single_water_class: bool = field(
        default=False,
        metadata={
            'help': 'Use a single class for inland water and ocean.'
        }
    )
    max_temp_variance: int = field(
        default=0,
        metadata={
            'help': 'The integer amount to vary the temperature by.'
        }
    )
    embedding_type: str = field(
        default="one-hot",
        metadata={
            'help': 'Use mode or distribution of classes for embedding.',
            'choices': ['one-hot', 'proportional', 'disabled']
        }
    )
    max_blur_radius: int = field(
        default=10,
        metadata={
            'help': 'The maximum blur radius for down-masks.'
        }
    )
    max_blur_amount: int = field(
        default=20,
        metadata={
            'help': 'The maximum blur amount for down-masks.'
        }
    )
    river_upa_mode: str = field(
        default="channel",
        metadata={
            'choices': ['none', 'channel', 'injected'],
            'help': 'How to handle river UPA. "none": no river UPA, "channel": use river UPA as a channel, "injected": inject river UPA into the dem'
        }
    )
    add_oceanmask_channel: bool = field(
        default=False,
        metadata={
            'help': 'Add an oceanmask image to the inputs'
        }
    )
    tile_mask_size: int = field(
        default=2,
        metadata={
            'help': 'Size of the tile mask. Controls how many tiles are sampled per epoch.'
        }
    )
    vflip_in_training: bool = field(
        default=False,
        metadata={
            "help": "Enable vertical flipping for train set."
        }
    )
    upscale_reg_strat: str = field(
        default="median+gauss",
        metadata={
            "help": "Type of regularization to use for upscaling inputs",
            "choices": ["none", "gauss", "mean", "median", "median+guass"]
        }
    )

    input_type_factory = {  # TODO change this back to a default factory
        'planet': ['downland_sketch', 'downtemp_sketch', 'downsketch'],
        'dem': ['downsketch'],
        'sat': ['downmodal', 'downsketch'],
        'sat-inpainting': ['sat', 'dem', 'mask', 'downmodal', 'downsketch'],
        'land': ['land', 'mask', 'downland_sketch', 'downtemp_sketch', 'downsketch'],
        'sketch-to-dem': ['sketch', 'river'],
        'sketch-inpainting': ['sketch', 'river'],
        'dem-inpainting': ['dem', 'mask', 'downsketch'],
        'sketch-upscaling': ['downsketch', 'downriver'],
        'dem-upscaling': ['downdem'],
        'none': ['none'],
        'noise': [],
        'normal-noise': [],
        'sketch-to-satellite': ['sat_sketch'],
        'dem-to-satellite': ['dem', 'land_sketch'],
        'satellite': ['sat_sketch', 'dem'],
        'sat-upscaling': ['downsat', 'downdem'],
        'bathy': ['downbathy_sketch'],
        'source': ['downland_sketch', 'downtemp_sketch', 'downsketch'],
        'classifier': ['downland_sketch', 'downtemp_sketch'],
    }
    output_type_factory = {
        'planet': ['sat', 'dem'],
        'dem': ['dem'],
        'sat': ['sat', 'dem'],
        'sat-inpainting': ['sat', 'dem'],
        'land': ['land'],
        'sketch-to-dem': ['dem'],
        'sketch-inpainting': ['sketch', 'river'],
        'dem-inpainting': ['dem'],
        'sketch-upscaling': ['sketch', 'river'],
        'dem-upscaling': ['dem'],
        'none': ['none'],
        'noise': ['dem'],
        'normal-noise': ['dem'],
        'sketch-to-satellite': ['sat'],
        'dem-to-satellite': ['sat'],
        'satellite': ['sat'],
        'sat-upscaling': ['sat', 'dem'],
        'bathy': ['bathy'],
        'source': ['sat', 'dem'],
        'classifier': ['sat', 'dem'],
    }

    no_operations: bool = field(
        default=False,
        metadata={
            'help': 'Disable all operations'
        }
    )


    satellite_colours = [
    0x000000,
    0x954a2a,
    0x3a2316,
    0x1a290a,
    0x38481f, 
    0x92784f,
    0xffffff, 
    0xa0acb2,  
    ]
    

    randomize_count = 0
    dem_dict = {}
    sketch_dict = {}
    river_dict = {}
    sat_dict = {}
    sat_sketch_dict = {}
    land_dict = {}
    land_sketch_dict = {}
    old_dem_dir = ''
    old_sketch_dir = ''
    old_river_dir = ''
    old_sat_dir = ''
    old_sat_sketch_dir = ''
    old_land_dir = ''
    old_land_sketch_dir = ''
    setup_done = False
    valid_tiles_list = []
    valid_coords = []

    @property
    def joint_channels(self):
        return "-".join(self.input_types) + "_" + "-".join(self.output_types)

    @property
    def add_river_upa_to_dem(self):
        return self.river_upa_mode == "injected"

    @property
    def add_river_upa_channel(self):
        return self.river_upa_mode == "channel"

    def __post_init__(self):
        if self.use_defaults:
            self.set_defaults()
        rot_list = [0, -15, 15, 165, 180, 195]
        angle = self.rotation_angle
        if self.extra_rotations:
            rot_list = list(range(-15, 16, angle))
            rot_list += list(range(165, 196, angle))
        self.operations = {
            #       Africa | South America | North America | Iceland | Australia | Eurasia | Antarctica
            'ref': [[False]]*7, # TODO Move reflection to dataloader

            'ang': [rot_list]*5 + [list(range(-5, 6, angle)) + list(range(175, 186, angle))] + [[0]]*1,
            'rem': [[True, False]]*7, 
        } if not self.no_operations else {
            'ref': [[False]]*7,
            'ang': [[0]]*7,
            'rem': [[True, False]]*7, 
        }
        self.spin = 32  # To allow size 0 -> size 5
        self.input_types: list[str] = [x for x in self.input_type_factory[self.image_mode]]
        self.output_types: list[str] = [x for x in self.output_type_factory[self.image_mode]]
        if self.add_river_upa_channel:
            self.input_types.append('river_upa')
        if self.add_oceanmask_channel:
            self.input_types.append('oceanmask')
        if self.downscaled:
            self.input_types = [x.replace('down', '') for x in self.input_types if 'down' in x]
            self.output_types = [x.replace('down', '') for x in self.output_types if 'down' in x]
        else:
            if any('down' in x for x in self.input_types + self.output_types) and self.downscale_offset <= 0:
                warnings.warn('Must use a downscale offset greater than 0 when using down input/output types. Setting to 1.')
                self.downscale_offset = 1

        if self.size > 3 and not self.use_mask_store:
            warnings.warn('Using a size greater than 3 without a mask store could result in the process being killed due to memory usage.')

        if self.use_parent_dem and self.size == 0:
            warnings.warn('No parent DEM for size 0. Disabling')
            self.use_parent_dem = False
        # Upscaling uses the current size as the target during training
        if not self.downscaled and ('upscaling' in self.image_mode or self.use_parent_dem or self.downscale_offset > 0):
            # if self.size < self.downscale_offset:
            #     print('Upscaling requires a size greater than or equal to downscale_offset since it uses the current size as the target during training')
            #     self.downscale_offset = self.size
            self.downscale_cfg = replace(self, size=self.size-self.downscale_offset, use_parent_dem=False, downscaled=True)
        else:
            self.downscale_cfg = None

        if self.randomize_steps > 0 and not self.on_fly_conditioning:
            warnings.warn('Using randomize_steps without on_fly_conditioning will result in a much slower training process.')
            self.on_fly_conditioning = True
        if self.randomize_steps > 0 and self.size > 2 and not self.fast_tiles:
            warnings.warn('Using randomize_steps with size greater than 2 without fast_tiles will result in a much slower training process.')
            self.fast_tiles = True
        if self.add_size_to_colours:
            self.colours = self.size + self.sketch_colours
        else:
            self.colours = self.sketch_colours
        self.sketch_max_pixel = 254
        self.dem_max_pixel = 255
        if self.inpainting_channels is None:
            i = 0
            c = 0
            maxi = min(len(self.input_types), len(self.output_types))
            while (i < maxi) and self.input_types[i] == self.output_types[i]:
                c += self.mask_channels()[self.input_types[i]]
                i += 1
            if i < len(self.input_types) and self.input_types[i] == 'mask':
                c += 1
            self.inpainting_channels = c
        self.setup_done = True
        if self.old_dem_dir != self.dem_dir(): 
            self.dem_dict = {}
            self.setup_done = False
        if self.old_sketch_dir != self.sketch_dir():
            self.sketch_dict = {}
            self.setup_done = False
        if self.old_river_dir != self.river_dir():
            self.river_dict = {}
            self.setup_done = False
        if self.old_sat_dir != self.sat_dir():
            self.sat_dict = {}
            self.setup_done = False
        if self.old_sat_sketch_dir != self.sat_sketch_dir():
            self.sat_sketch_dict = {}
            self.setup_done = False
        if self.old_land_dir != self.land_dir():
            self.land_dict = {}
            self.setup_done = False
        if self.old_land_sketch_dir != self.land_sketch_dir():
            self.land_sketch_dict = {}
            self.setup_done = False

        self.old_dem_dir = self.dem_dir()
        self.old_sketch_dir = self.sketch_dir()
        self.old_river_dir = self.river_dir()
        self.old_sat_dir = self.sat_dir()
        self.old_sat_sketch_dir = self.sat_sketch_dir()
        self.old_land_dir = self.land_dir()
        self.old_land_sketch_dir = self.land_sketch_dir()

        self.mask_dirs = {
            'dem': self.dem_dir(),
            'sketch': self.sketch_dir(),
            'river': self.river_dir(),
            'sat': self.sat_dir(),
            'sat_sketch': self.sat_sketch_dir(),
            'land': self.land_dir(),
            'land_sketch': self.land_sketch_dir(),
        }


        
        self.mask_store = mask_store
        assert self.bucketing_mode in ["none", "uniform", "local-max", "global-max"], 'bucketing_mode must be one of "none", "uniform", "local-max", "global-max"'
        if self.bucketing_mode == 'global-max':
            # dem = np.array(img.open(os.path.join(self.data_dir, f"World_DEM_{self.get_dims_str()}.png")))
            # self.dem_max_pixel = np.max(dem)
            self.dem_max_pixel = [174, 183, 193, 199, 235, 250][self.size] 
            
            # TODO: Set this statically

        if self.collision_mode == 'translate':
            # We only translate in the x direction to retain geological accuracy
            self.translations = [[0,-40/256], [0,-85/256], [0,0], [0,0], [0,0], [0,0], [0,0]]
        else:
            self.translations =[[0,0]]*7 

        self.values = {
            # 'sketch_colours': [4, 5, 6],
            'min_size': [50],
            'threshold': [50, 100],
            'top_n_orders': [0, 1, 2, 3],
            'river_size': [1, 2],
            'gaussian_blur': [0, 1, 5, 10],
            'blur_size': [1, 3, 5, 7],
            'discard_threshold': [0],
            'variable_rivers': [True, False],
            'sea_level': [0]*10 + list(range(15)) if self.use_sea_level else [0],
            'sea_level_rescale': [False],
            # 'bucketing_mode': ["none", "uniform", "local-max", "global-max"], # TODO Justify removing this
        } 
        if self.use_sea_level and self.num_configurations > 1:
            self.values['sea_level'] += list(range(15, 30)) + list(range(30, 55, 5))
        if self.sketch_mode == 'brush':
            self.values.update({
                'brush_size': [2, 3, 4],
            })
        else:
            self.values.update({
                'erode_size': [1, 2],
                'dilate_size': [2, 3, 4],
                'erode_iters': [0, 1],
                'dilate_iters': [1],
                'dilate_first': [True, False],
            })

    @property
    def delta(self):
        return 2**self.downscale_offset

    @property
    def full_delta(self):
        return 2 ** (5 - self.size)

    @property
    def H(self):
        return 256 * 2 ** self.size
    
    @property
    def W(self):
        return 512 * 2 ** self.size
    
    @property
    def h(self):
        return int(256 * 2 ** self.downscale_cfg.size)
    
    @property
    def w(self):
        return int(512 * 2 ** self.downscale_cfg.size)

    def set_defaults(self):
        defaults = planet_defaults.defaults[self.size]
        for k in defaults:
            setattr(self, k, defaults[k])

    def combined_classes(self):
        '''
        Get the number of combined classes
        '''
        return (self.colours + 1) * (self.temp_classes + 1) * (self.landcover_classes + 1)
    
    def get_param_dict(self):
        '''
        Get a dictionary of the parameters
        '''
        return {k: getattr(self, k) for k in planet_defaults.defaults[self.size]}

    def mask_types(self):
        '''
        Get the mask types for each image mode
        '''
        return list(filter(lambda x: not 'down' in x, self.input_types + self.output_types))

    @property
    def landcover_channels(self):
        return self.landcover_classes+1 if self.spread_landcover else 3 if self.rgb_landcover else 1

    def mask_channels(self):
        '''
        Get the number of channels of each type of mask
        '''
        return {
            'dem': 1,
            'sketch': 1,
            'river': 1,
            'sat': 3,
            'sat_sketch': 3,
            'land': self.landcover_channels,
            'land_sketch': self.landcover_channels,
            'temp': 1,
            'temp_sketch': 1,
            'mask': 1,
            'none': 1,
            'modal': 3,
            'bathy': 1,
            'bathy_sketch': 1,
            'river_mask': 1,
            'river_upa': 1,
            'oceanmask': 1,
        }

    def input_channels(self):
        '''
        Get the number of input channels
        '''
        return sum(self.mask_channels()[x.replace('down', '')] for x in self.input_types)

    def mask_index(self, mask_type: str, type_list: list) -> int | None:
        '''
        Get the mask index within a list of mask types considering the mask channels of each type.
        Args:
            mask_type (str): The mask type
            type_list (list): The list of mask types
        Returns:
            int: The index of the mask
        '''
        i = 0
        for t in type_list:
            if t == mask_type:
                return i
            i += self.mask_channels()[t]
        return None

    def input_index(self, mask_type: str, clean: bool=True):
        '''
        Get the index of the input mask type in the stacked image
        Args:
            mask_type (str): The mask type
            clean (bool): Whether to clean the downs from the mask type
        '''
        if clean:
            mask_type = mask_type.replace('down', '')
        return self.mask_index(mask_type, self.clean_input_types())
    
    def output_channels(self):
        '''
        Get the number of output channels
        '''
        return sum(self.mask_channels()[x.replace('down', '')] for x in self.output_types)
    
    def output_index(self, mask_type: str):
        '''
        Get the index of the output mask type in the stacked image
        Args:
            mask_type (str): The mask type
        '''
        return self.mask_index(mask_type, self.clean_output_types())
    
    def clean_input_types(self):
        '''
        Get the input types without the downscaled types
        '''
        return [x.replace('down', '') for x in self.input_types]
    
    def clean_output_types(self):
        '''
        Get the output types without the downscaled types
        '''
        return [x.replace('down', '') for x in self.output_types]

    def store_check_setup(self, force: bool=False) -> bool:
        '''
        Check that all the necessary masks are present in the store given the operations
        Args:
            force (bool): Force the masks to be reloaded
        Returns:
            bool: True if all the masks are present, False otherwise
        '''
        mask_dirs = {
            'dem': self.dem_dir(),
            'sketch': self.sketch_dir(),
            'river': self.river_dir(),
            'sat': self.sat_dir(),
            'sat_sketch': self.sat_sketch_dir(),
            'land': self.land_dir(),
            'land_sketch': self.land_sketch_dir(),
        }
        ops = self.operations
        if self.setup_done and not force:
            return True
        if not self.mask_store:
            return False
        mask_types = self.mask_types()
        if self.on_fly_conditioning:
            mask_types = filter(lambda x: x in self.input_types + self.output_types, ['sat', 'dem', 'land'])
        for mt in mask_types:
            if not os.path.exists(mask_dirs[mt]):
                self.setup_done = False
                return False
            files = os.listdir(mask_dirs[mt])
            for m in range(self.mask_count):
                for ref in ops['ref'][m]:
                    for ang in ops['ang'][m]:
                        name = f'{m}_{ref}_{ang}.{self.image_extension}'
                        if self.fast_tiles:
                            pattern = rf'{m}_{ref}_{ang}_\d+_\d+.{self.image_extension}'
                            if not any(re.match(pattern, x) for x in files):
                                self.setup_done = False
                                return False
                        else:
                            if not os.path.exists(os.path.join(mask_dirs[mt], name)):
                                self.setup_done = False
                                return False
                        # if self.get_mask(name, mt) is None:
                        #     self.setup_done = False
                        #     return False
        self.setup_done = True                
        return True

    def check_setup(self, force: bool=False) -> bool:
        '''
        Check that all the necessary masks are present given the operations
        Args:
            force (bool): Force the masks to be reloaded
        Returns:
            bool: True if all the masks are present, False otherwise
        '''
        # TODO: Change this to just check if the files exist
        if self.use_mask_store:
            return self.store_check_setup(force)
        ops = self.operations
        if not (self.dem_dict and self.sketch_dict and self.river_dict) or force:
            self.dem_dict, self.sketch_dict, self.river_dict, self.sat_dict, self.sat_sketch_dict = self.get_mask_dicts()
        elif self.setup_done:
            return True
        mask_types = self.mask_types()
        dicts = []
        if 'dem' in mask_types:
            dicts.append(self.dem_dict)
        if 'sketch' in mask_types:
            dicts.append(self.sketch_dict)
        if 'river' in mask_types:
            dicts.append(self.river_dict)
        if 'sat' in mask_types:
            dicts.append(self.sat_dict)
        if 'sat_sketch' in mask_types:
            dicts.append(self.sat_sketch_dict)
        if 'land' in mask_types:
            dicts.append(self.land_dict)
        if 'land_sketch' in mask_types:
            dicts.append(self.land_sketch_dict)
        if self.on_fly_conditioning:
            dicts = [self.sat_dict] if 'satellite' in self.image_mode else [self.dem_dict]
        for d in dicts:
            for m in range(self.mask_count):
                for ref in ops['ref'][m]:
                    for ang in ops['ang'][m]:
                        if f'{m}_{ref}_{ang}.{self.image_extension}' not in d:
                            self.setup_done = False
                            return False
        self.setup_done = True                
        return True
    
    def get_max_pixel(self, np_img: Union[np.ndarray, None]=None, sketch: bool=False) -> int:
        assert not (self.bucketing_mode == 'local-max' and np_img is None), 'np_img must be provided when using local-max bucketing mode'
        if np.max(np_img) > self.sketch_max_pixel and sketch:
            self.sketch_max_pixel = np.max(np_img)
        if np.max(np_img) > self.dem_max_pixel and not sketch:
            self.dem_max_pixel = np.max(np_img)
        case = {
            "none": 255,
            "uniform": 255,
            "local-max": np.max(np_img),
            "global-max": self.sketch_max_pixel if sketch else self.dem_max_pixel,
        }
        return case[self.bucketing_mode]

    def river_keys(self):
        return [
            "min_size",
            "threshold",
            "mask_count",
            "offset",
            "collision_mode",
            "gaussian_blur",
            "blur_size",
            "sea_level",
            "sea_level_rescale",
        ]
    
    def extra_river_keys(self):
        return [
            "top_n_orders",
            "river_size",
            "variable_rivers",
        ]
    
    def sketch_keys(self):
        to_return = [
            "sketch_colours",
            "bucketing_mode",
            "mask_count",
            "offset",
            "use_colour_map",
            "collision_mode",
            "gaussian_blur",
            "blur_size",
            "sketch_mode",
            "sea_level",
            "sea_level_rescale",
        ]
        if self.sketch_mode == 'brush':
            to_return.append('brush_size')
        else:
            to_return.extend([
                "erode_size",
                "dilate_size",
                "erode_iters",
                "dilate_iters",
                "dilate_first",
            ])
        return to_return
    
    def randomize_parameters(self):
        '''
        Randomize the parameters of the dataset
        '''
        # TODO Possibly don't purge if fast tiles is enabled
        if self.purge_mask_store: 
            self.mask_store.purge(purge_dems=not self.fast_tiles)
        for k in self.values:
            vals = self.values[k]
            seed = self.planet_seed + self.randomize_count if self.planet_seed is not None else None
            random.Random(seed).shuffle(vals)
            # print(f"Randomizing {k} from {self.__dict__[k]} to {vals[0]}")
            setattr(self, k, vals[0])
        self.randomize_count += 1
        self.__post_init__()
        # Don't want brush mode
        # sketch_modes = ['brush', 'dilate']
        # seed = self.planet_seed + self.randomize_count if self.planet_seed is not None else None
        # random.Random(seed).shuffle(sketch_modes)
        # self.sketch_mode = sketch_modes[0]
        # self.__post_init__()
        # TODO Possibly randomize downscale_cfg if doing upscaling
        # print(f"Randomizing sketch parameters to {str(self.get_param_dict())}")

    def get_wandb_dict(self):
        '''
        Get a dictionary of the parameters values for wandb sweeps
        '''
        params = {}
        for k in self.values:
            params[k] = {}
            params[k]['values'] = self.values[k]
        return params 

    def __str__(self):
        return '-'.join(
            [
                f"SK{self.sketch_mode}",
                f"IM{self.image_mode}",
                f"CH{self.collision_mode[0]}",
                f"BR{self.brush_size}",
                f"CC{self.colours}",
                f"BM{self.bucketing_mode[0]}",
                f"MC{self.mask_count}",
                f"OF{self.offset}",
                f"CM{int(self.use_colour_map)}",
                f"PD{int(self.use_parent_dem)}",
                f"MS{self.min_size}",
                f"TH{self.threshold}",
                f"DT{self.discard_threshold}",
                f"RS{self.river_size}",
                f"TN{self.top_n_orders}",
                f"GB{self.gaussian_blur}",
                f"VR{int(self.variable_rivers)}",
                f"SL{self.sea_level}",
                f"SR{int(self.sea_level_rescale)}",
                f"SE{self.planet_seed}",
            ]
        ) if self.randomize_steps == 0 else '-'.join(
            [
                f"IM{self.image_mode}",
                f"MC{self.mask_count}",
                f"OF{self.offset}",
                f"CM{int(self.use_colour_map)}",
                f"PD{int(self.use_parent_dem)}",
                f"RD{str(self.randomize_steps > 0)[0]}",
                f"SE{self.planet_seed}",
            ]
        )

    

    def sketch_str(self):
        to_return = [
                "Sketch",
                f"SK{self.sketch_mode}",
                f"CC{self.colours}",
                f"BM{self.bucketing_mode[0]}",
                f"MC{self.mask_count}",
                f"OF{self.offset}",
                f"CM{int(self.use_colour_map)}",
                f"CT{str(self.collision_mode=='translate')[0]}",
                f"GB{self.gaussian_blur}",
                f"BS{self.blur_size}",
                f"SL{self.sea_level}",
                f"SR{int(self.sea_level_rescale)}",
            ]
        if self.sketch_mode == 'brush':
            to_return.append(f"BR{self.brush_size}")
        else:
            to_return.extend([
                f"ER{self.erode_size}",
                f"DI{self.dilate_size}",
                f"EI{self.erode_iters}",
                f"DI{self.dilate_iters}",
                f"DF{str(self.dilate_first)[0]}",
            ])
        return '-'.join(
            to_return
        )
    
    def river_str(self):
        return '-'.join(
            [
                "River",
                f"MS{self.min_size}",
                f"TH{self.threshold}",
                f"MC{self.mask_count}",
                f"OF{self.offset}",
                f"CT{str(self.collision_mode=='translate')[0]}",
                f"GB{self.gaussian_blur}",
                f"SL{self.sea_level}",
                f"SR{int(self.sea_level_rescale)}",
            ]
        )
    
    def dem_rgb_str(self):
        return '-'.join(
            [
                "DEM_RGB",
                f"CM{int(self.use_colour_map)}",
                f"OF{self.offset}",
                f"BM{self.bucketing_mode[0]}",
                f"CT{str(self.collision_mode=='translate')[0]}",
            ]
        )
    
    def dem_str(self):
        return '-'.join(
            [
                "DEM",
                f"OF{self.offset}",
                f"CT{str(self.collision_mode=='translate')[0]}",
            ]
        )
    
    def water_str(self):
        return '-'.join(
            [
                "Water",
                f"BR{self.brush_size}",
                f"OF{self.offset}",
                f"TH{self.threshold}",
                f"CT{str(self.collision_mode=='translate')[0]}",
            ]
        )
    
    def sat_str(self):
        return '-'.join(
            [
                "Sat",
                f"OF{self.offset}",
            ]
        )
    
    def sat_sketch_str(self):
        return '-'.join(
            [
                "Sat_Sketch",
                f"OF{self.offset}",
                f"CT{str(self.collision_mode=='translate')[0]}",
                f"GB{self.gaussian_blur}",
                f"BR{self.brush_size}",
                f"SL{self.sea_level}",
            ]
        )
    
    def land_str(self):
        return '-'.join(
            [
                "Land",
                f"OF{self.offset}",
            ]
        )

    def land_sketch_str(self):
        return '-'.join(
            [
                "Land_Sketch",
                f"OF{self.offset}",
                f"GB{self.gaussian_blur}",
                f"GB{self.gaussian_blur}",
                f"BS{self.blur_size}",
                f"ER{self.erode_size}",
                f"DI{self.dilate_size}",
                f"EI{self.erode_iters}",
                f"DI{self.dilate_iters}",
                f"DF{str(self.dilate_first)[0]}",
            ]
        )

    def gen_txt_str(self):
        return '-'.join(
            [
                "Gen",
                f"SZ{self.size}",
                f"MC{self.mask_count}",
                f"OF{self.offset}",
                f"CT{self.collision_mode}",
                f"DT{self.discard_threshold}",
                f"ER{self.extra_rotations}",
                f"SD{self.planet_seed}",
            ] + [f"SL{self.sea_level}"] if self.sea_level < 0 else []
        ) + '.txt'

    def get_dims(self) -> tuple:
        '''
        Get the dimensions of the planet image (width, height)
        '''
        return 512*2**self.size, 256*2**self.size
    
    def get_dims_str(self) -> str:
        '''
        Get the dimensions of the planet image as a string "widthxheight"
        '''
        return f"{512*2**self.size}x{256*2**self.size}"
    
    def transformation_dir(self):
        return os.path.join(self.data_dir, "transformations")
    
    @property
    def test_dir(self):
        _test_dir = os.path.join(self.data_dir, "test")
        os.makedirs(_test_dir, exist_ok=True)
        return _test_dir
    
    def sketch_dir(self):
        return os.path.join(self.transformation_dir(), self.sketch_str(), self.get_dims_str())

    def river_dir(self):
        return os.path.join(self.transformation_dir(), self.river_str(), self.get_dims_str())

    def dem_dir(self):
        return os.path.join(self.transformation_dir(), self.dem_str(), self.get_dims_str())
    
    def sat_dir(self):
        return os.path.join(self.transformation_dir(), self.sat_str(), self.get_dims_str())
    
    def sat_sketch_dir(self):
        return os.path.join(self.transformation_dir(), self.sat_sketch_str(), self.get_dims_str())
    
    def land_dir(self):
        return os.path.join(self.transformation_dir(), self.land_str(), self.get_dims_str())
    
    def land_sketch_dir(self):
        return os.path.join(self.transformation_dir(), self.land_sketch_str(), self.get_dims_str())
    
    def gen_txt_dir(self):
        return os.path.join(self.transformation_dir(), 'gen lists', self.gen_txt_str())

    def make_dirs(self):
        if 'sketch' in self.input_types+self.output_types:
            os.makedirs(self.sketch_dir(), exist_ok=True)
        if 'river' in self.input_types+self.output_types:
            os.makedirs(self.river_dir(), exist_ok=True)
        # Just always make the DEM, sat and land dirs to avoid issues
        os.makedirs(self.dem_dir(), exist_ok=True)
        os.makedirs(self.sat_dir(), exist_ok=True)
        os.makedirs(self.land_dir(), exist_ok=True)
        if 'sat_sketch' in self.input_types+self.output_types:
            os.makedirs(self.sat_sketch_dir(), exist_ok=True)
        
        if 'land_sketch' in self.input_types+self.output_types:
            os.makedirs(self.land_sketch_dir(), exist_ok=True)
        os.makedirs(os.path.join(self.transformation_dir(), 'gen lists'), exist_ok=True)

    def valid_tiles(self, force: bool=False):
        '''
        Return the y,x coords, mask number, reflection and angle of the tiles in the DEM dir
        Args:
            force (bool): Force the tiles to be reloaded
        Returns:
            list: A list of (y, x, mask, reflection, angle) tuples
        '''
        if self.valid_tiles_list and not force:
            return self.valid_tiles_list
        tiles_dirs = os.listdir(self.dem_dir())
        tiles = []
        for d in tiles_dirs:
            if not os.path.isdir(os.path.join(self.dem_dir(), d)):
                continue
            small_tiles = os.listdir(os.path.join(self.dem_dir(), d))
            for st in small_tiles:
                tiles.append(d + '_' + st)
        tiles = [x.split('.')[0].split('_') for x in tiles]
        tiles = list(filter(lambda x: len(x) == 5, tiles)) 
        valid = []
        for mask, ref, ang, y, x in tiles: # TODO This can be further optimized by using all data
            ref = ref == 'True'
            mask = int(mask)
            ang = int(ang)
            y = int(y)
            x = int(x)
            if ref in self.operations['ref'][mask] and ang in self.operations['ang'][mask]:
                valid.append((y, x, mask, ref, ang))
        self.valid_tiles_list = valid
        self.valid_coords = [(y, x) for y, x, _, _, _ in valid]
        return valid

        
    

    def check_created(self):
        return os.path.exists(self.transformation_dir())
    
    def get_mask_dicts(self, force=False) -> tuple: # TODO Remove
        '''
        DEPRECATED
        Get the mask dictionaries for the dem, sketch and river masks
        Args:
            force (bool): If True, will force the masks to be reloaded from disk
        Returns:
            tuple: (dem_dict, sketch_dict, river_dict)
        '''
        mask_types = self.mask_types()
        if force or (len(self.dem_dict) == 0 and 'dem' in mask_types):
            self.dem_dict = self.get_mask_dict(self.dem_dir())
        if force or (len(self.sketch_dict) == 0 and 'sketch' in mask_types):
            self.sketch_dict = self.get_mask_dict(self.sketch_dir())
        if force or (len(self.river_dict) == 0 and 'river' in mask_types):
            self.river_dict = self.get_mask_dict(self.river_dir())
        if force or (len(self.sat_dict) == 0 and 'sat' in mask_types):
            self.sat_dict = self.get_mask_dict(self.sat_dir())
        if force or (len(self.sat_sketch_dict) == 0 and 'sat_sketch' in mask_types):
            self.sat_sketch_dict = self.get_mask_dict(self.sat_sketch_dir())
        return self.dem_dict, self.sketch_dict, self.river_dict, self.sat_dict, self.sat_sketch_dict

    def get_mask_dict(self, trans_dir: str) -> dict:
        '''
        Get all of the possible masks from the transformations folder
        Args:
            transdir: The directory of the transformations
        Returns:
            A dictionary of all of the possible masks
        '''
        mask_dict = {}
        if not os.path.exists(trans_dir):
            return mask_dict
        regex = r'\d+\_(False|True)\_-?\d+\.' + self.image_extension
        for file in os.listdir(trans_dir):
            if re.match(regex, file):
                im = open_image_array(os.path.join(trans_dir, file))
                if im is not None:
                    mask_dict[file] = im
        return mask_dict

    def save_mask(self, mask: np.ndarray, mask_name: str, mask_type: str):
        '''
        Save a mask to disk and add it to the mask dictionary
        Args:
            mask (np.ndarray): The mask to save
            mask_name (str): The name of the mask
            mask_type (str): The type of mask (dem, sketch, river)
        '''
        dirs = {
            'dem': self.dem_dir(),
            'sketch': self.sketch_dir(),
            'river': self.river_dir(),
            'sat': self.sat_dir(),
            'sat_sketch': self.sat_sketch_dir(),
            'land': self.land_dir(),
            'land_sketch': self.land_sketch_dir(),
        }
        dicts = {
            'dem': self.dem_dict,
            'sketch': self.sketch_dict,
            'river': self.river_dict,
            'sat': self.sat_dict,
            'sat_sketch': self.sat_sketch_dict,
            'land': self.land_dict,
            'land_sketch': self.land_sketch_dict,
        }
        strs = {
            'dem': self.dem_str(),
            'sketch': self.sketch_str(),
            'river': self.river_str(),
            'sat': self.sat_str(),
            'sat_sketch': self.sat_sketch_str(),
            'land': self.land_str(),
            'land_sketch': self.land_sketch_str(),
        }
        if self.on_fly_save:
            os.makedirs(dirs[mask_type], exist_ok=True)
            img.fromarray(mask).save(os.path.join(dirs[mask_type], mask_name))
        if self.use_mask_store:
            self.mask_store._add_mask(strs[mask_type] + f'/{self.get_dims_str()}', mask_name, mask, mask_type)
        else:
            dicts[mask_type][mask_name] = mask

    def get_mask(self, mask_name: str, mask_type: str, y: int=None, x:int=None, tile_width: int=256) -> Union[np.ndarray, None]:
        '''
        Get a mask from the mask dictionary add tile
        Args:
            mask_name (str): The name of the mask
            mask_type (str): The type of mask (dem, sketch, river)
            y (int): The pixel y coordinate of the top left corner of the tile
            x (int): The pixel x coordinate of the top left corner of the tile
            tile_width (int): The width of the tile
        Returns:
            np.ndarray: The mask
        '''

        if self.use_mask_store:
            dirs = {
                'dem': self.dem_dir(),
                'sketch': self.sketch_dir(),
                'river': self.river_dir(),
                'sat': self.sat_dir(),
                'sat_sketch': self.sat_sketch_dir(),
                'land': self.land_dir(),
                'land_sketch': self.land_sketch_dir(),
            }
            strs = {
                'dem': self.dem_str(),
                'sketch': self.sketch_str(),
                'river': self.river_str(),
                'sat': self.sat_str(),
                'sat_sketch': self.sat_sketch_str(),
                'land': self.land_str(),
                'land_sketch': self.land_sketch_str(),
            }
            to_return = self.mask_store.get_mask(strs[mask_type] + f'/{self.get_dims_str()}', mask_name, mask_type, dirs[mask_type])
        else:
            dicts = {
                'dem': self.dem_dict,
                'sketch': self.sketch_dict,
                'river': self.river_dict,
                'sat': self.sat_dict,
                'sat_sketch': self.sat_sketch_dict,
            }
            if mask_name not in dicts[mask_type]:
                return None
            to_return = dicts[mask_type][mask_name]
        if to_return is None:
            to_return = np.zeros((tile_width, tile_width), dtype=np.uint8)
            if 'sat' in mask_type:
                to_return = np.zeros((tile_width, tile_width, 3), dtype=np.uint8)
        if y is not None and x is not None:
            to_return = _get_tile(to_return, y, x, tile_width)
        return to_return
    
    def colour_list(self):
        colour_size = 256 // self.colours
        return [0]+[i*colour_size-1 for i in range(1, self.colours+1)]
    
    def landcover_colour_list(self):
        colour_size = 256 // self.landcover_classes
        # 255 is for mars
        return [0] + [i*colour_size for i in range(1, self.landcover_classes+1)] + [255]
    
    def temp_colour_list(self):
        colour_size = 255 // self.temp_classes
        return [0]+[i*colour_size-1 for i in range(1, self.temp_classes+1)]
    
    def temp_to_int(self, temp: np.ndarray) -> np.ndarray:
        """
        Convert a temperature image to an integer temp image
        """
        temp = np.clip(temp, 0, 255).astype(np.uint16) + 1
        temp_step = 255 // self.temp_classes
        return temp // temp_step
    
    def int_to_temp(self, temp: np.ndarray) -> np.ndarray:
        """
        Convert an integer temp image to a temperature image
        """
        temp_step = 255 // self.temp_classes
        temp = temp.copy() * temp_step
        temp[temp > 0] -= 1
        return temp
    
    def dem_to_int(self, dem: np.ndarray) -> np.ndarray:
        """
        Convert a DEM image to an integer DEM image
        """
        dem = np.clip(dem, 0, 255).astype(np.uint16) + 1
        dem_step = 256 // self.colours
        return dem // dem_step
    
    def int_to_dem(self, dem: np.ndarray) -> np.ndarray:
        """
        Convert an integer DEM image to a DEM image
        """
        dem_step = 256 // self.colours
        dem = dem.copy() * dem_step
        dem[dem > 0] -= 1
        return dem

    def landcover_to_int(self, landcover: np.ndarray) -> np.ndarray:
        """
        Convert a landcover image to an integer landcover image
        """
        new_landcover = np.clip(landcover, 0, 255).astype(np.uint16)
        landcover_step = 256 // self.landcover_classes
        new_landcover = landcover // landcover_step
        new_landcover[landcover == 255] = self.landcover_classes + 1
        return new_landcover

    def int_to_landcover(self, landcover: np.ndarray) -> np.ndarray:
        """
        Convert an integer landcover image to a landcover image
        """
        landcover_step = 256 // self.landcover_classes
        landcover = landcover.copy() * landcover_step
        return landcover
    
    def temp_labels(self):
        labels = ["Water"]
        min_temp = 228 # K
        max_temp = 310 # K

        # Convert to Celsius
        min_temp -= 273
        max_temp -= 273
        temp_range = max_temp - min_temp
        temp_step = temp_range / self.temp_classes
        # Degree symbol: \u00b0
        for i in range(self.temp_classes):
            labels.append(f"{round(min_temp + i*temp_step, 1)}\u00b0C - {round(min_temp + (i+1)*temp_step, 1)}\u00b0C")
        return labels

    def sketch_labels(self):
        labels = ["Sea Level"]
        min_elevation = 0
        max_elevation = 8271
        elevation_range = max_elevation - min_elevation
        elevation_step = elevation_range / self.colours
        for i in range(self.colours):
            labels.append(f"{round(min_elevation + i*elevation_step, 1)}m - {round(min_elevation + (i+1)*elevation_step, 1)}m")
        return labels

    def post_process(self, cond_images: torch.tensor, outputs: torch.tensor) -> torch.tensor:
        """
        TODO: Fix this with the new mask system
        Post process outputs based on the conditional images
        Args:
            cond_images (torch.tensor): The conditioned images in shape (batch_size, channels, height, width)
            outputs (torch.tensor): The generated outputs
        Returns:
            tuple: The post processed outputs
        """

        return outputs

        # 1. Extract masks from cond_images
        mask_channel = self.mask_index('mask', self.mask_types())
        masks = cond_images[:, mask_channel]
        # 2. Replace unmasked regions in outputs with the corresponding regions in cond_images
        outputs = outputs.clone()
        for i in range(len(masks)):
            mask = (masks[i] + 1) / 2
            output = outputs[i].clone()
            for c in range(output.shape[0]):
                output[c] = output[c] * mask + cond_images[i, c] * (1-mask)
            outputs[i] = output
        # 3. Rescale each channel so that the border of the masked region corresponds to the border of the unmasked region
        for c in range(outputs.shape[1]):
            for i in range(len(masks)):
                mask = ((masks[i].cpu().detach().numpy() + 1)/2).astype(np.uint8)
                if mask.min() == 1:
                    continue
                output = outputs[i, c].clone()
                
                # Create a kernel to remove the corners of the mask
                # 010
                # 111
                # 010
                kernel = np.ones((3, 3), np.uint8)
                kernel[0, 0] = 0
                kernel[0, 2] = 0
                kernel[2, 0] = 0
                kernel[2, 2] = 0
                temp_output = output.clone()
                temp_output[mask == 1] = -1.0
                temp_output = tensor_to_np(temp_output)
                # Dilate temp_output to get outer border values on inner border positions
                temp_output = cv2.dilate(temp_output, kernel)
                temp_output = np_to_tensor(temp_output).to(output.device)
                # Dilate inverted mask and logical and to get inner border
                inner_border = cv2.bitwise_and(cv2.dilate(1-mask, kernel), mask)
                # Get all values on inner border
                inner_values: torch.tensor = output[inner_border == 1]
                # Get all values on outer border
                outer_values: torch.tensor = temp_output[inner_border == 1]


                diff = inner_values - outer_values
                # assert inner_values.shape == outer_values.shape, f"Inner and outer values are not the same shape: {inner_values.size} != {outer_values.size}"
                # Rescale values to match the outer border
                min_inner = torch.min(inner_values)
                max_inner = torch.max(inner_values)
                mean_inner = torch.mean(inner_values)
                min_outer = torch.min(outer_values)
                max_outer = torch.max(outer_values)
                mean_outer = torch.mean(outer_values)
                # TODO Normalise bassed on average difference between adjacent inner and outer values
                output[mask == 1] = output[mask == 1] - torch.mean(diff)
                output = torch.clamp(output, -1.0, 1.0)
                outputs[i, c] = output
        return outputs


    def output_display(
        self,
        cond_images: torch.tensor,
        outputs: torch.tensor,
        target_images: torch.tensor,
        show_outputs: bool = True
    ) -> torch.tensor:
        '''
        Create a grid of images for display given possibly batched images.
        Args:
            cond_images (torch.tensor): The conditioned images
            outputs (torch.tensor): The generated outputs
            target_images (torch.tensor): The target images
        Returns:
            torch.tensor: The grid of images
        '''
        input_channels = [self.mask_channels()[x.replace('down', '')] for x in self.input_types]
        output_channels = [self.mask_channels()[x.replace('down', '')] for x in self.output_types]
        grid = []
        device = cond_images.device
        old = self.landcover_classes == 9
        try:
            postprocessed = self.post_process(cond_images, outputs)
        except:
            # print("Post processing failed")
            postprocessed = outputs
        for cond, output, target, processed in zip(list(cond_images), list(outputs), list(target_images), list(postprocessed)):
            row = []
            if cond is not None:
                for i, c in enumerate(input_channels):
                    if 'land' in self.input_types[i] and not self.rgb_landcover:
                        land = tensor_to_np(cond[0])
                        if self.spread_landcover:
                            land = spread_to_continuous(land)
                        land = gray_to_land(land, old)
                        row.append(np_to_tensor(land).to(device))
                    elif (
                        'river_upa' in self.input_types[i] or
                        'dem' in self.input_types[i] or
                        self.input_types[i] in ("sketch", "downsketch")
                    ):
                        river_upa = tensor_to_np(cond[0])
                        river_upa = np_rgb(river_upa, cmap='viridis')
                        row.append(np_to_tensor(river_upa).to(device))
                    elif "temp" in self.input_types[i]:
                        temp = tensor_to_np(cond[0])
                        temp = np_rgb(temp, cmap="coolwarm")
                        row.append(np_to_tensor(temp).to(device))
                    else:
                        row.append(torch.stack(list(cond[:c])+[cond[0]]*(3-c)))
                    cond = cond[c:]
            if show_outputs:
                for i, o in enumerate(output_channels):
                    if 'land' in self.output_types[i] and not self.rgb_landcover:
                        land = tensor_to_np(output[0])
                        if self.spread_landcover:
                            land = spread_to_continuous(land)
                        land = gray_to_land(land, old)
                        row.append(np_to_tensor(land).to(device))
                    elif "dem" in self.output_types[i]:
                        dem = tensor_to_np(output[0])
                        dem = np_rgb(dem, cmap='viridis')
                        row.append(np_to_tensor(dem).to(device))
                    else:
                        row.append(torch.stack(list(output[:o])+[output[0]]*(3-o)))
                    output = output[o:]
            for i, t in enumerate(output_channels):
                if 'land' in self.output_types[i] and not self.rgb_landcover:
                    land = tensor_to_np(target[0])
                    if self.spread_landcover:
                        land = spread_to_continuous(land)
                    land = gray_to_land(land, old)
                    row.append(np_to_tensor(land).to(device))
                elif "dem" in self.output_types[i]:
                    dem = tensor_to_np(target[0])
                    dem = np_rgb(dem, cmap='viridis')
                    row.append(np_to_tensor(dem).to(device))
                else:
                    row.append(torch.stack(list(target[:t])+[target[0]]*(3-t)))
                target = target[t:]
            # for i, t in enumerate(output_channels):
            #     if 'land' in self.output_types[i]:
            #         land = tensor_to_np(processed[:t])
            #         if self.spread_landcover:
            #             land = spread_to_continuous(land)
            #         land = gray_to_land(land, old)
            #         row.append(np_to_tensor(land).to(device))
            #     else:
            #         row.append(torch.stack(list(processed[:t])+[processed[0]]*(3-t)))
            #     processed = processed[t:]
            grid.append(torch.cat(row, dim=2))
        return torch.cat(grid, dim=1)

    def open_image_array(self, name: str) -> np.ndarray | None:
        '''
        Open an image array from disk
        Args:
            name (str): The name of the image
        Returns:
            np.ndarray: The image array
        '''
        return open_image_array(os.path.join(self.data_dir, name))


def spread_to_continuous(spread: np.ndarray) -> np.ndarray:
    """
    Convert an n channel image to a continuous image with uniformly distributed classes.
    One channel should not count as a class (water, empty, etc.)
    """
    h, w, c = spread.shape
    step = 255 // (c - 1)
    # TODO: Use posterior instead of argmax
    maxes = np.argmax(spread, axis=2)
    return maxes * step

def continuous_to_spread(continuous: np.ndarray, classes: int) -> np.ndarray:
    """
    Convert a continuous image to an n channel image with uniformly distributed classes.
    The provided number of classes should not include the empty class
    """
    h, w = continuous.shape
    step = 255 // classes
    to_return = np.zeros((h, w, classes+1), dtype=np.uint8)
    for i in range(classes+1):
        to_return[:, :, i] = (continuous == i * step) * 255
    return to_return


def tensor_to_np(x: torch.Tensor):
    if isinstance(x, np.ndarray) and x.dtype == np.uint8:
        return x
    x = ((x.cpu().detach().numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8)
    if len(x.shape) == 3 and x.shape[0] < x.shape[1]:
        x = x.transpose(1, 2, 0)
    return x

def np_to_tensor(x: np.ndarray):
    if len(x.shape) == 3:
        x = x.transpose(2, 0, 1)
    return torch.from_numpy(x).float().div(127.5).sub(1)


def land_loss(raw_outputs: torch.tensor, raw_targets: torch.tensor, reduction: str='mean', classes: int=10) -> torch.tensor:
    """
    Calculate the class based loss for landcover images.
    It is the average of correct class predictions.
    """
    half_c = (classes-1)/2
    raw_outputs = (raw_outputs * half_c) + half_c
    raw_outputs = raw_outputs.round()
    raw_outputs = (raw_outputs - half_c) / half_c

    incorrect = (raw_outputs != raw_targets).float()
    loss = torch.mean(incorrect, dim=0)
    if reduction == 'mean':
        return torch.mean(loss)
    elif reduction == 'none':
        return loss
    elif reduction == 'sum':
        return torch.sum(loss)
    else:
        raise ValueError(f"Invalid reduction {reduction}")



@timing
def open_image_array(path: str, tile_width: int=None) -> Union[np.ndarray, None]:
    '''
    Open an image as a numpy array
    Args:
        path (str): The path to the image
        tile_width (int): The width of the tile. Setting this will return zeros instead of None.
    Returns:
        np.ndarray: The image as a numpy array
    '''
    zero_return = None if tile_width is None else np.zeros((tile_width, tile_width), dtype=np.uint8)
    if 'Sat' in path:
        zero_return = None if tile_width is None else np.zeros((tile_width, tile_width, 3), dtype=np.uint8)
    try:
        if not os.path.exists(path):
            return zero_return
        return np.array(img.open(path))
    except (UnidentifiedImageError, OSError) as error:       
        print(f"{path} corrupted. Deleting...")
        # TODO: Change setup check to false for this tile
        os.remove(path)
        return zero_return


def open_array(path: str) -> np.ndarray | None:
    if not os.path.exists(path):
        return None
    try:
        return np.load(path)
    except Exception:
        return None


def save_image_array(path: str, im: np.ndarray):
    '''
    Save an image array to disk
    Args:
        path (str): The path to save the image
        im (np.ndarray): The image to save
    '''
    img.fromarray(im).save(path)

def text_box(width: int, height: int, text: str, text_colour: int=0, padding_colour: int=0, text_box_colour: int=255, 
             vertical: bool=False, is_label: bool=True) -> Image:
    try:
        font = ImageFont.truetype('LinLibertine_R.ttf', size=int(height*0.75))
    except:
        warnings.warn('LinLibertine_R.ttf not found. Using default font.')
        font = ImageFont.load_default()
    text_box = img.new('RGB', size=(width, height), color=(text_box_colour,)*3)
    draw = ImageDraw.Draw(text_box)
    _, _, w, h = draw.textbbox((0, 0), text, font=font)
    width = w + width//5 if is_label else width
    text_box = img.new('RGB', size=(width, height), color=(text_box_colour,)*3)
    draw = ImageDraw.Draw(text_box)
    draw.rectangle([(0, 0), (width, height)], fill=(text_box_colour,)*3, outline=(padding_colour,)*3, width=height//20)
    draw.text(((width-w)/2, (height-h)/2), text, fill=text_colour, font=font)
    if vertical:
        text_box = text_box.rotate(90, expand=True)
    return text_box

def show_imgs(imgs: list[np.ndarray]):
    '''
    Show a list of images
    Args:
        imgs (list[np.ndarray]): The images to show
    '''
    for im in imgs:
        show_img(im)

def show_img(im: np.ndarray):
    '''
    Show an image
    Args:
        im (np.ndarray): The image to show
    '''
    im = im.copy()
    if im.max() <= 1:
        im *= 255
    if im.dtype == np.float32 or im.dtype == np.float64:
        im = im.astype(np.uint8)
    if len(im.shape) == 3 and im.shape[2] == 1:
        im = im[:, :, 0]
    img.fromarray(im).show()

def image_grid(imgs: list[Image], rows: int, cols: int, texts: list[str]=[], 
               padding: int=2, col_labels: list[str]=[], row_labels: list[str]=[],
               text_colour: int=0, padding_colour: int=128, text_box_colour: int=255,
               size_multiplier: int=1, text_proportion: float=0.2) -> Image:
    '''
    Create a grid of images
    Args:
        imgs (list[Image]): The images to put in the grid
        rows (int): The number of rows in the grid
        cols (int): The number of columns in the grid
        texts (list[str]): The text to put under each image
        padding (int): The padding percentage between images
        col_labels (list[str]): The labels for each column
        row_labels (list[str]): The labels for each row
        text_colour (int): The colour of the text
        padding_colour (int): The colour of the padding
        text_box_colour (int): The colour of the text box
        size_multiplier (int): The size multiplier for the images
        text_proportion (float): The proportion of the image height to use for text boxes
    Returns:
        Image: The grid of images
    '''
    w, h = imgs[0].size
    w, h = w*size_multiplier, h*size_multiplier
    imgs = [im.resize((w, h)) for im in imgs]
    s = min(w, h)
    padding = s*padding//100
    grid_w = cols*w + (cols+1)*padding
    grid_h = rows*h + (rows+1)*padding

    text_proportion = min(max(text_proportion, 0.01), 0.5)
    horizontal_box_size = (w, int(ceil(s * text_proportion)))
    vertical_box_size = (h, int(ceil(s * text_proportion)))
    if len(col_labels) == cols:
        grid_h += horizontal_box_size[1] + padding
    if len(row_labels) == rows:
        grid_w += vertical_box_size[1] + padding
    grid = img.new('RGB', size=(grid_w, grid_h), color=(padding_colour,)*3)
    
    if len(col_labels) == cols:
        for i, label in enumerate(col_labels):
            text = text_box(horizontal_box_size[0], horizontal_box_size[1], label, 
                            text_colour, padding_colour, text_box_colour, is_label=False)
            x = i*(w+padding) + padding
            y = padding
            if len(row_labels) == rows:
                x += vertical_box_size[1] + padding
            grid.paste(text, box=(x, y))
    if len(row_labels) == rows:
        for i, label in enumerate(row_labels):
            text = text_box(vertical_box_size[0], vertical_box_size[1], label, text_colour, 
                            padding_colour, text_box_colour, vertical=True, is_label=False)
            x = padding
            y = i*(h+padding) + padding
            if len(col_labels) == cols:
                y += horizontal_box_size[1] + padding
            grid.paste(text, box=(x, y))

    for i, im in enumerate(imgs):
        if len(texts) == len(imgs):
            lw, lh = horizontal_box_size
            text = text_box(lw, lh, texts[i], text_colour, padding_colour, text_box_colour, is_label=True)
            im.paste(text, box=(0, h - lh))
        x = (i%cols)*(w+padding) + padding
        y = (i//cols)*(h+padding) + padding
        if len(col_labels) == cols:
            y += horizontal_box_size[1] + padding
        if len(row_labels) == rows:
            x += vertical_box_size[1] + padding

        grid.paste(im, box=(x, y))
    return grid

def plt_grid(imgs, rows, cols, texts=[]):
    f, axarr = plt.subplots(rows, cols)
    for i, img in enumerate(imgs):
        if len(texts) == len(imgs):
            axarr[i//cols, i%cols].set_title(texts[i])
        axarr[i//cols, i%cols].imshow(img)
    plt.show()


def call_depth():
    '''
    Return the call depth of a function.
    '''
    return len(inspect.stack(0))


@lru_cache(maxsize=100)
def get_brush_deltas(brush_size: int, center: tuple=(0.5, 0.5)) -> list:
    '''
    Get pixels in circle around center with radius brush_size
    Args:
        brush_size (int): Radius of circle
        center (tuple): Center of circle
    Returns:
        list: List of pixels in circle
    '''
    # TODO: consider changing this to always use distance from 0,0 but use brush sizes in increments of 0.5
    if brush_size == 0:
        return [(0, 0)]
    brush_deltas = []
    ibrush_size = int(brush_size+1)
    cy, cx = center
    for dx in range(-ibrush_size, ibrush_size+1):
        for dy in range(-ibrush_size, ibrush_size+1):
            # Distance from center at 0.5 0.5 instead of 0 0 to draw more natural small circles
            if (dx-cx)*(dx-cx) + (dy-cy)*(dy-cy) <= brush_size*brush_size:
                brush_deltas.append((dy, dx))
    return brush_deltas


@lru_cache(maxsize=100)
def get_brush_kernel(brush_size: int, center: tuple=(0.5, 0.5)) -> np.ndarray:
    '''
    Get a kernel for a brush with a given size and center
    Args:
        brush_size (int): The size of the brush
        center (tuple): The center of the brush
    Returns:
        np.ndarray: The kernel for the brush
    '''
    brush_deltas = get_brush_deltas(brush_size, center)
    kernel = np.zeros((2*brush_size+1, 2*brush_size+1), dtype=np.uint8)
    for dy, dx in brush_deltas:
        kernel[dy+brush_size, dx+brush_size] = 255
    return kernel

def brush_mask(mask: np.ndarray, brush_size: int, center: tuple=(0.5, 0.5), wrap_x: bool=True) -> np.ndarray:
    '''
    Get the mask with with a brish applied to it.
    Args:
        mask (np.ndarray): The boolean mask to apply the brush to
        brush_size (int): The size of the brush to use
        center (tuple, optional): The center of the brush. Defaults to (0.5, 0.5).
        wrap_x (bool, optional): Whether to wrap the brush around the x axis. Defaults to True.
    Returns:
        np.ndarray: The mask with the brush applied
    '''
    assert mask.dtype == bool, "Mask must be boolean"
    shift_mask = np.zeros_like(mask, dtype=bool)
    for dx, dy in get_brush_deltas(brush_size, center):
        roll = np.roll(mask, (dy, dx), axis=(0, 1))
        if dy > 0:
            roll[:dy,:] = False
        elif dy < 0:
            roll[dy:,:] = False
        if not wrap_x:
            if dx > 0:
                roll[:,:dx] = False
            elif dx < 0:
                roll[:,dx:] = False
        shift_mask |= roll
    return shift_mask

def test_brushes(max_brush_size=10):
    mx_brush = max_brush_size

    sample = np.zeros((2*mx_brush**2, 2*mx_brush**2))
    for bs in range(1, mx_brush):
        x, y = bs*mx_brush*2, bs*mx_brush*2
        bds = get_brush_deltas(bs)
        for dx, dy in bds:
            if x+dx < 0 or x+dx >= 2*mx_brush**2 or y+dy < 0 or y+dy >= 2*mx_brush**2:
                continue
            sample[x+dx, y+dy] = 255
    if not os.path.exists(os.path.join(data_dir, f"brushes")):
        os.mkdir(os.path.join(data_dir, f"brushes"))
    img.fromarray(sample).convert('RGB').show()

def np_rgba(np_arr: np.ndarray, cmap: str, max_pixel: int=255, min_pixel: int=0) -> np.ndarray:
    '''
    Convert a grayscale image to RGBA using a matplotlib colormap
    Args:
        np_arr (np.ndarray): The grayscale image
        cmap (str): The name of the matplotlib colormap to use
        max_pixel (int): The maximum pixel value in the image to use for normalisation
        min_pixel (int): The minimum pixel value in the image to use for normalisation
    Returns:
        np.ndarray: The RGBA image
    '''
    max_value = max_pixel

    if max_value < np_arr.max():
        max_value = np_arr.max()

    normalised = (np_arr.astype(np.float32)/max_value*(255-min_pixel)+min_pixel) / 255
    mapper = plt.get_cmap(cmap)
    rgba = mapper(normalised)
    rgba = rgba * 255
    rgba = rgba.astype(np.uint8)
    # rgba[np_arr == 0] = [21, 59, 106, 255] #153b6a
    return rgba

def np_rgb(np_arr: np.ndarray, cmap: str='gray', max_pixel: int=255, min_pixel: int=0) -> np.ndarray:
    '''
    Convert a grayscale image to RGB using a matplotlib colormap
    Args:
        np_arr (np.ndarray): The grayscale image
        cmap (str): The name of the matplotlib colormap to use
        max_pixel (int): The maximum pixel value in the image to use for normalisation
        min_pixel (int): The minimum pixel value in the image to use for normalisation
    Returns:
        np.ndarray: The RGB image
    '''
    rgba = np_rgba(np_arr, cmap, max_pixel, min_pixel)
    return rgba[:,:,:3]

def all_paths_exist(paths: list) -> bool:
    '''
    Check if all paths in a list exist
    Args:
        paths (list): List of paths to check
    Returns:
        bool: True if all paths exist, False otherwise
    '''
    for path in paths:
        if not os.path.exists(path):
            return False
    return True

def fix_colours(im: np.ndarray, colours: int) -> np.ndarray:
    '''
    Fix incorrect colours in a given sketch, setting the colours to the nearest colour in the palette
    Args:
        im (np.ndarray): The sketch to fix
        colours (int): The number of colours in the palette
    Returns:
        np.ndarray: The fixed sketch
    '''
    if type(colours) == list:
        colour_list = colours
        colour_size = colours[1]
    else:
        colour_size = 256 // colours
        colour_list = [i*colour_size-1 for i in range(0, colours+1)]
    dists = np.array([np.abs(im-col) for col in colour_list])
    to_return = np.argmin(dists, axis=0)*colour_size
    to_return[to_return > 0] -= 1
    to_return = to_return.astype(np.uint8)
    assert to_return.max() <= colour_list[-1]
    return to_return
        
def overlap(im0: np.ndarray, im1: np.ndarray) -> bool:
    '''
    Get the overlap between two images
    Args:
        im0 (np.ndarray): The first image
        im1 (np.ndarray): The second image
    Returns:
        bool: True if the images overlap, False otherwise
    '''
    mask0 = im0 > 0
    mask1 = im1 > 0
    return np.any(mask0 & mask1)

def create_mask(mask: np.ndarray, horizontal_width: float=0.25, vertical_width: float=0.25, 
                seed: int=None, extra_sides_p: float=0.5) -> np.ndarray:
    '''
    Create a random mask.
    The mask is a random segment of the image set to True.
    The context occurs in stripes at the edges of the image.
    Args:
        im (np.ndarray): The image to mask
        horizontal_width (float): The width of the horizontal stripes as a fraction of the image width
        vertical_width (float): The width of the vertical stripes as a fraction of the image height
        seed (int): The seed to use for the random number generator
        extra_sides_p (float): The probability of adding extra sides to the mask
    Returns:
        np.ndarray: The mask
    '''
    h, w = mask.shape
    mask = np.ones_like(mask, dtype=np.uint8) * 255
    horizontal_width = int(horizontal_width * w)
    vertical_width = int(vertical_width * h)
    contexts = [1, 2, 4, 8]

    while len(contexts) > 0:
        context = random.choice(contexts)
        contexts.remove(context)
        if context & 1:
            mask[:, :horizontal_width] = 0
        if context & 2:
            mask[:, -horizontal_width:] = 0
        if context & 4:
            mask[:vertical_width, :] = 0
        if context & 8:
            mask[-vertical_width:, :] = 0
        if random.random() > extra_sides_p:
            break
    return mask

def create_overlap_mask(shape: tuple, extra_sides_p: float=0.25) -> np.ndarray:
    """
    Create a mask with random tiles overlapping it.
    This is to improve next step inpainting for diffinfinite sampling
    Args:
        shape (tuple): The shape of the mask
        extra_sides_p (float): The probability of adding extra sides to the mask
    Returns:
        np.ndarray: The mask
    """
    to_return = np.ones(shape, dtype=np.uint8)
    h, w = shape
    while True:
        temp = np.ones(shape, dtype=np.uint8)
        # Internal point. 1 is chosen to avoid the mask being completely empty on first iteration
        x = random.randint(1, w)
        y = random.randint(1, h)
        # L/R
        dx = random.randint(0, 1) 
        dy = random.randint(0, 1)
        (x0, x1) = (0, x) if dx == 0 else (x, w)
        (y0, y1) = (0, y) if dy == 0 else (y, h)
        temp[y0:y1, x0:x1] = 0
        # Don't allow temp to be completely empty
        if np.sum(temp) == 0:
            if to_return.min() == 1:
                continue # We haven't added a single mask yet
            break
        to_return &= temp
        if random.random() > extra_sides_p:
            break
    return to_return * 255




def masking(im: torch.tensor, generator: torch.Generator=None) -> torch.tensor:
    '''
    Randomly set part of an image to gaussian noise.
    This image should be in the range [-1, 1].
    The context occurs in stripes at the edges of the image.
    The last channel of im is the mask.
    Args:
        im (torch.tensor): The image to mask
        generator (torch.Generator): The random number generator
    Returns:
        torch.tensor: The mask + image with a random segment set to noise
    '''
    assert im.max() <= 1.0, "Image must be in the range [0, 1]"
    assert im.min() >= -1.0, "Image must be in the range [0, 1]"
    h, w = im.shape[1:3]
    mask = im[-1]
    h, w, H, W = get_bounds(mask.cpu().numpy())
    to_return = im.clone()
    mask_where = mask > 0
    for i in range(im.shape[0]-1):
        noise = randn_tensor(to_return[i].shape, device=im.device, generator=generator)
        to_return[i][mask_where] = noise[mask_where]
    return to_return


def random_masking(im: torch.tensor, mode: str='blending', mask_width: float=0.5, idx: int=None) -> torch.tensor:
    '''
    Randomly set part of an image to 0.
    This image should be in the range [-1, 1].
    The masking can be a vertical stripe(left, right, middle), horizontal stripe(left, right, middle), or a square(top/bottom left/right and center).
    Args:
        im (torch.tensor): The image to mask
        mode (str): The type of masking to use. Choices: (blending, generation, both).
        mask_width (float): The width of the mask as a fraction of the image width/height
        idx (int): The index of the image in the batch
    Returns:
        torch.tensor: The image with a random segment set to noise
    '''
    if mode == 'blending':
        excluded = [
            0, 2, 3, 5, 7, 8, 9, 10
        ]
    elif mode == 'generation':
        excluded = [
            1, 4, 6
        ]
    else:
        excluded = []
    # assert im.max() <= 1.0, "Image must be in the range [0, 1]"
    # assert im.min() >= -1.0, "Image must be in the range [0, 1]"
    included = [i for i in range(11) if i not in excluded]
    mode = random.choice(included) if idx is None else included[idx % len(included)]
    '''
    Generation: 0, 2, 3, 5, 7, 8, 9, 10
    Blending: 1, 4, 6
    Vertical
    0           1           2
    @ @ # #     # @ @ #     # # @ @
    @ @ # #     # @ @ #     # # @ @
    @ @ # #     # @ @ #     # # @ @
    @ @ # #     # @ @ #     # # @ @

    Horizontal
    3           4           5
    @ @ @ @     # # # #     # # # #
    @ @ @ @     @ @ @ @     # # # #
    # # # #     @ @ @ @     @ @ @ @
    # # # #     # # # #     @ @ @ @

    Square
    6           7           8           9           10
    # # # #     @ @ # #     # # @ @     # # # #     # # # #
    # @ @ #     @ @ # #     # # @ @     # # # #     # # # #
    # @ @ #     # # # #     # # # #     @ @ # #     # # @ @
    # # # #     # # # #     # # # #     @ @ # #     # # @ @
    '''
    h, w = im.shape[:2]
    to_remove = int(mask_width * (h if mode < 3 else w))
    sides = (w//2-to_remove)//2
    to_return = im
    # TODO: Find out if the strips should be random or fixed width
    if mode < 3:
        y_start = 0
        y_end = h
        start_sides = mode * sides
        end_sides = (mode - 2) * sides
        x_start = mode * w // 4 + start_sides
        x_end = (mode + 2) * w // 4 + end_sides
    elif mode < 6:
        start_sides = (mode - 3) * sides
        end_sides = (mode - 5) * sides
        y_start = (mode - 3) * h // 4 + start_sides
        y_end = (mode - 1) * h // 4 + end_sides
        x_start = 0
        x_end = w
    elif mode == 6:
        y_start = h // 4 + sides
        y_end = 3 * h // 4 - sides
        x_start = w // 4 + sides
        x_end = 3 * w // 4 - sides
    else:
        y = (mode - 7) // 2
        x = (mode - 7) % 2
        start_y_sides = y * sides * 2
        end_y_sides = (y - 1) * sides * 2
        start_x_sides = x * sides * 2
        end_x_sides = (x - 1) * sides * 2
        y_start = y * h // 2 + start_y_sides
        y_end = (y + 1) * h // 2 + end_y_sides
        x_start = x * w // 2 + start_x_sides
        x_end = (x + 1) * w // 2 + end_x_sides
    to_return[y_start:y_end, x_start:x_end] = -1.0
    # torch.randn_like(to_return[y_start:y_end, x_start:x_end],
    #                                                            device=to_return.device)
    return to_return

def _set_tile(im: np.ndarray, tile: np.ndarray, y: int, x: int):
    '''
    Add the given tile to the image at the given pixel coordinates
    Wrap around x if necessary but cut off y if it goes out of bounds
    Args:
        im: The image to add the tile to
        tile: The tile to add to the image
        y: The y coordinate of the top left corner of the tile
        x: The x coordinate of the top left corner of the tile
    '''
    # print("im\ttile\ty\tx")
    # print(im.shape, tile.shape, y, x, sep='\t')
    if len(im.shape) > 2:
        h, w, c = im.shape
    else:
        h, w = im.shape
    if y < 0:
        tile = tile[-y:]
        y = 0
    if y + tile.shape[0] > h:
        tile = tile[:h-y]
    while x < 0:
        x += w
    x = x % w
    if x + tile.shape[1] > w:
        im[y:y+tile.shape[0], x:] = tile[:, :w-x]
        rem = tile.shape[1] - w + x
        im[y:y+tile.shape[0], :rem] = tile[:, w-x:]
    else:
        im[y:y+tile.shape[0], x:x+tile.shape[1]] = tile


def _get_tile(im: np.ndarray, y: int, x: int, width: int = 256, edge_pad: bool = True) -> np.ndarray:
    '''
    Get a square tile from the image with the top left corner at y, x
    We allow wrapping around the sides of the image but not the top or bottom

    Note: This does not return a copy of the tile, it returns a view of the original image
    Args:
        im: The image to get the tile from
        y: The y coordinate of the top left corner of the tile
        x: The x coordinate of the top left corner of the tile
        width: The width of the tile
        edge_pad: Whether to pad the edges of the image with the tile if the tile goes out of bounds
    Returns:
        A square tile from the image
    '''
    if len(im.shape) > 2:
        h, w, c = im.shape
    else:
        h, w = im.shape
    # assert y >= 0 and y + width <= h
    pad_mode = 'edge' if edge_pad else 'constant'
    if y < 0:
        padding = ((-y, 0), (0, 0), (0, 0)) if len(im.shape) > 2 else ((-y, 0), (0, 0))
        im = np.pad(im, padding, mode=pad_mode)
        y = 0
    if y + width > h:
        padding = ((0, y+width-h), (0, 0), (0, 0)) if len(im.shape) > 2 else ((0, y+width-h), (0, 0))
        im = np.pad(im, padding, mode=pad_mode)
    while x < 0:
        x += w
    if x + width > w:
        return np.concatenate((im[y:y+width, x:], im[y:y+width, :width+x-w]), axis=1)
    return im[y:y+width, x:x+width]

def get_bounds(mask: np.ndarray) -> tuple:
    '''
    Get the bounding box for a mask
    Args:
        mask: The mask to get the bounding box for
    Returns:
        The bounding box for the mask (ymin, xmin, ymax, xmax)
    '''
    # https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
    if np.sum(mask) == 0:
        return 0, 0, 0, 0
    ymin, ymax = np.where(np.any(mask, axis=1))[0][[0, -1]]
    xmin, xmax = np.where(np.any(mask, axis=0))[0][[0, -1]]

    return ymin, xmin, ymax, xmax

# TODO Speedup with numba
@njit(cache=True, parallel=True)
def fast_modal_resize(im: np.ndarray, dx: int, exclude_zeros: bool = True) -> np.ndarray:
    '''
    Resize using a modal filter by some dx.
    Note this will not work with pytorch workers
    Args:
        im: The image to resize
        dx: The amount to resize by (positive is smaller)
        exclude_zeros: Whether to exclude zeros from the modal filter
    '''
    if dx <= 1:
        return im
    h, w = im.shape
    output = np.zeros((h//dx, w//dx), dtype=im.dtype)
    for y in prange(0, h//dx):
        for x in prange(0, w//dx):
            snapshot = im[y*dx:y*dx+dx, x*dx:x*dx+dx].flatten()
            if exclude_zeros:
                snapshot = snapshot[snapshot != 0]
            if len(snapshot) == 0:
                value = 0
            else:
                value = np.argmax(np.bincount(snapshot))
            value = 0
            
            output[y, x] = value
    return output

@profile
def slow_modal_resize(im: np.ndarray, dx: int, exclude_zeros: bool = True) -> np.ndarray:
    """
    Resize using a modal filter by some dx.
    Args:
        im: The image to resize
        dx: The amount to resize by (positive is smaller)
        exclude_zeros: Whether to exclude zeros from the modal filter
    
    Returns:
        np.ndarray: The resized image
    """

    # TODO Explore using reshape and apply along axis to speed up

    if dx <= 1:
        return im
    h, w = im.shape
    output = np.zeros((h//dx, w//dx), dtype=im.dtype)
    for y in range(0, h//dx):
        for x in range(0, w//dx):
            snapshot = im[y*dx:y*dx+dx, x*dx:x*dx+dx].flatten()
            bins = np.bincount(snapshot)
            if exclude_zeros and len(bins) > 1:
                bins[0] = 0
            value = np.argmax(bins)
            
            output[y, x] = value
    return output

@profile
def modal_resize(im: np.ndarray, dx: int, exclude_zeros: bool = True, 
                 included_values: np.ndarray = None, is_hex: bool = False, use_fast: bool = False) -> np.ndarray:
    '''
    Resize using a modal filter by some dx
    Args:
        im: The image to resize
        dx: The amount to resize by (positive is smaller)
        exclude_zeros: Whether to exclude zeros from the modal filter
        included_values: The values to include in the modal filter (if None, all values are included)
        is_hex: Whether the image is in hex format
    '''
    if dx <= 1:
        return im
    if dx <= 2:
        return cv2.resize(im, (im.shape[1]//dx, im.shape[0]//dx), interpolation=cv2.INTER_NEAREST)

    if is_hex:
        im = np_rgb_to_hex(im)
        im = modal_resize(im, dx, exclude_zeros, included_values, is_hex=False)
        return np_hex_to_rgb(im)

    if dx >= 16 or exclude_zeros or included_values is not None:
        if use_fast:
            return fast_modal_resize(im, dx, exclude_zeros)
        else:
            return slow_modal_resize(im, dx, exclude_zeros)
    im = modal(im, rectangle(dx, dx), shift_x=-dx//2, shift_y=-dx//2)
    im = im[::dx, ::dx]
    return im


def np_hex_to_rgb(hex: np.ndarray) -> np.ndarray:
    r = hex // 256**2
    g = (hex % 256**2) // 256
    b = hex % 256
    return np.dstack([r, g, b]).astype(np.uint8)


def np_rgb_to_hex(rgb: np.ndarray) -> np.ndarray:
    to_return = np.zeros_like(rgb[:, :, 0], dtype=np.uint32)
    to_return = rgb[:, :, 0]*256**2 + rgb[:, :, 1]*256 + rgb[:, :, 2]
    return to_return


def hex_to_rgb(hex: int) -> tuple[int, int, int]:
    r = hex // 256**2
    g = (hex % 256**2) // 256
    b = hex % 256
    return (r, g, b)


def rgb_to_hex(rgb: tuple[int, int, int]) -> int:
    r, g, b = rgb
    return r*256**2 + g*256 + b


def hex_is_water(hex: int) -> bool:
    return rgb_is_water(hex_to_rgb(hex))


def rgb_is_water(rgb: tuple[int, int, int]) -> bool:
    return rgb[0] < 10 and rgb[1] < 20 and rgb[2] < 35


def rgb_to_yuv(r: np.ndarray | int, g: np.ndarray | int, b: np.ndarray | int) -> tuple[np.ndarray | int]:
    r, g, b = np.array([r, g, b]) / 255
    y = 0.299*r + 0.587*g + 0.114*b
    u = -0.147*r - 0.289*g + 0.436*b
    v = 0.615*r - 0.515*g - 0.100*b
    return y, u, v


def hex_str_to_rgb(hex_string: str):
    return tuple(int(hex_string[i:i+2], 16) for i in (1, 3, 5))


def rgb_to_hex_str(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


ArrOrInt = Union[np.ndarray, int]


def redmean_distance(r1: ArrOrInt, g1: ArrOrInt, b1: ArrOrInt, 
                     r2: ArrOrInt, g2: ArrOrInt, b2: ArrOrInt) -> ArrOrInt:
    r_mean = (r1 + r2) / 2
    dr = (r1 - r2)**2
    dg = (g1 - g2)**2
    db = (b1 - b2)**2
    return np.sqrt((2 + r_mean/256) * dr + 4 * dg + (2 + (255 - r_mean)/256) * db)


@timing
def get_data_image(
    data_dir: str,
    shape: tuple[int, int],
    name: str,
    default_shape: tuple[int, int] = (8192, 16384),
    interpolation: int = cv2.INTER_LANCZOS4,
    custom_resizer: Callable[[np.ndarray, tuple[int, int]], np.ndarray] = None,
    generator_func: Callable[(...), np.ndarray] | None = None,
    generator_args: list = [],
    expiration_date: str = "28/08/2025",
) -> np.ndarray:
    h, w = shape
    H, W = default_shape
    _, extension = os.path.splitext(name)
    exp_datetime = datetime.strptime(expiration_date, "%d/%m/%Y")
    wanted_name = name.replace("WxH", f"{w}x{h}").replace(extension, '.npy')
    default_name = name.replace("WxH", f"{W}x{H}")
    wanted_path = os.path.join(data_dir, ".cache", wanted_name)
    file_last_modified = os.path.getmtime(wanted_path) if os.path.exists(wanted_path) else 0
    default_path = os.path.join(data_dir, default_name)
    im = open_array(wanted_path)
    os.makedirs(os.path.join(data_dir, ".cache"), exist_ok=True)
    if im is None:
        print(f"Could not find cached image {os.path.abspath(wanted_path)}")
        im = open_image_array(default_path)
    elif datetime.fromtimestamp(file_last_modified) < exp_datetime:
        print(f"Cached image {os.path.abspath(wanted_path)} has expired, regenerating")
        im = open_image_array(default_path)
    else:
        return im
    if im is None:
        print(f"Could not find default image {os.path.abspath(default_path)}, generating new one")
        im = generator_func(*generator_args)
        img.fromarray(im).save(default_path)
        return get_data_image(
            data_dir, shape, name, default_shape, interpolation, custom_resizer
        )
    if im.shape[:2] != (h, w):
        if custom_resizer is not None:
            im = custom_resizer(im, shape)
        else:
            im = cv2.resize(im, (w, h), interpolation=interpolation)
    np.save(wanted_path, im)

    return im


def draw_histogram(
    im: np.ndarray,
    num_bins: int = 256,
):
    '''
    Draw a histogram of the image
    Args:
        im (np.ndarray): The image to draw the histogram of
        num_bins (int): The number of bins to use in the histogram
    '''
    if len(im.shape) == 3:
        im = im[:, :, 0]
    im = im.flatten()
    im = im[im > 0]
    hist, bins = np.histogram(im.flatten(), bins=num_bins, range=(0, 256))
    plt.plot(bins[:-1], hist)
    plt.xlim([0, 256])
    plt.show()


def clip(x: int, lower: int, upper: int):
    return min(max(x, lower), upper)
