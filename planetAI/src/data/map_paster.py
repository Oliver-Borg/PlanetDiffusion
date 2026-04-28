try:
    import data_creator
    import sketch_gen
    from utils import PlanetConfig, open_image_array, all_paths_exist, _set_tile, _get_tile, modal_resize, create_mask, profile
    from dataclass_argparser import CustomArgumentParser
except:
    from . import data_creator
    from . import sketch_gen
    from .utils import PlanetConfig, open_image_array, all_paths_exist, _set_tile, _get_tile, modal_resize, create_mask, profile
    from .dataclass_argparser import CustomArgumentParser
from tqdm import tqdm
import os
import cv2
import numpy as np
import random
from PIL import Image as img
from PIL.Image import Image
import argparse
from math import log2
import re
from typing import Union, Optional, Tuple
import warnings
from dataclasses import dataclass, field, replace
import psutil
import json
from copy import deepcopy

sizes = [
    (512, 256),
    (1024, 512),
    (2048, 1024),
    (4096, 2048),
    (8192, 4096),
    (16384, 8192),
]


def save_mask_tiles(mask: np.ndarray, mask_dir: str, img_ext: str, i: int, ref: bool, ang: int, existing: list):
    '''
    Save 256x256 tiles of the given mask
    Args:
        mask: The mask to save the tiles of
        mask_dir: The directory to save the tiles to
        img_ext: The image extension to use
        i: The mask number
        ref: Whether the mask is reflected
        ang: The angle the mask is rotated by
        existing: A list of existing tiles
    '''
    h, w = mask.shape[:2]
    for y in range(h-256, -256, -256):
        for x in range(w-256, -256, -256):
            tile_name = f'{i}_{ref}_{ang}/{y}_{x}.{img_ext}'
            if tile_name in existing:
                # If last tile exists we can return because all preceding tiles have been created already
                return 
            tile = get_tile(mask, y, x, width=512)
            tile = tile[:256, :]
            if tile.max() == 0:
                continue
            tile_dir = os.path.join(mask_dir, f'{i}_{ref}_{ang}')
            os.makedirs(tile_dir, exist_ok=True)
            img.fromarray(tile).save(os.path.join(mask_dir, tile_name))

def setup_json(folder: str) -> dict:
    '''
    Get the json containing the already generated tiles
    Args:
        folder: The directory to get the json from
    Returns:
        A dictionary of the generated tiles
    '''
    
    # For each mask type we want a json file with the following columns:
    # masknum_ref_ang.ext: mask_saved, tiles_saved
    # masknum_ref_ang.ext is the name of the mask
    # mask_saved is whether the mask has been saved
    # tiles_saved is whether the tiles have been saved
    # The json file should be named _setup.json
    json_dir = os.path.join(folder, '_setup.json')
    if os.path.exists(json_dir):
        with open(json_dir, 'r') as f:
            return json.load(f)
    return {}

def save_json(folder: str, json_dict: dict):
    '''
    Save the given json dictionary to the given folder
    Args:
        folder: The folder to save the json to
        json_dict: The dictionary to save
    '''
    json_dir = os.path.join(folder, '_setup.json')
    with open(json_dir, 'w') as f:
        json.dump(json_dict, f)


def transform(mask: np.ndarray, trans: Tuple[int, int], ref: bool, ang: int, fcenter: Tuple[int, int]) -> np.ndarray:
    '''
    Transform the given mask by the given parameters
    Args:
        mask: 
    '''
    if trans[0] or trans[1]:
        mask = data_creator.numpy_translate(mask, trans)
    if ref:
        mask = data_creator.numpy_reflect(mask, int(fcenter[0]))
    if ang:
        mask = data_creator.cv2_rotate(mask, ang, fcenter)
    return mask



# @profile
def setup(planet_cfg: PlanetConfig, output_dir='dataset', force=False, show_overlaps=False):
    '''
    Setup the dem, river and sketch masks for a given planet config
    Args:
        planet_cfg: The planet config to use
        output_dir: The directory that the dataset will be output to. 
        Setup just places the generation list here and prepares the folders.
        force: Whether to force the setup to run again
        show_overlaps: Whether to show the overlaps between the dems (DEPRECATED)
    '''
    # We generate transformations at a higher resolution first to prevent aliasing issues

    if planet_cfg.downscale_cfg is not None:
        setup(planet_cfg.downscale_cfg, output_dir, force)
    size = planet_cfg.size
    offset = planet_cfg.offset
    mask_count = planet_cfg.mask_count
    operations = planet_cfg.operations
    max_size = min(size+offset, len(sizes)-1)
    dx = 2**(max_size - size)
    w, h = sizes[size]
    W, H = sizes[max_size]
    folder = str(planet_cfg) 
    to_generate = []
    train_dir = os.path.join(output_dir, f'{folder}/{w}x{h}', 'train')
    test_dir = os.path.join(output_dir, f'{folder}/{w}x{h}', 'test')
    valid_dir = os.path.join(output_dir, f'{folder}/{w}x{h}', 'valid')
    # Create directories for output
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)

    planet_cfg.make_dirs()

    dem_tiles = setup_json(planet_cfg.dem_dir())
    sat_tiles = setup_json(planet_cfg.sat_dir())
    land_tiles = setup_json(planet_cfg.land_dir())

    existing_dem = os.listdir(planet_cfg.dem_dir())
    existing_sat = os.listdir(planet_cfg.sat_dir())
    existing_land = os.listdir(planet_cfg.land_dir())
    total = 0
    done = 0
    for i in range(mask_count):
        for ref in operations['ref'][i]:
            for ang in operations['ang'][i]:
                img_ext = f'{i}_{ref}_{ang}.{planet_cfg.image_extension}'
                dem_tiles[img_ext] = dem_tiles.get(img_ext, [False, False])
                sat_tiles[img_ext] = sat_tiles.get(img_ext, [False, False])
                land_tiles[img_ext] = land_tiles.get(img_ext, [False, False])

                dem_done = dem_tiles[img_ext][0] if not planet_cfg.fast_tiles else dem_tiles[img_ext][1]
                sat_done = sat_tiles[img_ext][0] if not planet_cfg.fast_tiles else sat_tiles[img_ext][1]
                land_done = land_tiles[img_ext][0] if not planet_cfg.fast_tiles else land_tiles[img_ext][1]

                if force:
                    dem_done = False
                    sat_done = False
                    land_done = False

                if dem_done and sat_done and land_done and not force:
                    done += 1
                total += 1
    if done == total:
        return

    # if not force and planet_cfg.check_setup():
    #     if planet_cfg.planet_seed is not None and planet_cfg.gen_lists:
    #         gen_list(planet_cfg=planet_cfg, tiles=True)
    #     return

    brush_size = planet_cfg.brush_size
    colours = planet_cfg.colours
    min_size = planet_cfg.min_size
    threshold = planet_cfg.threshold
    river_size = planet_cfg.river_size
    seed = planet_cfg.planet_seed
    data_dir = planet_cfg.data_dir
    iters = planet_cfg.iters
    top_n_orders = planet_cfg.top_n_orders
    use_colour_map = planet_cfg.use_colour_map
    translations = planet_cfg.translations
    sketch_gen.create_masks(brush_size, [sizes[size], sizes[max_size], sizes[-1]], force=force, data_dir=data_dir)
    sketch_gen.extract_binary_masks(mask_count, brush_size, [sizes[size], sizes[max_size], sizes[-1]], data_dir=data_dir)
    folder = planet_cfg.dem_str()
    base_dir = os.path.join(data_dir, f"base/{folder}/{W}x{H}")
    sat_base_dir = os.path.join(data_dir, f"base/{planet_cfg.sat_str()}/{W}x{H}")
    land_base_dir = os.path.join(data_dir, f"base/{planet_cfg.land_str()}/{W}x{H}")
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(sat_base_dir, exist_ok=True)
    os.makedirs(land_base_dir, exist_ok=True)
    mask_dir = os.path.join(data_dir, f"masks/{W}x{H}/brush_{brush_size}")
    json_dir = os.path.join(data_dir, f"json/{W}x{H}/brush_{brush_size}")
    masks, poss, _ = data_creator.get_masks(mask_dir=mask_dir, num=mask_count, json_dir=json_dir)
    # Get origins of biggest masks to have consistent rotations for all mask sizes
    mask_dir = os.path.join(data_dir, f"masks/16384x8192/brush_{brush_size}")
    json_dir = os.path.join(data_dir, f"json/16384x8192/brush_{brush_size}")
    _, _, origins = data_creator.get_masks(mask_dir=mask_dir, num=mask_count, json_dir=json_dir)

    origins = np.array(origins).astype(np.float32)
    origins /= 8192
    origins *= H
    
    oceanmask = None
    dem = None
    sat = None
    land = None
    with tqdm(total=len(masks), desc="Creating base masks...") as pbar:
        for i, mask in enumerate(masks):
            base_img_dir = os.path.join(base_dir, f'{i}.png')
            base_sat_img_dir = os.path.join(sat_base_dir, f'{i}.png')
            base_land_img_dir = os.path.join(land_base_dir, f'{i}.png') 
            if all_paths_exist([base_img_dir, base_sat_img_dir, base_land_img_dir]) and not force:
                pbar.update(1)
                continue
            if os.path.exists(base_img_dir) and not force:
                base_mask = np.array(img.open(base_img_dir))
            else:
                if dem is None:
                    oceanmask = np.array(img.open(os.path.join(data_dir, f"world.oceanmask.{W}x{H}.png")))
                    dem = np.array(img.open(os.path.join(data_dir, f"World_DEM_{W}x{H}.png")))

                    base = np.zeros((H, 2*W), dtype=np.uint8)
                    base[:, :W] = 255
                    base[:, W:] = dem
                full_mask = np.zeros_like(base, dtype=bool)
                full_mask[:, :W][mask > 0] = True
                full_mask[:, W:][mask > 0] = True
                full_mask[:, W:][oceanmask < 30] = False
                base_mask = base.copy()
                base_mask[~full_mask] = 0
                img.fromarray(base_mask).save(base_img_dir)
            if not os.path.exists(base_sat_img_dir) or force:
                if sat is None:
                    sat = np.array(img.open(os.path.join(data_dir, f"world.satellite.{W}x{H}.png")))
                sat_mask = np.dstack((base_mask, base_mask, base_mask))
                sat_mask[:, W:] = sat
                sat_mask[:, W:][np.all(sat_mask[:, :W]==0, axis=2)] = [0, 0, 0]
                img.fromarray(sat_mask).save(base_sat_img_dir)
            if not os.path.exists(base_land_img_dir) or force:
                if land is None:
                    land = np.array(img.open(os.path.join(data_dir, f"World_LandCover_{W}x{H}.png")))
                land_mask = base_mask.copy()
                land_mask[:, W:] = land
                land_mask[:, W:][base_mask[:, :W]==0] = 0
                img.fromarray(land_mask).save(base_land_img_dir)
            pbar.update(1)

    planet_cfg.make_dirs()

    # Set these to None to prevent opening them if they're not needed
    watermask = None
    oceanmask = None

    brush_deltas = sketch_gen.get_brush_deltas(brush_size)
    overlaps = []

    total_count = 0
    for i in range(mask_count):
        total_count += len(operations['ref'][i]) * len(operations['ang'][i])
    

    with tqdm(total=total_count) as pbar:
        for i in range(mask_count):
            # Set base to None to prevent opening it if it's not needed
            base = None
            sat_base = None
            land_base = None
            trans = translations[i]
            cnt = len(operations['ref'][i]) * len(operations['ang'][i])
            # x, y center
            fcenter = tuple(origins[i] + np.array(trans[::-1])*H)
            # masknum_ref_ang.png
            pbar.set_description(f"Creating transformations (mask {i+1}/{mask_count})")
            for ref in operations['ref'][i]:
                for ang in operations['ang'][i]:
                    img_ext = f'{i}_{ref}_{ang}.{planet_cfg.image_extension}'
                    dem_dir = os.path.join(planet_cfg.dem_dir(), img_ext)
                    land_dir = os.path.join(planet_cfg.land_dir(), img_ext)
                    sat_dir = os.path.join(planet_cfg.sat_dir(), img_ext)
                    
                    dem_tiles[img_ext] = dem_tiles.get(img_ext, [False, False])
                    sat_tiles[img_ext] = sat_tiles.get(img_ext, [False, False])
                    land_tiles[img_ext] = land_tiles.get(img_ext, [False, False])

                    dem_done = dem_tiles[img_ext][0] if not planet_cfg.fast_tiles else dem_tiles[img_ext][1]
                    sat_done = sat_tiles[img_ext][0] if not planet_cfg.fast_tiles else sat_tiles[img_ext][1]
                    land_done = land_tiles[img_ext][0] if not planet_cfg.fast_tiles else land_tiles[img_ext][1]

                    if force:
                        dem_done = False
                        sat_done = False
                        land_done = False

                    if dem_done and sat_done and land_done and not force:
                        pbar.update(1)
                        continue

                    if not dem_done:
                        dem_dir = os.path.join(planet_cfg.dem_dir(), img_ext)
                        dem = None
                        if not planet_cfg.fast_tiles and os.path.exists(dem_dir):
                            dem = open_image_array(dem_dir)
                        else:
                            if base is None:
                                base = np.array(img.open(os.path.join(base_dir, f'{i}.png')))
                            dem = base[:, W:]
                            dem = transform(dem, trans, ref, ang, fcenter)
                            dem = np.array(img.fromarray(dem).convert("L").resize((w, h), resample=img.Resampling.LANCZOS))
                        if planet_cfg.fast_tiles:
                            save_mask_tiles(dem, planet_cfg.dem_dir(), planet_cfg.image_extension, i, ref, ang, existing_dem)
                            dem_tiles[img_ext][1] = True
                        else:
                            img.fromarray(dem).save(dem_dir)
                            dem_tiles[img_ext][0] = True
                        save_json(planet_cfg.dem_dir(), dem_tiles)

                    if not land_done:
                        land = None
                        land_dir = os.path.join(planet_cfg.land_dir(), img_ext)
                        if not force and not planet_cfg.fast_tiles and os.path.exists(land_dir):
                            land = open_image_array(land_dir)
                        else:
                            if land_base is None:
                                land_base = np.array(img.open(os.path.join(land_base_dir, f'{i}.png')))
                            land = land_base[:, W:]
                            land = transform(land, trans, ref, ang, fcenter)
                            
                            if dx > 1:
                                land = modal_resize(land, dx) # This is quite slow for offset > 0
                        if planet_cfg.fast_tiles:
                            save_mask_tiles(land, planet_cfg.land_dir(), planet_cfg.image_extension, i, ref, ang, existing_land)
                            land_tiles[img_ext][1] = True
                        else:
                            img.fromarray(land).save(land_dir)
                            land_tiles[img_ext][0] = True
                        save_json(planet_cfg.land_dir(), land_tiles)

                    
                    if not sat_done:
                        sat = None
                        if not force and not planet_cfg.fast_tiles and os.path.exists(sat_dir):
                            sat = open_image_array(sat_dir)
                        else:
                            if sat_base is None:
                                sat_base = np.array(img.open(os.path.join(sat_base_dir, f'{i}.png')))
                            sat = sat_base[:, W:]
                            sat = transform(sat, trans, ref, ang, fcenter)
                            sat = np.array(img.fromarray(sat).resize((w, h), resample=img.Resampling.LANCZOS))
                        if planet_cfg.fast_tiles:
                            save_mask_tiles(sat, planet_cfg.sat_dir(), planet_cfg.image_extension, i, ref, ang, existing_sat)
                            sat_tiles[img_ext][1] = True
                        else:
                            img.fromarray(sat).save(sat_dir)
                            sat_tiles[img_ext][0] = True
                        save_json(planet_cfg.sat_dir(), sat_tiles)
                    
                    pbar.update(1)
    
    if planet_cfg.planet_seed is not None and planet_cfg.gen_lists:
        gen_list(planet_cfg=planet_cfg, tiles=True)

def set_tile(im: np.ndarray, tile: np.ndarray, y: int, x: int):
    '''
    Add the given tile to the image at the given pixel coordinates
    Wrap around x if necessary but cut off y if it goes out of bounds
    Args:
        im: The image to add the tile to
        tile: The tile to add to the image
        y: The y coordinate of the top left corner of the tile
        x: The x coordinate of the top left corner of the tile
    '''
    _set_tile(im, tile, y, x)


def get_tile(im: np.ndarray, y: int, x: int, width: int = 256, edge_pad: bool = True):
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
    return _get_tile(im, y, x, width, edge_pad)

def efficient_tiles(tile_list: list) -> list:
    '''
    Convert from a generation list to a tile list
    [(k0, y0, x0),...(kn, yn, xn)] -> [(k0, [(y0, x0), (y1, x1)]),...(kn, [(yn, xn)])]
    Args:
        tile_list: The list of tiles to convert
    Returns:
        A list of tiles
    '''
    tile_list = sorted(tile_list, key=lambda x: x[0])
    to_return = [(tile_list[0][0], [])]
    for i, (k, y, x) in enumerate(tile_list):
        if k != to_return[-1][0]:
            to_return.append((k, []))
        to_return[-1][1].append((y, x))
    return to_return

def get_combinations(planet_cfg: PlanetConfig) -> int:
    '''
    Get the number of combinations that can be generated
    '''
    combinations = 1
    total_imgs = 0
    operations = planet_cfg.operations
    for i in range(planet_cfg.mask_count):
        # This technically isn't correct but allows the probability of removals to be 50%
        mask_i_combs = len(operations['ang'][i]) * len(operations['ref'][i]) * len(operations['rem'][i])
        total_imgs += mask_i_combs - 1
        combinations *= mask_i_combs

    combinations*=2 # Vertical Flip
    combinations*=2 # Horizontal Flip
    num_spins = 256//planet_cfg.spin
    combinations*=num_spins # Spinning TODO: Improve the spin for downscaling since it messes up alignment
    # print(f"Total Combinations: {combinations}")
    return combinations

# @timing
# @profile
def get_next_tile_pair(seed: int, planet_cfg: PlanetConfig, tile_y: int=None) -> Tuple[np.ndarray, np.ndarray, int, int, int]:
    '''
    Get the next tile pair for a given seed. 
    This seed should be changed each step such as a dataloader idx
    Args:
        seed: The seed to use
        planet_cfg: The planet config to use
        tile_y (optional): The tile y coordinate
    Returns:
        A tuple of the next tile pair
    '''
    h_tiles = 2**(planet_cfg.size+1)
    v_tiles = 2**planet_cfg.size if tile_y is None else 1
    combinations = get_combinations(planet_cfg)*h_tiles*v_tiles
    coverage = 0.75
    tries = 0
    valid_tiles = planet_cfg.valid_tiles()
    valid_coords = planet_cfg.valid_coords
    discard = planet_cfg.collision_mode == 'discard'
    th = planet_cfg.discard_threshold
    # Sometimes creates overlapping sat images
    while True and tries < 100:
        tries += 1
        if planet_cfg.size < 0: # This seems to be faster so I will just use this for now
            r = random.Random(seed).randint(0, combinations-1)
            seed = r
            x = r % h_tiles
            r //= h_tiles
            y = r % v_tiles if tile_y is None else tile_y
            r //= v_tiles
            k = r
            rx = x*256
            ry = y*256
            if not (y*256, x*256) in valid_coords:
                continue
        else:
            tile = random.Random(seed).choice(valid_tiles)
            y, x, mask, ref, ang = tile
            # This gives an empty config with a random horizontal flip, vertical flip and spin
            cfg = get_mask_config(0, planet_cfg)
            cfg['hflip'] = random.Random(seed).randint(0, 1)
            cfg['vflip'] = random.Random(seed).randint(0, 1)
            # Quick and dirty fix. I'd like to figure out how to do this properly
            x += random.Random(seed).randint(0, 256//planet_cfg.spin-1)*planet_cfg.spin 
            vflip = cfg['vflip']
            hflip = cfg['hflip']
            # spin = cfg['spin']
            tile_width = 256
            ry, rx = get_translated_coords(y, x, vflip, hflip, 0, planet_cfg.size, tile_width)
            rry, rrx = get_translated_coords(ry, rx, vflip, hflip, 0, planet_cfg.size, tile_width)
            assert rry == y and rrx == x, f"Translated coordinates are not the same as the original coordinates: {rry, rrx} != {y, x}"
            cfg['mask_refs'][mask] = ref
            cfg['mask_angs'][mask] = ang
            cfg['mask_rems'][mask] = False
            k = get_k_val(cfg, planet_cfg)
            # sketch, dem = gen_tile_pair(k, ry, rx, planet_cfg)
            # return sketch, dem, k, y, x
        dem = _gen_image(None, get_mask_config(k, planet_cfg), 'dem', planet_cfg, y=ry, x=rx)
        if data_creator.check_empty(dem, coverage):
            continue
        if discard:
            overlap_tile = gen_overlap_tile(k, ry, rx, planet_cfg)
            if np.sum(overlap_tile == 128)/overlap_tile.size > th/(256*512):
                continue
        sketch, dem = gen_tile_pair(k, ry, rx, planet_cfg)
        return sketch, dem, k, ry, rx
    print("Failed to generate tile pair: randomizing parameters")
    # TODO Maybe randomize
    planet_cfg.randomize_parameters()
    sketch, dem = gen_tile_pair(k, 256*y, 256*x, planet_cfg)
    return sketch, dem, k, y, x
    
    


def gen_list(planet_cfg: PlanetConfig, tiles: bool=False, pairs: Union[dict, None]=None) -> list:
    '''
    Get a random list of combinations to generate making sure to remove empty combinations
    Args:
        planet_cfg: The planet config to use
        tiles: Whether to generate tiles or not
        pairs: A dictionary of the generated pairs
    Returns:
        A list of tuples of combinations to generate (k, y, x) if tiles is true
        A list of combinations to generate (k) if tiles is false
    '''
    iters = planet_cfg.iters
    size = planet_cfg.size
    mask_count = planet_cfg.mask_count
    seed = planet_cfg.planet_seed
    operations = planet_cfg.operations
    num_removals = lambda x: get_mask_config(x, planet_cfg)['mask_rems'].count(True)
    to_generate = []
    combinations = get_combinations(planet_cfg)
    h, w = 256*2**size, 512*2**size

    h_tiles = h//256
    w_tiles = w//256

    discarded = 0
    created = 0

    gen_txt_dir = planet_cfg.gen_txt_dir()

    if tiles and seed is not None:
        if os.path.exists(gen_txt_dir) and not planet_cfg.force_gen_list:
            with open(gen_txt_dir, 'r') as f:
                for line in f:
                    to_generate.append(tuple(map(int, line.split())))
                if len(to_generate) >= iters:
                    return to_generate[:iters]
                created = len(to_generate)
    with tqdm(total=iters-created, desc='Generating List (Discarded 0 images 0.00%)') as pbar:
        created = 0
        while len(to_generate) < iters:
            r = random.Random(seed).randint(0, combinations-1)
            # We do this since otherwise it will repeat the same seed
            if seed is not None: 
                seed = r
            if r in to_generate or num_removals(r) == mask_count:
                continue
            pair = None
            discard = planet_cfg.collision_mode == 'discard'
            th = planet_cfg.discard_threshold
            if discard:
                overlap_mask = gen_overlap(r, planet_cfg)
            if tiles:
                coverage = 0.75
                for y in range(h_tiles):
                    for x in range(w_tiles):
                        if len(to_generate) == iters:
                            break 
                        if discard:
                            overlap_tile = get_tile(overlap_mask, y*256, x*256)
                            if np.sum(overlap_tile == 128)/overlap_tile.size > th/(256*512):
                                discarded += 1
                                pbar.set_description(f'Generating List (Discarded {discarded} images {discarded/(created+discarded):.2%})')
                                continue
                        if pair is None:
                            pair = gen_image(r, planet_cfg)
                            dem = pair[:, w:, :]
                        if data_creator.check_empty(get_tile(dem, y*256, x*256), coverage):
                                discarded += 1
                                pbar.set_description(f'Generating List (Discarded {discarded} images {discarded/(created+discarded):.2%})')
                                continue
                        to_generate.append((r, y, x))
                        created += 1
                        pbar.update(1)
                        if pairs is not None:
                            pairs[r] = pair
            else:
                if discard:
                    if np.sum(overlap_mask == 128)/overlap_mask.size > th/(256*512):
                        discarded += 1
                        pbar.set_description(f'Generating List (Discarded {discarded} images {discarded/(created+discarded):.2%})')
                        continue
                to_generate.append(r)
                created += 1
                pbar.update(1)

    if tiles and seed is not None:
        with open(gen_txt_dir, 'w') as f:
            for k, y, x in to_generate:
                f.write(f'{k} {y} {x}\n')
    return to_generate

def combine_arrays(arr0: np.ndarray, arr1: np.ndarray, mode: str='avg') -> np.ndarray:
    '''
    Combine two arrays using a given mode to deal with overlapping values
    Args:
        arr0: The first array to combine
        arr1: The second array to combine
        mode: The mode to use to combine the arrays (avg, min, max, rep)
    
    Returns:
        The combined array
    '''
    assert arr0.shape == arr1.shape
    to_return = arr0.copy()
    if len(arr0.shape) == 2:
        overlap = np.logical_and(arr0 > 0, arr1 > 0)
    else:
        overlap = np.logical_and(np.any(arr0 > 0, axis=2), np.any(arr1 > 0, axis=2))
        overlap = np.dstack([overlap]*arr0.shape[2])
    assert overlap.shape == arr0.shape

    if mode == 'avg':
        to_return[overlap] = (arr0[overlap].astype(np.uint16) + arr1[overlap].astype(np.uint16))/2
    elif mode == 'min':
        to_return[overlap] = np.minimum(arr0[overlap], arr1[overlap])
    elif mode == 'max':
        to_return[overlap] = np.maximum(arr0[overlap], arr1[overlap])
    elif mode == 'rep':
        to_return[overlap] = arr1[overlap]
    
    to_return[~overlap] += arr1[~overlap]

    return to_return

def gen_mask(mask_name: str, im_type: str, planet_cfg: PlanetConfig, y: int=None, x: int=None, tile_width: int=256) -> np.ndarray:
    '''
    Gen the mask on the fly for the given mask name and type
    Args:
        mask_name: The name of the mask to generate
        im_type: The type of image to generate the mask for
        planet_cfg: The planet config to use
        y (optional): The pixel y coordinate of the top left corner of the tile
        x (optional): The pixel x coordinate of the top left corner of the tile
        tile_width (optional): The width of the tile to generate
    Returns:
        The generated mask
    '''
    tile_mode = y is not None and x is not None
    dem = planet_cfg.get_mask(mask_name, 'dem')
    if dem is None:
        dem = np.zeros((256*2**planet_cfg.size, 512*2**planet_cfg.size), dtype=np.uint8)
    if tile_mode:
        # TODO I shouldn't actually do this here since the rivers and sketches are no longer global
        # It's necessary for speed though so I'll leave it for now
        # Local strahler orders are kind of more representative anyway
        # Bucketing mode is also not supported for tiles
        dem = get_tile(dem, y, x, tile_width) 
    if im_type == 'river':
        landmask = (dem > 0).astype(np.uint8)*255
        condition = sketch_gen.get_strahler_orders(dem, landmask, landmask, planet_cfg)
    elif im_type == 'sketch':
        if planet_cfg.sketch_mode == "brush":
            condition = sketch_gen.paint_img(dem, planet_cfg)
        else:
            condition = sketch_gen.dilate_paint(dem, planet_cfg)
    elif im_type == 'sat_sketch':
        sat = planet_cfg.get_mask(mask_name, 'sat')
        if tile_mode:
            sat = get_tile(sat, y, x, tile_width)
        condition = sketch_gen.paint_satellite_image(sat, dem, planet_cfg)
    elif im_type == 'land_sketch':
        land = planet_cfg.get_mask(mask_name, 'land')
        if tile_mode:
            land = get_tile(land, y, x, tile_width)
        condition = sketch_gen.landcover_paint(land, planet_cfg)
    elif im_type == 'land':
        land = planet_cfg.get_mask(mask_name, 'land')
        if tile_mode:
            land = get_tile(land, y, x, tile_width)
        condition = land
    else:
        raise ValueError(f'Invalid im_type {im_type}')

    if not tile_mode:
        planet_cfg.save_mask(condition, mask_name, im_type)
    return condition

def get_translated_coords(y: int, x: int, vflip: bool, hflip: bool, spin: int, size: int, tile_width: int):
    '''
    Get the coordinates of the translated image
    Args:
        y: The y coordinate of the top left corner of the image
        x: The x coordinate of the top left corner of the image
        vflip: Whether the image is vertically flipped
        hflip: Whether the image is horizontally flipped
        spin: The amount to spin the image
    Returns:
        The translated coordinates
    '''
    h = 256*2**size
    w = 512*2**size
    if spin and hflip:
        x = (x + spin) % w
    elif spin:
        x = (x - spin) % w
    if vflip:
        y = h-y-tile_width
    if hflip:
        x = w-x-tile_width
    return y%h, x%w

@profile
def _gen_full_image(im_dict: dict, im_conf: dict, im_type: str, planet_cfg: PlanetConfig,
                    combine_mode: str='max') -> np.ndarray:
    '''
    Generate an image from a given config
    Args:
        im_dict: The dictionary of images to use
        im_conf: The config to use
        im_type: The type of image to generate[sketch, dem, river, overlap]
        planet_cfg: The planet config to use
        combine_mode: The mode to use to combine the images[avg, max, rep]
    Returns:
        The generated image
    '''
    w = 512*2**planet_cfg.size
    h = 256*2**planet_cfg.size
    to_return = None
    for y in range(0, h, 256):
        for x in range(0, w, 256):
            tile = _gen_image(im_dict, im_conf, im_type, planet_cfg, combine_mode, y, x)
            if to_return is None:
                to_return = np.zeros((h, w), dtype=np.uint8)
                if len(tile.shape) > 2:
                    to_return = np.dstack([to_return]*tile.shape[2])
            to_return[y:y+256, x:x+256] = tile
    return to_return

def _gen_image(im_dict: dict, im_conf: dict, im_type: str, planet_cfg: PlanetConfig, 
               combine_mode: str='max', y: int=None, x: int=None, tile_width: int=256, 
               use_surrounds: bool=False) -> np.ndarray:
    '''
    Generate an image from a given config
    Args:
        im_dict: The dictionary of images to use
        im_conf: The config to use
        im_type: The type of image to generate[sketch, dem, river, overlap]
        planet_cfg: The planet config to use
        combine_mode: The mode to use to combine the images[avg, max, rep]
        y (optional): The pixel y coordinate of the top left corner of the tile
        x (optional): The pixel x coordinate of the top left corner of the tile
        tile_width (optional): The width of the tile to generate
        use_surrounds (optional): Whether to use the surrounds when generating the image
    Returns:
        The generated image
    '''
    if im_type == 'mask':
        to_return = np.zeros((tile_width, tile_width), dtype=np.uint8) 
        to_return = create_mask(to_return, planet_cfg.inpainting_width, planet_cfg.inpainting_width)
        return to_return
    
    if im_type == 'none':
        return np.zeros((tile_width, tile_width), dtype=np.uint8)

    if 'down' in im_type:
        down_conf = deepcopy(im_conf)
        downscale_factor = 2**planet_cfg.downscale_offset
        down_conf['spin'] //= downscale_factor
        tw = tile_width//downscale_factor
        top = y//downscale_factor if y is not None else None
        left = x//downscale_factor if x is not None else None
        if use_surrounds and top is not None and left is not None:
            top -= tw
            left -= tw
            tw *= 3
        down_img = _gen_image(im_dict, down_conf, im_type[4:], planet_cfg.downscale_cfg, 
                          combine_mode, top, left, tw, False)
        return cv2.resize(down_img, (tile_width, tile_width), interpolation=cv2.INTER_NEAREST)
    
    mask_count = planet_cfg.mask_count
    use_colour_map = planet_cfg.use_colour_map
    mask_refs = im_conf['mask_refs']
    mask_angs = im_conf['mask_angs']
    mask_rems = im_conf['mask_rems']
    name = im_conf['name']
    hflip = im_conf['hflip']
    vflip = im_conf['vflip']
    spin = im_conf['spin']
    name_mod = ""
    xleft = x
    ytop = y
    if y is not None and x is not None:
        y, x = get_translated_coords(y, x, vflip, hflip, spin, planet_cfg.size, tile_width)
        xleft = x // 256 * 256
        ytop = y // 256 * 256
        spin = 0
        name_mod = f'/{ytop}_{xleft}'
    extra_gen_kwargs = {}
    # if planet_cfg.fast_tiles:
    #     extra_gen_kwargs = {'tile_width': tile_width, 'y': y, 'x': x}

    mask_names = [f'{i}_{mask_refs[i]}_{mask_angs[i]}{name_mod}.{planet_cfg.image_extension}' for i in range(mask_count)
                if mask_rems[i] == False]
    if not planet_cfg.fast_tiles or im_type in ['dem', 'overlap', 'sat', 'land']:
        ims = [planet_cfg.get_mask(
            mask_name, 'dem' if im_type == 'overlap' else im_type, **extra_gen_kwargs
            ) for mask_name in mask_names]
    else:
        ims = [None for _ in mask_names]

    
    wanted = len(ims)
    have = 0
    for i in range(wanted):
        if ims[i] is None and im_type not in ['dem', 'overlap', 'sat', 'land']:
            ims[i] = gen_mask(mask_names[i], 
                              'dem' if im_type == 'overlap' else im_type, 
                              planet_cfg, **extra_gen_kwargs)
            have += 1
        elif ims[i] is not None:
            have += 1
    ims = [im for im in ims if im is not None]
    if x is not None: # Get proper sized tile from 512x256 image
        ims = [get_tile(im, y-ytop, x-xleft, tile_width) for im in ims]
    # if y is not None and x is not None and not planet_cfg.fast_tiles:
    #     ims = [get_tile(im, y, x, tile_width) for im in ims]
    if im_type == 'river':
        ims = [sketch_gen.process_strahler_rivers(im, planet_cfg) for im in ims]
    # if have == 0:
    #     # TODO Remove this warning as it happens frequently due to spin on wide tiles
    #     warnings.warn(f'{have} {"dem" if im_type == "overlap" else im_type} masks found for y: {ytop}, x: {xleft}')

    sea_level = planet_cfg.sea_level
    rescale = planet_cfg.sea_level_rescale
    if (im_type == 'dem' or im_type == 'overlap') and sea_level != 0:
        ims = [sketch_gen.change_sea_level(im, sea_level, rescale) for im in ims]
    if ims == []:
        return np.zeros((tile_width, tile_width), dtype=np.uint8)
    if im_type == 'overlap':
        # TODO: Optimise
        stacked_masks = np.dstack(ims)
        # Check for overlap (masks different non-zero values)
        maxs = np.max(stacked_masks, axis=2)
        stacked_masks[stacked_masks == 0] = 255
        mins = np.min(stacked_masks, axis=2)
        overlap_mask = np.zeros(ims[0].shape[:2], np.uint8)
        overlap_mask[maxs > 0] = 255
        overlap_mask[mins != maxs] = 128
        overlap_mask[mins == 255] = 0
        # overlap_mask = np.zeros(ims[0].shape[:2], np.uint8)
        # for im in ims:
        #     overlap_mask += im.astype(bool)
        # overlap_mask[overlap_mask > 1] = 128
        # overlap_mask[overlap_mask == 1] = 255
        im = overlap_mask
    
    else:
        im = ims[0].copy()
        for i in ims[1:]:
            im = combine_arrays(i, im, combine_mode)

    if spin:
        im = np.roll(im, spin, axis=1)

    if hflip:
        im = np.flip(im, 1)
    
    if vflip:
        im = np.flip(im, 0) 
    
    return im

def gen_dem(dem_conf: dict, dem_dict: dict, planet_cfg: PlanetConfig) -> np.ndarray:
    '''
    Generate the dem for the given k value
    Args:
        dem_conf: The config to use
        dem_dict: The dictionary of dems to use
        planet_cfg: The planet config to use
    Returns:
        The generated dem
    '''
    return _gen_image(dem_dict, dem_conf, 'dem', planet_cfg)

def gen_river(river_conf: dict, river_dict: dict, planet_cfg: PlanetConfig) -> np.ndarray:
    '''
    Generate the river for the given k value
    Args:
        river_conf: The config to use
        river_dict: The dictionary of rivers to use
        planet_cfg: The planet config to use
    Returns:
        The generated river
    '''
    return _gen_image(river_dict, river_conf, 'river', planet_cfg)

def gen_sketch(sketch_conf: dict, sketch_dict: dict, planet_cfg: PlanetConfig) -> np.ndarray:
    '''
    Generate the sketch for the given k value
    Args:
        k: The k value to generate the sketch for
        sketch_dict: The dictionary of sketches to use
        planet_cfg: The planet config to use
    Returns:
        The generated sketch
    '''
    return _gen_image(sketch_dict, sketch_conf, 'sketch', planet_cfg, 'max')

def gen_sat(sat_conf: dict, sat_dict: dict, planet_cfg: PlanetConfig) -> np.ndarray:
    '''
    Generate the satellite image for the given config
    Args:
        sat_conf: The config to use
        planet_cfg: The planet config to use
        sat_dict: The dictionary of satellite images to use
    Returns:
        The generated satellite mask
    '''
    return _gen_image(sat_dict, sat_conf, 'sat', planet_cfg, 'avg')

def gen_sat_sketch(sat_sketch_conf: dict, sat_sketch_dict: dict, planet_cfg: PlanetConfig) -> np.ndarray:
    '''
    Generate the satellite sketch image for the given config
    Args:
        sat_sketch_conf: The config to use
        planet_cfg: The planet config to use
        sat_sketch_dict: The dictionary of satellite sketch images to use
    Returns:
        The generated satellite sketch mask
    '''
    return _gen_image(sat_sketch_dict, sat_sketch_conf, 'sat_sketch', planet_cfg, 'avg')

def gen_overlap(k: int, planet_cfg: PlanetConfig) -> np.ndarray:
    '''
    Generate the overlap for the given config
    Args:
        dem_conf: The config to use
        dem_dict: The dictionary of dems to use
        planet_cfg: The planet config to use
    Returns:
        The generated overlap mask
    '''
    dem_dict, _, _, _, _ = planet_cfg.get_mask_dicts()
    dem_conf = get_mask_config(k, planet_cfg)
    return _gen_image(dem_dict, dem_conf, 'overlap', planet_cfg, 'avg')


def gen_target(k: int, planet_cfg: PlanetConfig) -> np.ndarray:
    '''
    Just generate the target for a given k value
    Args:
        k: The k value to generate the target for
        planet_cfg: The planet config to use
    Returns:
        The generated target
    '''
    # ['sketch-to-dem', 'sketch-inpainting', 'dem-inpainting', 'sketch-upscaling', 
    #  'none', 'noise', 'normal-noise', 'satellite', 'dem-to-satellite']
    dem_dict, sat_dict = {}, {}
    conf = get_mask_config(k, planet_cfg)
    if planet_cfg.image_mode in ['satellite', 'dem-to-satellite']:
        if not planet_cfg.use_mask_store:
            sat_dict = planet_cfg.get_mask_dict(planet_cfg.sat_dir())
            planet_cfg.sat_dict = sat_dict
        to_return = gen_sat(conf, sat_dict, planet_cfg)[:, :, np.newaxis]
    else:
        if not planet_cfg.use_mask_store:
            dem_dict = planet_cfg.get_mask_dict(planet_cfg.dem_dir())
            planet_cfg.dem_dict = dem_dict
        dem = gen_dem(conf, dem_dict, planet_cfg)
        to_return = dem[:, :, np.newaxis]
    if planet_cfg.image_mode == 'sketch-inpainting':
        to_return = gen_full_sketch(dem, k, planet_cfg)
    return to_return

def gen_full_sketch(dem: np.ndarray, k: int, planet_cfg: PlanetConfig) -> np.ndarray:
    '''
    Generate the full sketch for the given k value
    Args:
        dem: The dem to generate the full sketch for
        k: The k value to generate the full sketch for
        planet_cfg: The planet config to use
    Returns:
        The generated full sketch
    '''
    if len(dem.shape) == 3:
        dem = dem.copy()[:, :, 0]
    landmask = (dem > 0).astype(np.uint8)*255
    rivers = sketch_gen.strahler_paint_river(dem, landmask, landmask, planet_cfg) 
    if planet_cfg.sketch_mode == "brush":
        sketch = sketch_gen.paint_img(dem, planet_cfg)
    else:
        sketch = sketch_gen.dilate_paint(dem, planet_cfg)
    if planet_cfg.use_parent_dem:
        downdem = gen_downdem(dem)
        to_return = np.dstack([sketch, rivers, downdem])
    else:
        to_return = np.dstack([sketch, rivers])
    return to_return

def gen_condition(target: np.ndarray, k: int, planet_cfg: PlanetConfig) -> np.ndarray:
    '''
    Generate the condition on the fly without needing to do setup
    DEMs must still be setup
    Args:
        target: The target image to generate the condition for
        k: The k value to generate the condition for
        planet_cfg: The planet config to use
    Returns:
        The generated condition
    '''
    # ['sketch-to-dem', 'sketch-inpainting', 'dem-inpainting', 'sketch-upscaling', 
    #  'none', 'noise', 'normal-noise', 'satellite', 'dem-to-satellite']
    if len(target.shape) == 3:
        target = target.copy()[:, :, 0]
    if planet_cfg.image_mode == 'satellite': # TODO fix
        to_return = sketch_gen.paint_satellite_image(target, target, planet_cfg)
    elif planet_cfg.image_mode in ['dem-to-satellite']:
        dem_dict = {}
        conf = get_mask_config(k, planet_cfg)
        if not planet_cfg.use_mask_store:
            dem_dict = planet_cfg.get_mask_dict(planet_cfg.dem_dir())
            planet_cfg.dem_dict = dem_dict
        to_return = gen_dem(conf, dem_dict, planet_cfg)[:, :, np.newaxis]
    elif planet_cfg.image_mode in ['sketch-inpainting', 'sketch-to-dem']:
        to_return = gen_full_sketch(target, k, planet_cfg)
    elif planet_cfg.image_mode in ['dem-inpainting', 'sketch-upscaling']:
        if planet_cfg.use_parent_dem:
            to_return = np.dstack([target, gen_downdem(target)])
        else:
            to_return = target[:, :, np.newaxis]
    elif planet_cfg.image_mode in ['none', 'noise', 'normal-noise']:
        to_return = np.zeros_like(target)
        if planet_cfg.use_parent_dem:
            to_return = np.dstack([to_return, to_return, to_return])
        else:
            to_return = np.dstack([to_return, to_return])
    else:
        raise ValueError(f"Unknown image mode {planet_cfg.image_mode}")
    assert len(to_return.shape) == 3
    return to_return

def gen_downdem(dem: np.ndarray) -> np.ndarray:
    if len(dem.shape) == 3:
        dem = dem.copy()[:, :, 0]
    to_return = np.zeros_like(dem)
    h, w = dem.shape
    to_return[:h//2, :w//2] = cv2.resize(dem, (w//2, h//2), interpolation=cv2.INTER_LANCZOS4)
    return to_return


def gen_tile_pair(k: int, y: int, x: int, planet_cfg: PlanetConfig) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Generate the image tiles for the given k, y, x values
    Args:
        k: The k value to generate the image for
        y: The y value to generate the image for
        x: The x value to generate the image for
        planet_cfg: The planet config to use
    Returns:
        A tuple containing the target image and the conditional image
    '''
    im_conf = get_mask_config(k, planet_cfg)
    inputs = [_gen_image({}, im_conf, inp, planet_cfg, 'max', y, x, 
                         use_surrounds=planet_cfg.use_surrounds and 'sketch' in inp
                         ) for inp in planet_cfg.input_types]
    outputs = [_gen_image({}, im_conf, out, planet_cfg, 'max', y, x) for out in planet_cfg.output_types]

    if 'land_sketch' in planet_cfg.clean_input_types() and 'sketch' in planet_cfg.clean_input_types():
        # Clean the edges of the DEM
        land_sketch = inputs[planet_cfg.clean_input_types().index('land_sketch')]
        sketch = inputs[planet_cfg.clean_input_types().index('sketch')]
        # Check if there are spots where sketch == 0 and land_sketch > 0
        missing = np.logical_and(sketch == 0, land_sketch > 0)
        if np.any(missing):
            land_sketch[missing] = 0
            inputs[planet_cfg.clean_input_types().index('land_sketch')] = land_sketch

    if planet_cfg.use_parent_dem:
        padding = planet_cfg.parent_dem_padding
        # TODO Check parent dem
        parent_dem = _gen_image({}, im_conf, 'dem', planet_cfg, 'max', y-padding//2, x-padding//2, 256+padding)
        parent_dem = cv2.resize(parent_dem, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        inputs = [parent_dem] + inputs
    if len(inputs) == 0:
        inputs.append(np.zeros((256, 256), dtype=np.uint8))
    if len(outputs) == 0:
        outputs.append(np.zeros((256, 256), dtype=np.uint8))

    for im in inputs:
        if im.shape[:2] != (256, 256):
            warnings.warn(f'Input image shape is {im.shape[:2]} not (256, 256)')
            print(im_conf)
    for im in outputs:
        if im.shape[:2] != (256, 256):
            warnings.warn(f'Output image shape is {im.shape[:2]} not (256, 256)')
            print(im_conf)

    inputs = [im if im.shape[:2] == (256, 256) else np.zeros((256, 256), dtype=np.uint8) for im in inputs]
    outputs = [im if im.shape[:2] == (256, 256) else np.zeros((256, 256), dtype=np.uint8) for im in outputs]
    return np.dstack(inputs).copy(), np.dstack(outputs).copy()

# @timing
def gen_overlap_tile(k: int, y: int, x: int, planet_cfg: PlanetConfig) -> np.ndarray:
    '''
    Generate the overlap tile for the given k, y, x values
    Args:
        k: The k value to generate the image for
        y: The y value to generate the image for
        x: The x value to generate the image for
        planet_cfg: The planet config to use
    Returns:
        The generated overlap tile
    '''
    im_conf = get_mask_config(k, planet_cfg)
    return _gen_image({}, im_conf, 'overlap', planet_cfg, 'avg', y, x)

def gen_pair(k: int, planet_cfg: PlanetConfig) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Generate the image for the given k value
    Args:
        k: The k value to generate the image for
        planet_cfg: The planet config to use
    Returns:
        A tuple containing the target image and the conditional image
    '''
    warnings.warn('gen_pair is deprecated, use gen_tile_pair instead', DeprecationWarning)
    # size = planet_cfg.size
    # w = 512*2**size
    # h = 256*2**size
    # cond_rows = []
    # target_rows = []
    # for y in range(0, h, 256):
    #     cond_row = []
    #     target_row = []
    #     for x in range(0, w, 256):
    #         target, cond = gen_tile_pair(k, y, x, planet_cfg)
    #         cond_row.append(cond)
    #         target_row.append(target)
    #     cond_rows.append(np.concatenate(cond_row, axis=1))
    #     target_rows.append(np.concatenate(target_row, axis=1))
    # cond = np.concatenate(cond_rows, axis=0)
    # target = np.concatenate(target_rows, axis=0)
    # return cond, target
    if planet_cfg.use_mask_store:
        dem_dict, sketch_dict, river_dict, sat_dict, sat_sketch_dict = {}, {}, {}, {}, {}
    else:
        dem_dict, sketch_dict, river_dict, sat_dict, sat_sketch_dict = planet_cfg.get_mask_dicts() 
    conf = get_mask_config(k, planet_cfg)
    
    if planet_cfg.image_mode == 'sketch-to-dem':
        dem = gen_dem(conf, dem_dict, planet_cfg)
        sketch = gen_sketch(conf, sketch_dict, planet_cfg)
        river = gen_river(conf, river_dict, planet_cfg)
        if planet_cfg.use_parent_dem:
            downdem = gen_downdem(dem)
            cond_image = np.dstack((sketch, river, downdem))
        else:
            cond_image = np.dstack((sketch, river))
        target_image = dem
    elif planet_cfg.image_mode == 'sketch-inpainting':
        sketch = gen_sketch(conf, sketch_dict, planet_cfg)
        river = gen_river(conf, river_dict, planet_cfg)
        if planet_cfg.use_parent_dem:
            dem = gen_dem(conf, dem_dict, planet_cfg)
            downdem = gen_downdem(dem)
            cond_image = np.dstack((sketch, river, downdem))
        else:
            cond_image = np.dstack((sketch, river))
        target_image = np.dstack((sketch, river))
    elif planet_cfg.image_mode == 'dem-inpainting':
        dem = gen_dem(conf, dem_dict, planet_cfg)
        if planet_cfg.use_parent_dem:
            downdem = gen_downdem(dem)
            cond_image = np.dstack((dem, downdem))
        else:
            cond_image = dem
        target_image = dem
    elif planet_cfg.image_mode == 'sketch-upscaling':
        if planet_cfg.use_mask_store:
            downsketch_dict, downriver_dict = {}, {}
        else:
            _, downsketch_dict, downriver_dict, _, _ = planet_cfg.downscale_cfg.get_mask_dicts()
        sketch = gen_sketch(conf, sketch_dict, planet_cfg)
        river = gen_river(conf, river_dict, planet_cfg)
        downsketch = gen_sketch(conf, downsketch_dict, planet_cfg.downscale_cfg)
        downriver = gen_river(conf, downriver_dict, planet_cfg.downscale_cfg)
        downsketch = np.pad(downsketch, ((0, downsketch.shape[0]), (0, downsketch.shape[1])), 'constant')
        downriver = np.pad(downriver, ((0, downriver.shape[0]), (0, downriver.shape[1])), 'constant')
        if planet_cfg.use_parent_dem:
            dem = gen_dem(conf, dem_dict, planet_cfg)
            downdem = gen_downdem(dem)
            cond_image = np.dstack((downsketch, downriver, downdem))
        else:
            cond_image = np.dstack((downsketch, downriver))
        target_image = np.dstack((sketch, river))
    elif planet_cfg.image_mode == 'satellite' or planet_cfg.image_mode == 'sketch-to-satellite': 
        _, _, _, sat_dict, sat_sketch_dict = planet_cfg.get_mask_dicts()
        sat = gen_sat(conf, sat_dict, planet_cfg)
        sat_sketch = gen_sat_sketch(conf, sat_sketch_dict, planet_cfg)
        cond_image = sat_sketch
        target_image = sat
    elif planet_cfg.image_mode == 'dem-to-satellite':
        dem_dict, _, _, sat_dict, _ = planet_cfg.get_mask_dicts()
        dem = gen_dem(conf, dem_dict, planet_cfg)
        sat = gen_sat(conf, sat_dict, planet_cfg)
        cond_image = dem
        target_image = sat
    else:
        # Just create a pair of an empty image and a dem for unconditional generation
        dem = gen_dem(conf, dem_dict, planet_cfg)
        cond_image = np.zeros((dem.shape[0], dem.shape[1], 3), np.uint8)
        target_image = dem
    if len(cond_image.shape) == 2:
        cond_image = np.expand_dims(cond_image, axis=2)
    if len(target_image.shape) == 2:
        target_image = np.expand_dims(target_image, axis=2)

    return cond_image.copy(), target_image.copy()
    
def change_channels(im: np.ndarray, channels: int, fillmode: str='repeat') -> np.ndarray:
    '''
    Change the number of channels of a given image array by concatenating the first channel multiple times
    Args:
        im: The image to change the number of channels of
        channels: The number of channels to change to
        fillmode: The mode to use to fill the channels[repeat, zeros]
    Returns:
        The image with the new number of channels
    '''
    if len(im.shape) == 2:
        im = np.expand_dims(im, axis=2)

    if im.shape[2] >= channels:
        return im[:, :, :channels]
    else:
        needed_channels = channels - im.shape[2]
        if fillmode == 'repeat':
            return np.concatenate([im[:, :, :1]]*needed_channels+[im], axis=2)
        elif fillmode == 'zeros':
            return np.concatenate([im] + [np.zeros_like(im[:, :, :1])]*needed_channels, axis=2)
        else:
            raise ValueError(f'Invalid fillmode {fillmode}')


def gen_image(k: int, planet_cfg: PlanetConfig, pair_channels: int=3) -> np.ndarray:
    '''
    Generate the image for the given k value
    Args:
        k: The k value to generate the image for
        planet_cfg: The planet config to use
        pair_channels: The number of channels to use for the pair
    Returns:
        The generated image
    '''
    cond_image, target_image = gen_pair(k, planet_cfg)
    cond_image = change_channels(cond_image, pair_channels)
    target_image = change_channels(target_image, pair_channels)
    return np.concatenate((cond_image, target_image), axis=1)


def get_mask_config(k: int, planet_cfg: PlanetConfig) -> dict:
    '''
    Get the mask configuration for a given k value
    Args:
        k: The k value to use
        planet_cfg: The planet config to use
    Returns:
        A dict containing the configuration
    '''
    conf = {}
    operations = planet_cfg.operations
    mask_count = planet_cfg.mask_count
    mask_refs = [False for i in range(mask_count)]
    mask_angs = [0 for i in range(mask_count)]
    mask_rems = [True for i in range(mask_count)]
    # name = f'{k}_' # Not necessary to use k value since operations identify the mask
    # Use get_k_val to get k value from mask config
    name = ""
    for i in range(mask_count):
        mask_i_combs = len(operations['ang'][i]) * len(operations['ref'][i]) * len(operations['rem'][i])
        mask_i = k % mask_i_combs
        k //= mask_i_combs
        mask_refs[i] = operations['ref'][i][mask_i % len(operations['ref'][i])]
        mask_i //= len(operations['ref'][i])
        mask_angs[i] = operations['ang'][i][mask_i % len(operations['ang'][i])]
        mask_i //= len(operations['ang'][i])
        mask_rems[i] = operations['rem'][i][mask_i % len(operations['rem'][i])]
        if mask_angs[i] == 0 and mask_refs[i] == False and mask_rems[i] == False:
            mask_rems[i] = True # Don't allow no operation for evaluation (can evaluate fairly on default atlas)
        name += f'{mask_refs[i]}_{mask_angs[i]}_{mask_rems[i]}_'

    
    conf['hflip'] = k%2
    name += f"{k%2}_"
    k//=2
    conf['vflip'] = k%2
    name += f"{k%2}"
    k//=2
    conf['name'] = name
    conf['mask_refs'] = mask_refs
    conf['mask_angs'] = mask_angs
    conf['mask_rems'] = mask_rems
    spin = planet_cfg.spin
    num_spins = 256//spin
    conf['spin'] = spin*(k%num_spins)
    k//=num_spins
    assert k == 0

    return conf

def get_k_val(conf: dict, planet_cfg: PlanetConfig) -> int:
    '''
    Get the k value for a given mask configuration
    Args:
        conf: The mask configuration to use
        planet_cfg: The planet config to use
    Returns:
        The k value for the given mask configuration
    '''
    operations = planet_cfg.operations
    mask_count = planet_cfg.mask_count
    k = 0
    k += conf['spin']//planet_cfg.spin
    k *= 2
    k += conf['vflip']
    k *= 2
    k += conf['hflip']
    for i in range(mask_count-1, -1, -1):
        mask_i = operations['rem'][i].index(conf['mask_rems'][i])
        mask_i *= len(operations['ang'][i])
        mask_i += operations['ang'][i].index(conf['mask_angs'][i])
        mask_i *= len(operations['ref'][i])
        mask_i += operations['ref'][i].index(conf['mask_refs'][i])
        k *= len(operations['rem'][i]) * len(operations['ang'][i]) * len(operations['ref'][i])
        k += mask_i
    return k

# @profile
def gen_images(planet_cfg: PlanetConfig, output_dir: str='dataset', threads=1, n=0):
    
    size = planet_cfg.size
    data_dir = planet_cfg.data_dir
    iters = planet_cfg.iters
    mask_count = planet_cfg.mask_count
    w, h = sizes[size]
    folder = str(planet_cfg)
    

    to_generate = gen_list(planet_cfg, tiles=True)
    train_dir = os.path.join(output_dir, folder, f'{w}x{h}', 'train')
    test_dir = os.path.join(output_dir, folder, f'{w}x{h}', 'test')
    valid_dir = os.path.join(output_dir, folder, f'{w}x{h}', 'valid')

    to_generate = list(set([tile for tile, y, x in to_generate]))

    total_iters = len(to_generate)

    train_iters = round(total_iters*0.8)
    test_iters =  round(total_iters*0.1)
    valid_iters = total_iters - train_iters - test_iters

    t_train_iters = round(train_iters/threads)
    t_test_iters = round(test_iters/threads)
    t_valid_iters = round(valid_iters/threads)

    t_total_iters = t_train_iters + t_test_iters + t_valid_iters

    ki_train = n * t_train_iters
    ki_test = n * t_test_iters + threads * t_train_iters
    ki_valid = n * t_valid_iters + threads * (t_train_iters + t_test_iters)

    if n == threads-1:
        rem = total_iters - t_total_iters*threads
        t_valid_iters += rem

    train_t = (train_dir, t_train_iters, ki_train)
    test_t = (test_dir, t_test_iters, ki_test)
    valid_t = (valid_dir, t_valid_iters, ki_valid)


    for d, it, ki in [train_t, test_t, valid_t]:
        print(f"Generating {it} images starting at {ki}")
        os.makedirs(os.path.join(d, '_full'), exist_ok=True)
        with tqdm(total=it, disable=(n!=0)) as pbar:
            for _ in range(it):
                k = to_generate[ki]
                ki += 1  
                name = get_mask_config(k, planet_cfg)['name']
                pair = gen_image(k, planet_cfg)
                data_creator.save_images(pair, d, name, size, full=True, planet_cfg=planet_cfg)
                pbar.update(1)
    if n==0:
        print("Master thread done. Waiting for other threads...")


@dataclass
class MapPaster:
    threads: int = field(
        default=1,
        metadata={
            "help": "Number of threads being used"
        }
    )
    n: int = field(
        default=0,
        metadata={
            "help": "Number of this thread"
        }
    )
    setup: bool = field(
        default=False,
        metadata={
            "help": "Setup the directories"
        }
    )
    output_dir: str = field(
        default='dataset',
        metadata={
            "help": "Output directory"
        }
    )
    force: bool = field(
        default=False,
        metadata={
            "help": "Force recreation of masks"
        }
    )

if __name__ == '__main__':
    # Add argparse arguments

    parser = CustomArgumentParser(
        (
            MapPaster, PlanetConfig
        ),
        description="Generate planet images" 
    )
    map_paster_cfg, planet_cfg = parser.parse_args_into_dataclasses()

    output_dir = map_paster_cfg.output_dir
    threads = map_paster_cfg.threads
    n = map_paster_cfg.n
    force = map_paster_cfg.force

    # Manually set some parameters
    # planet_cfg = replace(planet_cfg, size=1, iters=100, use_parent_dem=True, offset=0)

    if map_paster_cfg.setup:
        setup(planet_cfg=planet_cfg, output_dir=output_dir, force=force)    
    else:
        try:
            gen_images(planet_cfg=planet_cfg, output_dir=output_dir, threads=threads, n=n)
        except:
            if n==0:
                setup(planet_cfg=planet_cfg, output_dir=output_dir, force=force)
            gen_images(planet_cfg=planet_cfg, output_dir=output_dir, threads=threads, n=n)