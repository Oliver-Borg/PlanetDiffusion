import os

from PIL import Image as img
import numpy as np
import cv2
from itertools import product

from .utils import (
    open_image_array, PlanetConfig, timing,
    np_rgb_to_hex, hex_to_rgb, rgb_to_yuv, np_hex_to_rgb, redmean_distance, profile
)

from .landcover_utils import LandcoverClasses, landcover_index_map

from .modal_sketch import ModalSketch
from .sketch_gen import temperature_paint

img.MAX_IMAGE_PIXELS = 1000000000

@profile
def process_rivers(
    all_rivers: np.ndarray, orders: int, dilation_iters: list[int], landcover: np.ndarray | None = None
) -> np.ndarray:
    """
    Process the river channels to dilate them and combine them into a single image
    Args:
        `all_rivers`: The river channels
        `orders`: The number of river channels to consider
        `dilation_iters`: The number of dilations to apply to each river channel
        `landcover`: The landcover image used to process rivers
    Returns:
        The combined river channels
    """
    if landcover is not None:
        assert landcover.shape == all_rivers.shape
        empty_mask = (
            (landcover == LandcoverClasses.OPEN_WATER.gray_colour) |
            (landcover == LandcoverClasses.ANTARCTICA.gray_colour) |
            (landcover == LandcoverClasses.SNOW_AND_ICE.gray_colour) |
            (landcover == LandcoverClasses.MARS.gray_colour)
        )

        dry_mask = (
            (landcover == LandcoverClasses.GRASSLAND.gray_colour)
        )

        super_dry_mask = (
            (landcover == LandcoverClasses.BARE.gray_colour) |
            (landcover == LandcoverClasses.SHRUBLAND.gray_colour)
        )

        wet_mask = (
            (landcover == LandcoverClasses.WETLAND.gray_colour) |
            (landcover == LandcoverClasses.WATER.gray_colour) |
            (landcover == LandcoverClasses.CROPLAND.gray_colour) |
            (landcover == LandcoverClasses.BUILT_UP.gray_colour) |
            (landcover == LandcoverClasses.TREE_COVER.gray_colour)
        )

        if empty_mask.any():
            all_rivers[empty_mask] = 0
        if dry_mask.any():
            all_rivers[dry_mask & (all_rivers < all_rivers[dry_mask].max() - 2)] = 0
        if super_dry_mask.any():
            all_rivers[super_dry_mask & (all_rivers < all_rivers[super_dry_mask].max() - 1)] = 0
        if wet_mask.any():
            all_rivers[wet_mask & (all_rivers > 0)] += 1

    kernel = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    kernel = np.array(kernel, np.uint8)
    rivers = np.zeros_like(all_rivers)
    if len(dilation_iters) < orders:
        dilation_iters += [dilation_iters[-1]] * (orders - len(dilation_iters))
    for i in range(0, orders):
        order = i + 1
        river_channel = np.zeros_like(all_rivers)
        river_channel[all_rivers == order] = 1
        river_channel = cv2.dilate(river_channel, kernel, iterations=dilation_iters[i])
        rivers[river_channel == 1] = 255
    return rivers



class RiverModalSketch:
    """
    Modal river sketch for automatically colouring rivers based on landcover and temperature
    Args:
        `orders`: The number of river channels to consider
        `distance_weight`: The weight to apply to the distance between the modal colour and the satellite image
        `max_dilate`: The maximum number of dilations to apply to each river channel
    """
    @profile
    def __init__(self, planet_cfg: PlanetConfig, orders: int=4, distance_weight: int=2, max_dilate: int=4, order_weight: int=3, 
                 all_rivers: np.ndarray=None, rgb_landcover: np.ndarray=None, landcover: np.ndarray=None, 
                 temp: np.ndarray=None, sat: np.ndarray=None, force: bool=False):

        modal_sketch = ModalSketch(planet_cfg)
        modal_mapping = modal_sketch.get_tuple_mapping()

        l = planet_cfg.landcover_classes + 1
        t = planet_cfg.temp_classes + 1
        self.l = l
        self.t = t
        self.planet_cfg = planet_cfg
        self.mapping_file = f'river_mapping{planet_cfg.size}_{planet_cfg.temp_classes}_{planet_cfg.landcover_classes}_' +\
                            f'{orders}_{distance_weight}_{max_dilate}_{order_weight}.txt'
        if self.load_mapping(planet_cfg.data_dir) and not force:
            return

        output_dir = os.path.join(planet_cfg.data_dir, 'test', 'rivers', f'ords{orders}_dw{distance_weight}_md{max_dilate}_ow{order_weight}')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if all_rivers is None:
            all_rivers = open_image_array(os.path.join(planet_cfg.data_dir, f'global_river_order_new.tif')).astype(np.uint8)
        ords = orders
        dw = distance_weight
        dilate_iters = list(range(0, ords))
        dilate_iters.reverse()
        dilate_iters = list(map(lambda x: min(x, max_dilate), dilate_iters))
        rivers = process_rivers(all_rivers, ords, dilate_iters)
        
        h, w = rivers.shape

        if rgb_landcover is None:
            rgb_landcover = open_image_array(os.path.join(planet_cfg.data_dir, 'World_LandCover_RGB_512x256.png'))
            rgb_landcover = cv2.resize(rgb_landcover, (w, h), interpolation=cv2.INTER_NEAREST)
        if landcover is None:
            landcover = open_image_array(os.path.join(planet_cfg.data_dir, 'World_LandCover_512x256.png'))
            land_step = 255 // planet_cfg.landcover_classes
            landcover //= land_step
            img.fromarray((landcover/landcover.max()*255).astype(np.uint8)).save(os.path.join(output_dir, f'river_landcover.png'))
            landcover = cv2.resize(landcover, (w, h), interpolation=cv2.INTER_NEAREST)

        if temp is None:
            temp = open_image_array(os.path.join(planet_cfg.data_dir, 'World_Temp_512x256.png'))
            temp_step = 255 // planet_cfg.temp_classes
            temp[temp == 0] += 1
            temp = temperature_paint(temp, planet_cfg.downscale_cfg).astype(np.uint16)
            temp = ((temp + 1) // temp_step).astype(np.uint8)
            img.fromarray((temp/temp.max()*255).astype(np.uint8)).save(os.path.join(output_dir, f'river_temp.png'))
            temp = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
        
        if sat is None:
            sat = open_image_array(os.path.join(planet_cfg.data_dir, 'world.satellite.16384x8192.png'))
        sat_hex = np_rgb_to_hex(sat)

        mapping = {}

        top_n_colours = 100
        colour_matrix = np.zeros((l*t, top_n_colours+3, 3), dtype=np.uint8)

        river_mask = rivers == 255
        for i, (lc, tc) in enumerate(product(range(l), range(t))):
            modal_mask = (landcover == lc) & (temp == tc)
            mask = river_mask & modal_mask
            modal_colour = modal_mapping[(lc, tc)]
            rgb_landcover[modal_mask] = modal_colour
            full_lc = landcover_index_map(lc)
            temp_colour = np.array([tc / t * 255] * 3).astype(np.uint8)
            colour_matrix[i, 0] = full_lc.display_colour
            colour_matrix[i, 1] = temp_colour
            colour_matrix[i, 2] = np.array(modal_colour)
            all_counts = np.zeros((256**3), np.float32)
            all_values = np.arange(256**3)
            # Weight the counts based on strahler order
            for order in range(1, ords+1):
                values, counts = np.unique(sat_hex[mask & (all_rivers == order)], return_counts=True)
                counts = counts.astype(np.float32)
                counts /= (order**order_weight)
                all_counts[values] += counts
            values = all_values[all_counts > 0]
            counts = all_counts[all_counts > 0]
            rgb = np_hex_to_rgb(values)[0]
            r, g, b = rgb.transpose((1, 0))
            modal_r, modal_g, modal_b = modal_colour
            dist = redmean_distance(r, g, b, modal_r, modal_g, modal_b)

            # We want to find the most common colour that is furthest from the modal colour

            weights = counts * (dist**dw)
            # weights = dist
            
            if len(values) == 0:
                mapping[(lc, tc)] = [0, 0, 0]
                continue
            _sorter = np.argsort(weights)[::-1]
            values = values[_sorter]
            counts = counts[_sorter]
            weights = weights[_sorter]
            print(lc, tc, counts[:10], values[:10], weights[:10], dist[:10])
            for j, hex_val in enumerate(values[:top_n_colours]):
                r, g, b = hex_to_rgb(hex_val)
                colour_matrix[i, j+3] = [r, g, b]
            max_hex = values[0]
            r, g, b = hex_to_rgb(max_hex)
            rgb_landcover[mask] = [r, g, b]
            mapping[(lc, tc)] = [r, g, b]

        expected_rgb_landcover = rgb_landcover.copy()
        expected_rgb_landcover[rivers == 255] = sat[rivers == 255]
        img.fromarray(expected_rgb_landcover).save(os.path.join(output_dir, f'rivers_expected.png'))

        loss = np.abs(expected_rgb_landcover - rgb_landcover).sum() / (4 * rivers.sum())
        with open(os.path.join(output_dir, f'loss_{loss:.3f}'), 'w') as f:
            pass
        print(f'Loss: {loss}')
        
        img.fromarray(rgb_landcover).save(os.path.join(output_dir, f'rivers.png'))
        h, w, _ = colour_matrix.shape
        h *= 10
        w *= 10
    
        colour_matrix = cv2.resize(colour_matrix, (w, h), interpolation=cv2.INTER_NEAREST)
        img.fromarray(colour_matrix).save(os.path.join(output_dir, 'rivers_colour_matrix.png'))

        self.mapping = mapping
        self.save_mapping(planet_cfg.data_dir)


    def save_mapping(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, self.mapping_file), 'w') as f:
            for lc, tc in self.mapping:
                r, g, b = self.mapping[(lc, tc)]
                f.write(f"{lc} {tc} {r} {g} {b}\n")
            

    def load_mapping(self, path: str) -> bool:
        if not os.path.exists(os.path.join(path, self.mapping_file)):
            return False
        with open(os.path.join(path, self.mapping_file), 'r') as f:
            self.mapping = {}
            for line in f:
                l, t, r, g, b = line.split()
                self.mapping[int(l), int(t)] = (int(r), int(g), int(b))
        return True

    @timing
    def get_sketch(self, landcover: np.ndarray, temp: np.ndarray, rivers: np.ndarray) -> np.ndarray:
        if landcover.shape != rivers.shape:
            h, w = rivers.shape
            landcover = cv2.resize(landcover, (w, h), interpolation=cv2.INTER_NEAREST)
            temp = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
        h, w = landcover.shape
        sketch = np.zeros((h, w, 3), np.uint8)
        land_step = 255 // self.planet_cfg.landcover_classes
        temp_step = 255 // self.planet_cfg.temp_classes
        landcover = landcover // land_step
        temp = (temp + 1) // temp_step
        temp[landcover == 0] = 0
        vals, num = np.unique(temp, return_counts=True)
        nonzero = vals != 0
        vals = vals[nonzero]
        num = num[nonzero]
        if len(vals) == 0:
            mode_temp = 0
        else:
            mode_temp = vals[np.argmax(num)]
        temp[(landcover != 0) & (temp == 0)] = mode_temp
        for i, (lc, tc) in enumerate(product(range(self.l), range(self.t))):
            mask = (landcover == lc) & (temp == tc) & (rivers > 0)
            if not mask.any():
                continue
            r, g, b = self.mapping.get((lc, tc), [0, 0, 0])
            sketch[mask] = [r, g, b]
        return sketch

if __name__ == '__main__':

    planet_cfg = PlanetConfig()
    RiverModalSketch(planet_cfg)
    RiverModalSketch(planet_cfg)

    
    all_rivers = open_image_array(os.path.join(planet_cfg.data_dir, f'global_river_order_new.tif')).astype(np.uint8)
    h, w = all_rivers.shape
    rgb_landcover = open_image_array(os.path.join(planet_cfg.data_dir, 'World_LandCover_RGB_512x256.png'))
    rgb_landcover = cv2.resize(rgb_landcover, (w, h), interpolation=cv2.INTER_NEAREST)
        
    landcover = open_image_array(os.path.join(planet_cfg.data_dir, 'World_LandCover_512x256.png'))
    land_step = 255 // planet_cfg.landcover_classes
    landcover //= land_step
    landcover = cv2.resize(landcover, (w, h), interpolation=cv2.INTER_NEAREST)

    temp = open_image_array(os.path.join(planet_cfg.data_dir, 'World_Temp_512x256.png'))
    t = planet_cfg.temp_classes + 1
    temp_step = 255 // planet_cfg.temp_classes
    temp[temp == 0] += 1
    temp = temperature_paint(temp, planet_cfg.downscale_cfg).astype(np.uint16)
    temp = ((temp + 1) // temp_step).astype(np.uint8)
    temp = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

    sat = open_image_array(os.path.join(planet_cfg.data_dir, 'world.satellite.16384x8192.png'))

    orders = [3, 4]
    distance_weights = [1, 2]
    mds = [0, 4]
    order_weights = [1, 2, 3]

    for ords, dw, md, ow in product(orders, distance_weights, mds, order_weights):
        RiverModalSketch(planet_cfg, ords, dw, md, ow, all_rivers, rgb_landcover.copy(), landcover.copy(), temp.copy(), sat)
