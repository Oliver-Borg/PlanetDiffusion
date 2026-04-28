from .utils import PlanetConfig, open_image_array
from .landcover_utils import gray_to_land
from .sketch_gen import temperature_paint, get_buckets, dilate_paint

from torch.utils.data import DataLoader
import numpy as np
from scipy.ndimage import generic_filter
from tqdm import tqdm
from PIL import Image
from numpy.random import choice
import os
from cv2 import dilate, erode, circle
from dataclasses import replace
import cv2

class InverseLandcover():

    def __init__(self, planet_cfg: PlanetConfig, temp: np.ndarray = None, 
                 land: np.ndarray = None, sat: np.ndarray = None,
                 dem: np.ndarray = None) -> None:
        self.planet_cfg = planet_cfg

        land_step = 255 // planet_cfg.landcover_classes
        temp_step = 255 // planet_cfg.temp_classes
        dem_step = 256 // planet_cfg.colours

        H = 256*2**planet_cfg.size
        W = 512*2**planet_cfg.size
        if temp is None:
            temp = open_image_array(os.path.join(planet_cfg.data_dir, f'World_Temp_{W}x{H}.png'))
        if land is None:
            land = open_image_array(os.path.join(planet_cfg.data_dir, f'World_LandCover_{W}x{H}.png'))
        if sat is None:
            sat = open_image_array(os.path.join(planet_cfg.data_dir, f'world.satellite.{W}x{H}.png'))
        if dem is None:
            dem = open_image_array(os.path.join(planet_cfg.data_dir, f'World_DEM_{W}x{H}.png'))

        temp_sketch = temperature_paint(temp, planet_cfg.downscale_cfg).astype(np.uint16)
        temp_sketch = ((temp_sketch + 1) // temp_step).astype(np.uint8)

        land_sketch = land // land_step

        buckets = get_buckets(dem, planet_cfg)
        dem_sketch = dilate_paint(dem, planet_cfg, buckets=buckets).astype(np.uint16)
        dem_sketch = ((dem_sketch + 1) // dem_step).astype(np.uint8)

        # dem_sketch[:, :] = 0

        sat_hex = np.zeros_like(sat[:, :, 0]).astype(np.uint32)
        sat_hex = sat[:, :, 0]*256**2 + sat[:, :, 1]*256 + sat[:, :, 2]
        sat_uniques = np.unique(sat_hex)
        indices = np.arange(sat_uniques.shape[0])
        # Convert sat hex values to indexes in the range 0 to s
        mapping = dict(zip(sat_uniques, indices))
        sat_hex = np.vectorize(mapping.get)(sat_hex)

        l = planet_cfg.landcover_classes + 1
        t = planet_cfg.temp_classes + 1
        d = planet_cfg.colours + 1
        s = sat_uniques.shape[0] + 1


        

        combined_sketch = np.zeros_like(sat_hex).astype(np.uint32)
        combined_sketch = (combined_sketch + temp_sketch) * d
        combined_sketch = (combined_sketch + dem_sketch) * s
        combined_sketch = combined_sketch + sat_hex
        combined_sketch = combined_sketch.astype(np.uint32)

        # Find the mapping between combined_sketch and land_sketch
        # Then reduce on the combined_sketch to find the counts of each landcover class for
        # each combined_sketch value
        counts = np.zeros((t*d*s, l), dtype=np.uint32)
        for i in range(l):
            counts[:, i] = np.bincount(combined_sketch[land_sketch == i].ravel(), minlength=t*d*s)
        self.counts = counts

        # Use argmax to find the most likely landcover class for each combined_sketch value
        new_land = np.zeros_like(land_sketch)
        combined_values = np.argmax(counts, axis=1)
        combined_indices = np.arange(t*d*s)
        combined_mapping = dict(zip(combined_indices, combined_values))
        new_land = np.vectorize(combined_mapping.get)(combined_sketch)
        self.new_land = new_land

        new_land = new_land * land_step
        new_land = new_land.astype(np.uint8)
        Image.fromarray(new_land).save(os.path.join(planet_cfg.data_dir, f'World_LandCover_{W}x{H}_inverse.png'))
        new_land = gray_to_land(new_land)
        Image.fromarray(new_land).save(os.path.join(planet_cfg.data_dir, f'World_LandCover_RGB_{W}x{H}_inverse.png'))


        


if __name__ == "__main__":
    planet_cfg = PlanetConfig(size=5)
    modal_sketch = InverseLandcover(planet_cfg)