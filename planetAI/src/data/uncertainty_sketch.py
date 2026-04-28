from dataclasses import replace
import numpy as np
import os
from PIL import Image

from planetAI.src.data.modal_sketch import ModalSketch

from .sketch_gen import dilate_paint, get_buckets, temperature_paint
from .sphere_mapping import normal_to_quad
from .landcover_utils import gray_to_land, translate_land
from .utils import PlanetConfig, np_rgb, timing
from .atlas_loader import AtlasLoader
from .landcover_utils import used_classes, landcover_mapping


class UncertaintySketcher:
    def __init__(self, planet_cfg: PlanetConfig, force: bool = False):
        self.planet_cfg = planet_cfg
        self.count_text = os.path.join(self.planet_cfg.data_dir, "sketch_counts.txt")
        self.counts: dict[tuple[int, int, int], float] = {}
        if not self.load_counts() or force:
            self.get_counts()

    def load_counts(self):
        if not os.path.exists(self.count_text):
            return False
        try:
            with open(self.count_text, "r") as f:
                lines = f.readlines()
            for line in lines:
                key_part, val_part = line.rsplit(" ", maxsplit=1)
                parts = map(int, key_part.split(" "))
                land_value, temp_value, dem_value = parts
                self.counts[(land_value, temp_value, dem_value)] = float(val_part)
        except Exception:
            return False
        return True

    def save_counts(self):
        with open(self.count_text, "w") as f:
            for k, count in self.counts.items():
                land_value, temp_value, dem_value = k
                f.write(f"{land_value} {temp_value} {dem_value} {count}\n")

    def get_counts(self):

        values_list = []
        counts_list = []
        self.counts = {}

        for i in range(self.planet_cfg.sketch_lod_levels + 1):
            counts, values = self._get_counts(self.planet_cfg.downscale_offset - i)
            values_list.append(values)
            # The counts from larger sketches have 4x as many pixels and in training we use these 10% per level
            counts_list.append(counts * 4**-i * 10**-i)

        values = np.concatenate(values_list)
        counts = np.concatenate(counts_list)

        for value, count in zip(values, counts):
            land_value = value // (256**2)
            temp_value = value // 256 % 256
            dem_value = value % 256
            if land_value == 0:
                continue
            self.counts.setdefault((land_value, temp_value, dem_value), 0)
            self.counts[(land_value, temp_value, dem_value)] += count

        self.save_counts()

    @timing
    def _get_counts(self, downscale_offset: int = 5):
        # TODO add weighted counts for more detailed sketches
        planet_cfg = replace(self.planet_cfg, downscale_offset=downscale_offset)
        atlas_loader = AtlasLoader(planet_cfg)
        downland_sketch = atlas_loader.downland_sketch
        downtemp_sketch = atlas_loader.downtemp_sketch
        downsketch = atlas_loader.downsketch
        downtemp_summer_sketch = atlas_loader.downtemp_sketch_summer

        down_mars_dem = atlas_loader.down_mars_dem
        down_mars_buckets = (
            get_buckets(down_mars_dem, self.planet_cfg) if self.planet_cfg.bucketing_mode == "uniform" else None
        )

        mars_downtemp_sketch = temperature_paint(atlas_loader.down_mars_temp, self.planet_cfg)
        mars_downland_sketch = np.full_like(mars_downtemp_sketch, 255)
        mars_downsketch = dilate_paint(
            down_mars_dem,
            self.planet_cfg.downscale_cfg,
            buckets=down_mars_buckets,
        )

        full_combined_sketch = (
            translate_land(downland_sketch).astype(np.uint32) * 256**2
            + downtemp_sketch.astype(np.uint32) * 256
            + downsketch.astype(np.uint32)
        )

        full_combined_sketch_summer = (
            translate_land(downland_sketch).astype(np.uint32) * 256**2
            + downtemp_summer_sketch.astype(np.uint32) * 256
            + downsketch.astype(np.uint32)
        )

        full_mars_combined_sketch = (
            mars_downland_sketch.astype(np.uint32) * 256**2
            + mars_downtemp_sketch.astype(np.uint32) * 256
            + mars_downsketch.astype(np.uint32)
        )

        full_combined_sketch = normal_to_quad(full_combined_sketch, True)
        full_combined_sketch_summer = normal_to_quad(full_combined_sketch_summer, True)
        full_mars_combined_sketch = normal_to_quad(full_mars_combined_sketch, True)

        full_combined_sketch = np.vstack([full_combined_sketch, full_combined_sketch_summer])

        values, counts = np.unique(full_combined_sketch, return_counts=True)
        mars_values, mars_counts = np.unique(full_mars_combined_sketch, return_counts=True)
        values = np.append(values, mars_values)
        counts = np.append(counts, mars_counts)
        land_values = values // 256**2 > 0
        counts = counts[land_values]
        values = values[land_values]

        return counts, values

    def get_uncertainty_sketch(self, combined_sketch: np.ndarray, save: bool = False):
        uncertainty_mask = np.zeros((*combined_sketch.shape[:2], 3), dtype=np.uint8)

        # TODO Excluded mars statistics from this count
        filtered_counts = {k: v for k, v in self.counts.items() if k[0] < 256}

        uncertainty_map = {
            0.0: (161, 38, 45),
            (0.1 / len(filtered_counts)): (203, 97, 25),
            (0.25 / len(filtered_counts)): (232, 191, 40),
            (1.0): (0, 142, 94),
        }
        values = np.unique(combined_sketch)
        total = sum(filtered_counts.values())

        for value in values:
            land_value = value // (256**2)
            temp_value = value // 256 % 256
            dem_value = value % 256
            if land_value == 0:
                continue
            count = filtered_counts.get((land_value, temp_value, dem_value), 0.0)
            for k, v in uncertainty_map.items():
                if count / total <= k:
                    uncertainty_mask[combined_sketch == value] = v
                    break
        if save:
            Image.fromarray(uncertainty_mask).save(os.path.join(self.planet_cfg.test_dir, "uncertainty_mask.png"))

        return uncertainty_mask


if __name__ == "__main__":
    planet_cfg = PlanetConfig(sketch_lod_levels=2)
    uncertainty_sketcher = UncertaintySketcher(planet_cfg, force=False)

    shape = (256, 512)
    dem = np.zeros(shape, dtype=np.uint8)
    temp = np.zeros(shape, dtype=np.uint8) + 256 // 5 - 1
    land = np.zeros(shape, dtype=np.uint8)

    stacked_sketch = np.dstack([dem, temp, land])

    shape = stacked_sketch.shape[:2]

    h, w = shape

    num_landcover_classes = len(used_classes)
    num_temp_classes = 5
    num_dem_classes = 4

    spacing = 0
    size = 16

    width = (num_temp_classes * num_dem_classes) * size + (num_temp_classes * num_dem_classes + 1) * spacing
    height = num_landcover_classes * size + (num_landcover_classes + 1) * spacing

    start_y = (h - height) // 2
    start_x = (w - width) // 2

    for i, tc in enumerate(range(1, num_temp_classes + 1)):
        for j, dc in enumerate(range(1, num_dem_classes + 1)):
            for k, lc in enumerate(used_classes):
                y_off = start_y + spacing + k * (size + spacing)
                x_off = start_x + spacing + j * (size + spacing) + i * num_dem_classes * (size + spacing)
                colour = (
                    dc * (256 // num_dem_classes) - 1,
                    tc * (256 // num_temp_classes) - 1,
                    landcover_mapping[lc].gray_colour,
                )
                stacked_sketch[y_off: y_off + size, x_off: x_off + size] = colour

    dem = stacked_sketch[:, :, 0]
    temp = stacked_sketch[:, :, 1]
    land = translate_land(stacked_sketch[:, :, 2])

    full_combined_sketch = (
        land.astype(np.uint32) * 256**2
        + temp.astype(np.uint32) * 256
        + dem.astype(np.uint32)
    )

    uncertainty_palette = uncertainty_sketcher.get_uncertainty_sketch(full_combined_sketch, save=True)

    modal_sketch = ModalSketch(planet_cfg).get_sketch(land, temp)
    Image.fromarray(modal_sketch).save(os.path.join(planet_cfg.test_dir, "modal_palette.png"))
    Image.fromarray(gray_to_land(land)).save(os.path.join(planet_cfg.test_dir, "land_palette.png"))
    Image.fromarray(np_rgb(temp, cmap="coolwarm")).save(os.path.join(planet_cfg.test_dir, "temp_palette.png"))
    Image.fromarray(np_rgb(dem, cmap="viridis")).save(os.path.join(planet_cfg.test_dir, "dem_palette.png"))
