import numpy as np
from dataclasses import replace
import itertools
import json
import os
from tqdm import tqdm
from PIL import Image

from planetAI.src.data.landcover_utils import translate_land, gray_to_land
from planetAI.src.data.utils import PlanetConfig, np_rgb, image_grid, clip

from .atlas_loader import AtlasLoader, QuadAtlasLoader
from .sketch_gen import dilate_paint, temperature_paint


class PrecipSketch:
    def __init__(self, planet_cfg: PlanetConfig, force: bool = False, exclude_dem: bool = True):
        self.planet_cfg = replace(planet_cfg, size=5)
        self.loader = QuadAtlasLoader(planet_cfg)
        self.exclude_dem = exclude_dem
        if force:
            self.mapping = self.extract_precip_mapping()
            self.save_mapping()
            return
        try:
            self.mapping = self.load_mapping()
        except (FileNotFoundError, json.JSONDecodeError):
            self.mapping = self.extract_precip_mapping()
            self.save_mapping()

    def extract_precip_mapping(self):
        dem = self.loader.quad_dem
        land = self.loader.quad_land
        land = translate_land(land)
        temp = self.loader.quad_temp
        temp_summer = self.loader.quad_temp_summer
        precipitation = self.loader.quad_precipitation
        precipitation_summer = self.loader.quad_precipitation_summer
        dem_sketch = dilate_paint(dem, self.planet_cfg)
        temp_sketch = temperature_paint(temp, self.planet_cfg)
        temp_sketch_summer = temperature_paint(temp_summer, self.planet_cfg)

        dems = np.hstack((dem_sketch, dem_sketch))
        lands = np.hstack((land, land))
        temps = np.hstack((temp_sketch, temp_sketch_summer))
        precips = np.hstack((precipitation, precipitation_summer))

        if self.exclude_dem:
            dems[:, :] = 0

        dem_vals = np.unique(dems)
        land_vals = np.unique(lands)
        temp_vals = np.unique(temps)

        mapping: dict[tuple[int, int, int], float] = {}

        for dem_val, land_val, temp_val in tqdm(list(itertools.product(dem_vals, land_vals, temp_vals))):
            precip_vals = precips[(dems == dem_val) & (lands == land_val) & (temps == temp_val)]
            combo_str = f"DEM: {dem_val}, Land: {land_val}, Temp: {temp_val}"
            if precip_vals.size == 0:
                print(f"No precipitation values found for {combo_str}. Setting to 0.")
                mapping[(dem_val, land_val, temp_val)] = 0.0
            else:
                mean_precip = float(np.mean(precip_vals))
                mapping[(dem_val, land_val, temp_val)] = mean_precip
                print(f"{combo_str} -> Mean Precipitation: {mean_precip:.2f}")
        return mapping

    def save_mapping(self):
        filepath = os.path.join(self.planet_cfg.data_dir, f"precip_mapping_{self.exclude_dem}.json")
        with open(filepath, "w") as f:
            json.dump({str(k): v for k, v in self.mapping.items()}, f, indent=4)

    def load_mapping(self):
        filepath = os.path.join(self.planet_cfg.data_dir, f"precip_mapping_{self.exclude_dem}.json")
        with open(filepath, "r") as f:
            data = json.load(f)
            self.mapping = {tuple([int(x) for x in k.strip("() ").split(",")]): v for k, v in data.items()}
        return self.mapping

    def get_sketch(
        self,
        dem_sketch: np.ndarray,
        land_sketch: np.ndarray,
        temp_sketch: np.ndarray,
        full_mapping: dict[tuple[int, int, int], float] | None = None,
    ) -> np.ndarray:
        dem_sketch = dem_sketch.copy()
        if self.exclude_dem:
            dem_sketch[:, :] = 0
        precip_sketch = np.zeros_like(dem_sketch, dtype=np.float32)
        mapping = self.mapping if full_mapping is None else full_mapping
        for (dem_val, land_val, temp_val), precip_val in mapping.items():
            precip_sketch[(dem_sketch == dem_val) & (land_sketch == land_val) & (temp_sketch == temp_val)] = precip_val
        precip_sketch[land_sketch == 0] = 0
        return precip_sketch / precip_sketch.max()  # TODO Consider changing how this is done

    def get_full_mapping(
        self,
        land_mults: dict[int, float] = {},
        temp_mults: dict[int, float] = {},
        combo_mults: dict[tuple[int, int], float] = {},
    ) -> dict[tuple[int, int, int], float]:
        full_mapping = {}
        for (dem_val, land_val, temp_val), base_precip in self.mapping.items():
            land_mult = land_mults.get(land_val, 1.0)
            temp_mult = temp_mults.get(temp_val, 1.0)
            combo_mult = combo_mults.get((land_val, temp_val), 1.0)
            full_precip = base_precip * land_mult * temp_mult * combo_mult
            full_mapping[(dem_val, land_val, temp_val)] = full_precip
        return full_mapping

    def get_full_mapping_matrix(
        self,
        full_mapping: dict[tuple[int, int, int], float] = {},
        land_labels: dict[int, str] = {},
        temp_labels: dict[int, str] = {},
        tile_size: int = 16,
    ) -> Image.Image:
        shape = (tile_size, tile_size)
        base_arr = np.ones(shape, dtype=np.uint8)
        land_vals = list(set([land_val for _, land_val, _ in full_mapping.keys()]))
        temp_vals = list(set([temp_val for _, _, temp_val in full_mapping.keys()]))
        max_val = max(full_mapping.values())
        # min_val = min(full_mapping.values())
        images = [Image.fromarray(np.zeros(shape, dtype=np.uint8))]
        images.extend([
            Image.fromarray(gray_to_land(base_arr * land_val).astype(np.uint8)) for land_val in land_vals
        ])
        # TODO I will basically just ignore dem_val here for now
        for row, temp_val in enumerate(temp_vals):
            for col, land_val in enumerate(land_vals):
                if col == 0:
                    images.append(Image.fromarray(np_rgb(base_arr * temp_val, cmap="coolwarm").astype(np.uint8)))
                precip_val = full_mapping.get((0, land_val, temp_val))
                norm_precip_val = int(clip((precip_val) / (max_val) * 255, 0, 255))
                images.append(Image.fromarray(np_rgb(base_arr * norm_precip_val, cmap="Blues").astype(np.uint8)))

        matrix = image_grid(
            images,
            len(temp_vals) + 1,
            len(land_vals) + 1,
            row_labels=[temp_labels.get(val, val) for val in temp_vals],
            col_labels=[land_labels.get(val, val) for val in land_vals],
        )

        return matrix


if __name__ == "__main__":
    from planetAI.src.data.utils import PlanetConfig

    cfg = PlanetConfig(size=5)
    precip_sketch = PrecipSketch(cfg, force=False)
    mapping = precip_sketch.mapping
    print(mapping)

    loader = AtlasLoader(cfg)

    dem = loader.dem
    land = loader.land
    land = translate_land(land)
    temp = loader.temp
    temp_summer = loader.temp_summer
    precipitation = loader.precipitation
    precipitation_summer = loader.precipitation_summer
    dem_sketch = dilate_paint(dem, cfg)
    temp_sketch = temperature_paint(temp, cfg)
    temp_sketch_summer = temperature_paint(temp_summer, cfg)

    discrete_precip_sketch = precip_sketch.get_sketch(dem_sketch, land, temp_sketch)
    discrete_precip_sketch_summer = precip_sketch.get_sketch(dem_sketch, land, temp_sketch_summer)

    Image.fromarray(
        np_rgb((discrete_precip_sketch / np.max(discrete_precip_sketch) * 255).astype(np.uint8), cmap="Blues")
    ).save(os.path.join(cfg.test_dir, "mean_precip_winter.png"))
    Image.fromarray(
        np_rgb((
            discrete_precip_sketch_summer / np.max(discrete_precip_sketch_summer) * 255
        ).astype(np.uint8), cmap="Blues")
    ).save(os.path.join(cfg.test_dir, "mean_precip_summer.png"))
