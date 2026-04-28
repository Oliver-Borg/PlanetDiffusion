import os

import numpy as np
from PIL import Image
import cv2

from planetAI.src.data.utils import PlanetConfig, open_image_array
from planetAI.src.data.modal_sketch import ModalSketch
from planetAI.src.data.sketch_gen import temperature_paint
from .interface.utils import create_temp_preset, derive_landcover, planet_noise
from .interface.interface_types import TempPresetEventMetadata
from planetAI.src.data.noise_settings import NoiseSettings

wanted_years = [
    "225Ma",
    "200Ma",
    "150Ma",
    "65Ma",
    "0Ma",
]


if __name__ == "__main__":
    planet_cfg = PlanetConfig()
    paleo_dir = planet_cfg.data_dir.replace("/data", "/paleo_tifs")
    paleo_tifs = os.listdir(paleo_dir)
    tifs = []
    modal_sketcher = ModalSketch(planet_cfg)
    for year in wanted_years:
        for paleo_tif in paleo_tifs:
            if paleo_tif.endswith("_" + year + ".tif"):
                tif = open_image_array(os.path.join(paleo_dir, paleo_tif))
                tif = cv2.resize(tif, (512, 256), interpolation=cv2.INTER_LANCZOS4)
                tif = tif / tif.max() * 1.3
                tif_sketch = np.clip(np.ceil(tif * 4), 0.0, 4.0) * 64 - 1
                tif_sketch = np.maximum(tif_sketch, 0)
                tif = tif * 255

                def_num_latitudes = 9
                max_temp = 37
                min_temp = -45
                temp_steepness = 20
                temp_factor = 25

                lat_temp_list = []

                for i in range(def_num_latitudes):
                    latitude = (i * 180 / (def_num_latitudes - 1)) - 90
                    temperature = (
                        -abs(latitude) / 90 * (max_temp - min_temp) + max_temp
                    )
                    lat_temp_list.append((latitude, temperature))
                noise_settings = NoiseSettings(
                    frequency=0.016
                )
                noise = planet_noise((256, 512), noise_settings) * 255

                temperature = create_temp_preset(
                    TempPresetEventMetadata(
                        type=None,
                        x=None,
                        y=None,
                        brush_size=0,
                        lat_temp_list=lat_temp_list,
                        noise_weight=0.25,
                        pivot=0,
                        min_temp=min_temp,
                        max_temp=max_temp,
                        noise_settings=noise_settings,
                        noise=noise,
                        display_as_sketch=True,
                        modal_view=False,
                        steepness=temp_steepness,
                        factor=temp_factor,
                    ),
                    (256, 512),
                )
                combined = temperature.copy()
                factor = np.ceil(
                    ((tif / 255) ** temp_steepness)
                    * temp_factor
                )
                combined[factor > 0] = combined[factor > 0] / factor[factor > 0]
                combined = np.clip(combined, 0, 255).astype(np.uint8)
                combined[combined == 0] = 1

                landcover = derive_landcover(tif, combined / combined.max() * 255)

                modal_sketch = modal_sketcher.get_sketch(
                    landcover,
                    temperature_paint(combined, planet_cfg)
                )
                folder = f"./user_sketches/Paleo_{year}/"
                os.makedirs(folder, exist_ok=True)
                Image.fromarray(tif_sketch.astype(np.uint8)).save(os.path.join(folder, "dem_sketch.png"))
                Image.fromarray(combined).save(os.path.join(folder, "temperature_sketch.png"))
                Image.fromarray(landcover).save(os.path.join(folder, "landcover_sketch.png"))
                Image.fromarray(modal_sketch).save(os.path.join(folder, "modal_sketch.png"))

                continue
