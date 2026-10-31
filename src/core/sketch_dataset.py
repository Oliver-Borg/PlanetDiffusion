from dataclasses import replace
import os
from pathlib import Path
import cv2
from torch.utils.data import Dataset
import numpy as np

from planetAI.src.data.atlas_loader import AtlasLoader
from planetAI.src.data.dataset import get_river_upa_mask
from planetAI.src.data.landcover_utils import translate_land
from planetAI.src.data.sketch_gen import dilate_paint, temperature_paint, DEFAULT_BUCKETS
from planetAI.src.data.sphere_mapping import normal_to_quad
from planetAI.src.data.utils import PlanetConfig, open_image_array


class SketchDataset(Dataset):
    def __init__(self, root_dir, planet_cfg: PlanetConfig = PlanetConfig()):
        self.root_dir = Path(root_dir)
        self.sketch_folders: list[Path] = [f for f in self.root_dir.iterdir() if f.is_dir()]
        self.planet_cfg = planet_cfg
        self.atlas_loader = AtlasLoader(self.planet_cfg)

    def __len__(self):
        return len(self.sketch_folders)

    def __getitem__(self, idx):
        folder: Path = self.sketch_folders[idx]

        files = os.listdir(folder)
        mapping = {}
        for file in files:
            if "dem" in file:
                mapping["dem_sketch.png"] = file
            elif "land" in file:
                mapping["landcover_sketch.png"] = file
            elif "temp" in file:
                mapping["temperature_sketch.png"] = file
            elif "upa" in file:
                if "quad_river_upa.npy" in file:
                    mapping["quad_river_upa.npy"] = file
                elif "normal_river_upa.npy" in file:
                    mapping["normal_river_upa.npy"] = file
                else:
                    mapping["river_upa.tif"] = file

        dem_path = os.path.join(folder, mapping.get("dem_sketch.png", "dem_sketch.png"))
        landcover_path = os.path.join(folder, mapping.get("landcover_sketch.png", "landcover_sketch.png"))
        temperature_path = os.path.join(folder, mapping.get("temperature_sketch.png", "temperature_sketch.png"))
        river_path = os.path.join(folder, mapping.get("river_upa.tif", "river_upa.tif"))
        river_array_path = os.path.join(folder, mapping.get("quad_river_upa.npy", "quad_river_upa.npy"))
        normal_river_array_path = os.path.join(folder, mapping.get("normal_river_upa.npy", "normal_river_upa.npy"))

        required_files = [dem_path, landcover_path, temperature_path]
        for file_path in required_files:
            if not Path(file_path).exists():
                raise FileNotFoundError(
                    f"Required file {file_path} not found in {folder}"
                )

        dem_sketch = open_image_array(dem_path)
        landcover_sketch = open_image_array(landcover_path)
        temperature_sketch = open_image_array(temperature_path)
        if folder.name == "FlippedEarth":
            river_sketch = self.atlas_loader.river_upa
            river_sketch = get_river_upa_mask(
                river_sketch,
                river_sketch.max(),
                replace(self.planet_cfg, river_upa_variance_chance=0.0, river_upa_dropout_chance=0.0)
            )
            river_sketch = np.flipud(river_sketch)
        elif os.path.exists(river_array_path):
            river_sketch = np.load(river_array_path)
            print(f"Loaded rivers from np file with min {river_sketch.min()} and max {river_sketch.max()}")
            river_sketch = np.sqrt(river_sketch / river_sketch.max()) * 255
        else:
            river_sketch = open_image_array(river_path)

        normal_river_sketch = None
        if os.path.exists(normal_river_array_path):
            normal_river_sketch = np.load(normal_river_array_path)
            normal_river_sketch = ((normal_river_sketch / normal_river_sketch.max()) * 255).astype(np.uint8)
            normal_river_sketch[normal_river_sketch < 10] = 0
            n, river_segments = cv2.connectedComponents((normal_river_sketch > 0).astype(np.uint8) * 255, connectivity=8)
            filtered_river_sketch = np.zeros_like(normal_river_sketch)
            for i in range(n):
                seg_mask = river_segments == i
                if np.count_nonzero(seg_mask) < 100:
                    continue
                vals = normal_river_sketch[seg_mask]
                if vals.max() > 60:
                    filtered_river_sketch[seg_mask] = normal_river_sketch[seg_mask]
            normal_river_sketch = np.sqrt(filtered_river_sketch.astype(np.float32) / 255) * 255
            normal_river_sketch = normal_river_sketch.astype(np.uint8)
            river_sketch = normal_to_quad(normal_river_sketch, discrete=True)

        if len(np.unique(dem_sketch)) > 5:
            dem_sketch = dilate_paint(
                dem_sketch,
                self.planet_cfg,
                buckets=DEFAULT_BUCKETS if "Earth" in folder.name else None
            )
        if len(np.unique(temperature_sketch)) > 6:
            temperature_sketch = temperature_paint(temperature_sketch, self.planet_cfg)
        landcover_sketch = translate_land(landcover_sketch)

        return {
            "name": folder.name,
            "dem_sketch": dem_sketch,
            "landcover_sketch": landcover_sketch,
            "temperature_sketch": temperature_sketch,
            "river_sketch": river_sketch,
        }


if __name__ == "__main__":
    sketch_dataset = SketchDataset("./user_sketches")
    for i in range(len(sketch_dataset)):
        item = sketch_dataset[i]
        print(f"Loaded sketch {item['name']}")
