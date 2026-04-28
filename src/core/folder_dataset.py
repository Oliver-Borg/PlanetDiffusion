from dataclasses import replace
import itertools
import os
import random
import re
from typing import Dict, List, Optional
import json
from PIL import Image
from line_profiler import profile
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import cv2

from planetAI.src.data.dataset import RAMDataset
from planetAI.src.data.sphere_mapping import (
    QuadSphere,
    get_normal_shape,
    quad_to_normal,
)
from planetAI.src.data.utils import PlanetConfig, np_rgb, np_to_tensor, open_image_array


@profile
def load_args_config(filepath: str) -> Dict:
    """Load and clean config from JSON file"""
    with open(filepath) as f:
        file_content = f.read()
        try:
            data = json.loads(file_content)
        except json.JSONDecodeError:
            dicts = file_content.split("}")[:2]
            dicts[0] += "}"
            dicts[1] += "}"
            data = {
                "inference_args": json.loads(dicts[0]),
                "planet_cfg": json.loads(dicts[1]),
            }

    cleaned_data = {"inference_args": {}, "planet_cfg": {}}
    for t in ("inference_args", "planet_cfg"):
        for k in data[t]:
            if data[t][k] is None:
                continue
            cleaned_data[t][k] = data[t][k]

    return cleaned_data


def get_cached_timestamps(folder: str, inference_args_filter: str, planet_cfg_filter: str):
    key = "|".join([folder, inference_args_filter, planet_cfg_filter])
    cache_path = os.path.join(folder, "timestamp_cache.json")

    if not os.path.exists(cache_path):
        return None

    with open(cache_path, "r") as f:
        cached_json = json.load(f)
    return cached_json.get(key)


def save_cached_timestamps(folder: str, inference_args_filter: str, planet_cfg_filter: str, timestamps: list[str]):
    key = "|".join([folder, inference_args_filter, planet_cfg_filter])
    cache_path = os.path.join(folder, "timestamp_cache.json")

    os.makedirs(folder, exist_ok=True)
    cached_json = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            cached_json = json.load(f)

    cached_json[key] = timestamps

    with open(cache_path, "w") as f:
        cached_json = json.dump(cached_json, f)


im_cache = {}


class FolderDataset(Dataset):
    @profile
    def __init__(
        self,
        folder: str,
        inference_args_filter: str = "",
        planet_cfg_filter: str = "",
        required_files: Optional[List[str]] = None,
        resize_shape: tuple[int, int] | None = None,
        expected_shape: tuple[int, int] | None = None,
        is_blended: bool = False,
        use_subfolders: bool = False,
        max_len: int = 10000,
        force_hflip: bool = False,
    ):
        """
        Args:
            folder: Path to the folder containing the dataset
            required_files: List of file types that must be present (without timestamp)
                          e.g. ['args.json', 'dem_output.png', 'dem_real.png']
        """
        self.folder = folder
        self.transform = transforms.ToTensor()
        self.resize_shape = resize_shape
        self.expected_shape = expected_shape
        self.is_blended = is_blended
        self.max_len = max_len

        # Parse filters
        self.inference_filter = (
            dict(pair.split(":") for pair in inference_args_filter.split(",")) if inference_args_filter else {}
        )
        self.planet_filter = dict(pair.split(":") for pair in planet_cfg_filter.split(",")) if planet_cfg_filter else {}

        # Default required files if none specified
        if required_files is None:
            required_files = [
                "args.json",
                "dem_output.png",
                "dem_real.png",
                "sat_output.png",
                "sat_real.png",
                "display.png",
            ]
        if self.is_blended and "display.png" in required_files:
            required_files.remove("display.png")
            required_files.extend(["temp_sketch.png", "land_sketch.png", "sketch.png", "modal_sketch.png"])

        # Group files by timestamp
        self.timestamp_groups: Dict[str, Dict[str, str]] = {}

        if use_subfolders:
            # Scan all subfolders
            for subfolder in os.listdir(folder):
                subfolder_path = os.path.join(folder, subfolder)
                if not os.path.isdir(subfolder_path):
                    continue

                self._scan_folder(subfolder_path, required_files, prefix=f"{subfolder}_")
        else:
            # Scan just the main folder
            self._scan_folder(folder, required_files)

        keys_to_delete = []
        timestamp_keys = list(sorted(self.timestamp_groups.keys()))
        for i in range(len(timestamp_keys) - 1):
            t_key = timestamp_keys[i]
            files = self.timestamp_groups[t_key]
            next_key = timestamp_keys[i + 1]
            next_files = self.timestamp_groups[next_key]
            move_dem = "dem_real.png" in next_files and "dem_real.png" not in files
            move_sat = "sat_real.png" in next_files and "sat_real.png" not in files
            if move_dem:
                self.timestamp_groups[t_key]["dem_real.png"] = next_files["dem_real.png"]
            if move_sat:
                self.timestamp_groups[t_key]["sat_real.png"] = next_files["sat_real.png"]
            if move_dem and move_sat and "sat_output.png" not in next_files:
                keys_to_delete.append(next_key)

        for key in keys_to_delete:
            del self.timestamp_groups[key]

        # Filter groups that have all required files and match filters
        valid_timestamps = get_cached_timestamps(folder, inference_args_filter, planet_cfg_filter)
        if valid_timestamps is None:
            valid_timestamps = []
            for timestamp, files in self.timestamp_groups.items():
                if all(req in files.keys() for req in required_files):
                    # Load config and check filters
                    config_file = os.path.join(
                        self.folder,
                        os.path.dirname(files["args.json"]),
                        os.path.basename(files["args.json"]),
                    )
                    config = load_args_config(config_file)

                    if self._matches_filter(config["inference_args"], self.inference_filter) and self._matches_filter(
                        config["planet_cfg"], self.planet_filter
                    ):
                        valid_timestamps.append(timestamp)

        self.valid_timestamps = sorted(valid_timestamps)
        save_cached_timestamps(folder, inference_args_filter, planet_cfg_filter, self.valid_timestamps)
        base_len = len(self.valid_timestamps)
        self.hflip = False
        if self.is_blended:
            base_len *= 16
            grid_indices = list(range(16))
        else:
            grid_indices = [0]
        if (base_len < 2048 and base_len >= 1024) or force_hflip:
            # This is so we can have enough images to do the batched FID
            self.hflip = True
            base_len *= 2
            hflips = [False, True]
        else:
            hflips = [False]
        self.data_items = list(itertools.product(self.valid_timestamps, grid_indices, hflips))
        self.num_images = len(self.data_items)

        np.random.seed(0)
        self.data_mapping = np.arange(self.num_images)
        np.random.shuffle(self.data_mapping)

        print(f"Found {self.num_images} images matching the filters")

    def _matches_filter(self, config: Dict, filter_dict: Dict[str, str]) -> bool:
        """Check if config matches all key-value pairs in filter"""
        for key, value in filter_dict.items():
            value = value.strip()
            if key not in config or str(config[key]) != value:
                return False
        return True

    def _scan_folder(self, folder_path: str, required_files: List[str], prefix: str = ""):
        """Scan a folder for matching files and add them to timestamp_groups"""
        for filename in os.listdir(folder_path):
            if "_" not in filename:
                continue

            timestamp_regex = r"\d\d\d\d-\d\d-\d\d-\d\d-\d\d-\d\d(_\d+)?_"

            match = re.search(timestamp_regex, filename)
            if match is None:
                continue
            timestamp = prefix + match.group(0)
            file_type = filename[len(match.group(0)) :]

            if timestamp not in self.timestamp_groups:
                self.timestamp_groups[timestamp] = {}

            rel_path = os.path.relpath(folder_path, self.folder)
            self.timestamp_groups[timestamp][file_type] = os.path.join(rel_path, filename)

    def __len__(self):
        return min(self.num_images, self.max_len)

    def __getitem__(self, idx):
        return self.get_item(idx)

    @profile
    def get_item(self, idx):
        actual_idx = self.data_mapping[idx]
        timestamp, grid_idx, do_hflip = self.data_items[actual_idx]
        col = grid_idx % 4
        row = grid_idx // 4

        files = self.timestamp_groups[timestamp]

        result = {"valid": True}

        for file_type, filename in files.items():
            filepath = os.path.join(self.folder, filename)

            if file_type.endswith(".json"):
                result[file_type[:-5]] = load_args_config(filepath)

            elif file_type.endswith(".png"):
                if filepath in im_cache:
                    img = im_cache[filepath].copy()
                else:
                    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
                    im_cache[filepath] = img.copy()
                if img is None:
                    result["valid"] = False
                    continue

                if self.is_blended:
                    h, w = img.shape[:2]
                    r_x = random.randint(-10, 10)
                    r_y = random.randint(-10, 10)
                    s_y = 64 + row * 214 + r_y
                    s_x = 64 + col * 214 + r_x
                    img = img[s_y: s_y + 256, s_x: s_x + 256]
                if do_hflip:
                    if "display" in file_type:
                        h, w = img.shape[:2]
                        for i in range(0, w, h):
                            tile_w = min(i + h, w) - i
                            region = img[:, i: i + tile_w]
                            img[:, i: i + tile_w] = cv2.flip(region, 1)
                    else:
                        img = cv2.flip(img, 1)

                if self.resize_shape is not None:
                    target_h, target_w = self.resize_shape
                    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

                h, w = img.shape[:2]
                if self.expected_shape is not None:
                    if (h, w) != self.expected_shape and ("output" in file_type or "real" in file_type):
                        result["valid"] = False
                elif "output" in file_type or "real" in file_type:
                    self.expected_shape = (h, w)

                if img.dtype == np.uint16:
                    img = img.astype(np.float32) / 65535.0
                else:
                    img = img.astype(np.float32) / 255.0

                # Convert to RGB
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                elif len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


                result[file_type[:-4]] = self.transform(img) * 2 - 1

        return result


class FullPlanetFolderDataset(Dataset):
    def __init__(
        self,
        folder: str,
        timestamp: str,
    ):
        self.folder = folder
        self.timestamp = timestamp
        self.dem_output = open_image_array(os.path.join(folder, f"{timestamp}_dem_output.png"))
        h, w = self.dem_output.shape
        self.sat_output = open_image_array(os.path.join(folder, f"{timestamp}_sat_output.png"))
        self.temp_sketch = cv2.resize(
            open_image_array(os.path.join(folder, f"{timestamp}_temp_sketch.png")),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )
        self.land_sketch = cv2.resize(
            open_image_array(os.path.join(folder, f"{timestamp}_land_sketch.png")),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )
        self.dem_sketch = cv2.resize(
            open_image_array(os.path.join(folder, f"{timestamp}_sketch.png")),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )
        self.modal_sketch = cv2.resize(
            open_image_array(os.path.join(folder, f"{timestamp}_modal_sketch.png")),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )
        self.args = load_args_config(os.path.join(folder, f"{timestamp}_args.json"))

        self.planet_cfg = replace(PlanetConfig(**self.args["planet_cfg"]), data_dir="./planetAI/data")

        self.real_dataset = RAMDataset(
            self.planet_cfg,
            self.planet_cfg.output_channels(),
            self.planet_cfg.input_channels(),
            normalise=True,
            tile_size=256,
            mode="test",
            do_transforms=False,
        )

        self.images = np.dstack(
            [
                self.dem_output,
                self.sat_output,
                self.temp_sketch,
                self.land_sketch,
                self.dem_sketch,
                self.modal_sketch,
            ]
        )
        self.quad_shape = self.dem_output.shape

        normal_shape = get_normal_shape(self.quad_shape)

        self.quad_sphere = QuadSphere(shape=normal_shape)

        mask = quad_to_normal(self.dem_sketch)
        self.coords = self.quad_sphere.get_distributed_points(mask, n=40000)

        # 2025-09-22-02-21-05_temp_sketch.png
        # 2025-09-22-02-21-05_args.json
        # 2025-09-22-02-21-05_dem_normal.png
        # 2025-09-22-02-21-05_dem_output.png
        # 2025-09-22-02-21-05_dem_viridis.png
        # 2025-09-22-02-21-05_dem_viz.png
        # 2025-09-22-02-21-05_land_sketch.png
        # 2025-09-22-02-21-05_modal_sketch.png
        # 2025-09-22-02-21-05_normal_land_sketch.png
        # 2025-09-22-02-21-05_normal_modal_sketch.png
        # 2025-09-22-02-21-05_normal_rivers.png
        # 2025-09-22-02-21-05_normal_sketch.png
        # 2025-09-22-02-21-05_normal_sketch_gist_earth.png
        # 2025-09-22-02-21-05_normal_sketch_terrain.png
        # 2025-09-22-02-21-05_normal_sketch_viridis.png
        # 2025-09-22-02-21-05_normal_temp_sketch.png
        # 2025-09-22-02-21-05_normal_temp_sketch_coolwarm.png
        # 2025-09-22-02-21-05_normal_temp_sketch_RdYlBu_r.png
        # 2025-09-22-02-21-05_oceanmask.png
        # 2025-09-22-02-21-05_rivers.png
        # 2025-09-22-02-21-05_sat_normal.png
        # 2025-09-22-02-21-05_sat_output.png
        # 2025-09-22-02-21-05_sat_viz.png
        # 2025-09-22-02-21-05_sketch.png

    def __len__(self):
        return len(self.coords[0])

    def __getitem__(self, index: int):
        coord = (self.coords[0][index], self.coords[1][index])
        us, vs = self.quad_sphere.get_quad_tile_mapping(coord)

        delta = self.planet_cfg.delta

        _dem_output = np_to_tensor(self.dem_output[us, vs])
        _sat_output = np_to_tensor(self.sat_output[us, vs])

        sketches = [
            self.land_sketch,
            self.temp_sketch,
            self.dem_sketch,
            self.modal_sketch,
        ]
        _sketches: list[np.ndarray] = []
        for sketch in sketches:
            _sketch = cv2.resize(
                sketch[us, vs],
                fx=1 / delta,
                fy=1 / delta,
                dsize=None,
                interpolation=cv2.INTER_NEAREST,
            )
            _sketch = cv2.resize(
                _sketch,
                fx=delta,
                fy=delta,
                dsize=None,
                interpolation=cv2.INTER_NEAREST,
            )
            _sketches.append(_sketch)

        _land_sketch, _temp_sketch, _dem_sketch, _modal_sketch = _sketches

        if "FlippedEarth" in self.folder:
            _real_example = self.real_dataset.get_item_at_coords(
                -coord[0],
                coord[1],
                False,
                False,
                0.1,
                False,
                True,
                index % len(self.real_dataset),
            )
        else:
            _real_example = self.real_dataset.get_item(index % len(self.real_dataset))

        target = np_to_tensor(_real_example["target_image"])

        _viridis_dem = np_rgb(_dem_sketch, cmap="viridis")
        _rgb_temp = np.dstack([_temp_sketch] * 3)

        _display = np.hstack([_land_sketch, _rgb_temp, _viridis_dem])

        return {
            "args": self.args,
            "dem_real": target[3],
            "sat_real": target[:3],
            "dem_output": _dem_output,
            "sat_output": _sat_output,
            "valid": True,
            "display": np_to_tensor(_display),
        }
