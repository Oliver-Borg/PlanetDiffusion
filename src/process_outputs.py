from genericpath import isfile
import os

import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import shutil

from planetAI.src.data.landcover_utils import gray_to_land

from .core.sketch_dataset import SketchDataset

from planetAI.src.data.sphere_mapping import quad_to_normal
from planetAI.src.data.utils import np_rgb, open_image_array
from skimage.morphology import skeletonize


def calculate_likely_sketch_name(
    dem_sketch: np.ndarray,
    land_sketch: np.ndarray,
    temp_sketch: np.ndarray,
    dem_sketches: dict[str, np.ndarray] = {},
    land_sketches: dict[str, np.ndarray] = {},
    temp_sketches: dict[str, np.ndarray] = {},
) -> str:
    closest_score = 128  # Arbitrary score to prevent false positives for no matching name
    closest_name = "Other"
    for name in dem_sketches:
        mask = dem_sketches[name]
        if dem_sketch.shape != dem_sketches[name].shape:
            continue
        score = (
            np.mean(np.abs(dem_sketch[mask > 0] - dem_sketches[name][mask > 0])) +
            np.mean(np.abs(land_sketch[mask > 0] - land_sketches[name][mask > 0])) +
            np.mean(np.abs(temp_sketch[mask > 0] - temp_sketches[name][mask > 0]))
        )
        if score < closest_score:
            closest_score = score
            closest_name = name
    return closest_name


def viridis_to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert a viridis colormap image to grayscale."""
    if img.ndim == 3 and img.shape[2] == 3:
        # Convert RGB to grayscale using luminosity method
        gray = (
            0.299 * img[:, :, 0]
            + 0.587 * img[:, :, 1]
            + 0.114 * img[:, :, 2]
        )
        return gray.astype(np.uint8)
    elif img.ndim == 2:
        return img
    else:
        raise ValueError("Input image must be either grayscale or RGB.")


def get_files_and_paths(current_path: str, recursive: bool = True):
    filepaths: dict[str, str] = {}  # Full_path -> name
    for file in os.listdir(current_path):
        fullpath = os.path.join(current_path, file)
        if isfile(fullpath):
            filepaths[fullpath] = file
        elif recursive:
            filepaths.update(get_files_and_paths(fullpath, recursive))
    return filepaths


if __name__ == "__main__":
    # input_dir = "/mnt/e/evaluation/artifacts/Coarse-real_inference:v19"
    # output_dir = "/mnt/e/evaluation/Coarse outputs"
    # input_dir = "/mnt/e/evaluation/Cold Equator and Tropical"
    # output_dir = "/mnt/e/evaluation/Cold Equator and Tropical"
    # input_dir = "/mnt/e/evaluation/artifacts/Old-real_inference:v22"
    # output_dir = "/mnt/e/evaluation/artifacts/Old-real_inference:v22"
    # input_dir = "/mnt/e/evaluation/New River UPA"
    # output_dir = "/mnt/e/evaluation/New River UPA"
    # input_dir = "/mnt/e/evaluation/River CFG and ControlNet qualitative"
    # output_dir = "/mnt/e/evaluation/River CFG and ControlNet qualitative"
    input_dir = "/mnt/e/evaluation/FlippedEarth"
    output_dir = "/mnt/e/evaluation/FlippedEarth"
    os.makedirs(output_dir, exist_ok=True)
    river_viz = True
    save_sketches = True
    group_outputs_automatically = True
    create_oceanmask = True
    sort_only = False
    do_sort = False
    sort_by_timestamp = True
    force_dem = True
    sketch_shape = (256, 512)

    input_files = get_files_and_paths(input_dir, recursive=False)
    output_files = get_files_and_paths(output_dir)

    input_files.update(output_files)

    files_per_timestamp: dict[str, dict[str, str]] = {}
    t_len = len("2025-03-05-18-21-00")
    for fullpath, file in input_files.items():
        if not file.endswith(".png") and not file.endswith(".json") and not file.endswith(".tif"):
            continue
        timestamp = file[:t_len]
        suffix = file[t_len + 1:]
        files_per_timestamp.setdefault(timestamp, {})
        files_per_timestamp[timestamp][suffix] = fullpath

    if group_outputs_automatically:
        # Sometimes the dem_normal.png has a slightly later timestamp than the dem_output.png
        # 1. Sort the timestamps
        # 2. For each timestamp, check if dem_normal.png exists and if the previous timestamp has dem_output.png
        # 3. If such a pair exists, move the dem_normal.png and sat_normal.png
        # to the previous timestamp and remove the current timestamp
        sorted_timestamps = sorted(files_per_timestamp.keys())
        timestamps_to_remove = []
        for i in range(1, len(sorted_timestamps)):
            current_timestamp = sorted_timestamps[i]
            previous_timestamp = sorted_timestamps[i - 1]
            if (
                "dem_normal.png" in files_per_timestamp[current_timestamp]
                and "dem_output.png" in files_per_timestamp[previous_timestamp]
                and "dem_normal.png" not in files_per_timestamp[previous_timestamp]
                and "dem_output.png" not in files_per_timestamp[current_timestamp]
            ):
                files_per_timestamp[previous_timestamp]["dem_normal.png"] = (
                    files_per_timestamp[current_timestamp]["dem_normal.png"]
                )
                files_per_timestamp[previous_timestamp]["sat_normal.png"] = (
                    files_per_timestamp[current_timestamp]["sat_normal.png"]
                )
                timestamps_to_remove.append(current_timestamp)
        for timestamp in timestamps_to_remove:
            del files_per_timestamp[timestamp]

    sketch_dataset = SketchDataset("./user_sketches")
    dem_sketches: dict[str, np.ndarray] = {}
    land_sketches: dict[str, np.ndarray] = {}
    temp_sketches: dict[str, np.ndarray] = {}
    for i in range(len(sketch_dataset)):
        item = sketch_dataset.__getitem__(i)
        dem_sketches[item["name"]] = item["dem_sketch"]
        land_sketches[item["name"]] = gray_to_land(item["landcover_sketch"])
        temp_sketches[item["name"]] = item["temperature_sketch"]

    for timestamp in tqdm(files_per_timestamp):
        this_output_dir = output_dir
        if save_sketches or sort_only:
            to_save: dict[str, np.ndarray] = {}
            temp_sketch = np.zeros(sketch_shape, dtype=np.uint8)
            dem_sketch = np.zeros(sketch_shape, dtype=np.uint8)
            land_sketch = np.zeros((*sketch_shape, 3), dtype=np.uint8)
            modal_sketch = np.zeros((*sketch_shape, 3), dtype=np.uint8)
            river_weights = np.zeros(sketch_shape, dtype=np.float32)
            assert calculate_likely_sketch_name(
                dem_sketch,
                land_sketch,
                temp_sketch,
                dem_sketches,
                land_sketches,
                temp_sketches,
            ) == "Other"
            for sketch_name, alt_cmaps, target_sketch in zip(
                ["temp_sketch.png", "sketch.png", "land_sketch.png", "modal_sketch.png", "river_weights.tif"],
                [["coolwarm", "RdYlBu_r"], ["viridis", "gist_earth", "terrain"], [], [], ["viridis"]],
                [temp_sketch, dem_sketch, land_sketch, modal_sketch, river_weights],
            ):
                sketch = open_image_array(
                    files_per_timestamp[timestamp].get(sketch_name, "gar.png")
                )
                if sketch is None:
                    print(f"{timestamp}_{sketch_name} not found")
                    continue
                h, w = sketch.shape[:2]
                if w == 2 * h:
                    sketch = sketch / sketch.max() * 255
                elif w != h * 6:
                    print(f"{timestamp}_{sketch_name} not a quad sketch")
                    continue
                else:
                    sketch = quad_to_normal(
                        sketch, discrete=True
                    )
                sketch = cv2.resize(
                    sketch, sketch_shape[::-1], interpolation=cv2.INTER_NEAREST
                )
                target_sketch[:, :] = sketch[:, :]
                to_save[f"{timestamp}_normal_{sketch_name}"] = sketch
                for cmap in alt_cmaps:
                    mapped = np_rgb(sketch, cmap=cmap)
                    to_save[f"{timestamp}_normal_{sketch_name[:-4]}_{cmap}.png"] = mapped
            sketch_name = calculate_likely_sketch_name(
                dem_sketch,
                land_sketch,
                temp_sketch,
                dem_sketches,
                land_sketches,
                temp_sketches,
            )
            if sketch_name and do_sort:
                this_output_dir = os.path.join(output_dir, sketch_name)
                os.makedirs(this_output_dir, exist_ok=True)
            elif sort_by_timestamp:
                this_output_dir = os.path.join(output_dir, timestamp)
                os.makedirs(this_output_dir, exist_ok=True)

            if save_sketches:
                for name, sketch in to_save.items():
                    Image.fromarray(sketch.clip(0, 255).astype(np.uint8)).save(
                        os.path.join(this_output_dir, name)
                    )
        if sort_only:
            for suffix in files_per_timestamp[timestamp]:
                output_file = os.path.join(this_output_dir, f"{timestamp}_{suffix}")
                if os.path.exists(output_file):
                    continue
                shutil.copy(
                    files_per_timestamp[timestamp][suffix],
                    output_file,
                )
            continue

        if "args.json" in files_per_timestamp[timestamp]:
            with open(files_per_timestamp[timestamp]["args.json"], "r") as f:
                contents = f.read()
            with open(os.path.join(this_output_dir, f"{timestamp}_args.json"), "w") as f:
                f.write(contents)

        if (
            ("dem_normal.png" not in files_per_timestamp[timestamp] or force_dem) and
            "dem_output.png" in files_per_timestamp[timestamp]
        ):
            dem_output = cv2.imread(files_per_timestamp[timestamp]["dem_output.png"], cv2.IMREAD_UNCHANGED)
            dem_normal = quad_to_normal(dem_output)
            cv2.imwrite(os.path.join(this_output_dir, f"{timestamp}_dem_normal.png"), dem_normal)
            files_per_timestamp[timestamp]["dem_normal.png"] = os.path.join(
                this_output_dir, f"{timestamp}_dem_normal.png"
            )
        else:
            dem_normal = cv2.imread(files_per_timestamp[timestamp]["dem_normal.png"], cv2.IMREAD_UNCHANGED)
            cv2.imwrite(os.path.join(this_output_dir, f"{timestamp}_dem_normal.png"), dem_normal)

        if "sat_normal.png" not in files_per_timestamp[timestamp]:
            sat_output = open_image_array(files_per_timestamp[timestamp]["sat_output.png"])
            sat_normal = quad_to_normal(sat_output)
            Image.fromarray(sat_normal).save(
                os.path.join(this_output_dir, f"{timestamp}_sat_normal.png")
            )
            files_per_timestamp[timestamp]["sat_normal.png"] = os.path.join(
                this_output_dir, f"{timestamp}_sat_normal.png"
            )
        else:
            sat_normal = open_image_array(files_per_timestamp[timestamp]["sat_normal.png"])
            Image.fromarray(sat_normal).save(
                os.path.join(this_output_dir, f"{timestamp}_sat_normal.png")
            )

        do_river_viz = river_viz and (
            "dem_viz.png" not in files_per_timestamp[timestamp] or
            "sat_viz.png" not in files_per_timestamp[timestamp]
        )

        rivers_needed = (
            (
                "normal_rivers.png" not in files_per_timestamp[timestamp]
                and "rivers.png" in files_per_timestamp[timestamp]
            ) or (
                "oceanmask.png" not in files_per_timestamp[timestamp] and create_oceanmask
            ) or do_river_viz
        )
        if not rivers_needed:
            continue

        if "normal_rivers.png" in files_per_timestamp[timestamp]:
            normal_rivers = open_image_array(files_per_timestamp[timestamp]["normal_rivers.png"])
        else:
            if "rivers.png" in files_per_timestamp[timestamp]:
                quad_rivers = open_image_array(files_per_timestamp[timestamp]["rivers.png"]).astype(np.uint8)
                combined_rivers = quad_rivers[:, :, 0] * 256 ** 2 + quad_rivers[:, :, 1] * 256 + quad_rivers[:, :, 2]
                combined_rivers[
                    (quad_rivers[:, :, 0] == 68) &
                    (quad_rivers[:, :, 1] == 1) &
                    (quad_rivers[:, :, 2] == 84)
                ] = 0
            else:
                combined_rivers = np.zeros_like(dem_normal).astype(np.uint8)
            if combined_rivers.any():
                combined_rivers = cv2.dilate(
                    combined_rivers.astype(np.float32),
                    np.ones((3, 3), np.uint8),
                    iterations=1,
                ).astype(np.int32)
                combined_rivers = quad_to_normal(combined_rivers, discrete=True)
                river_mask = skeletonize(combined_rivers > 0) > 0
                combined_rivers[~river_mask] = 0
                normal_rivers = np.dstack([
                    combined_rivers // 256 ** 2,
                    (combined_rivers // 256) % 256,
                    combined_rivers % 256,
                ])
            else:

                print(f"No rivers found for {timestamp}")
                normal_rivers = np.zeros((*combined_rivers.shape, 3), dtype=np.uint8)
            normal_rivers[:, :, 0][combined_rivers == 0] = 68
            normal_rivers[:, :, 1][combined_rivers == 0] = 1
            normal_rivers[:, :, 2][combined_rivers == 0] = 84
            files_per_timestamp[timestamp]["normal_rivers.png"] = os.path.join(
                this_output_dir, f"{timestamp}_normal_rivers.png"
            )
            Image.fromarray(normal_rivers.clip(0, 255).astype(np.uint8)).save(
                files_per_timestamp[timestamp]["normal_rivers.png"]
            )

        gray_rivers = viridis_to_grayscale(normal_rivers)
        gray_rivers[
            (normal_rivers[:, :, 0] == 68) &
            (normal_rivers[:, :, 1] == 1) &
            (normal_rivers[:, :, 2] == 84)
        ] = 0

        if "oceanmask.png" not in files_per_timestamp[timestamp] and create_oceanmask:
            sat_normal = open_image_array(files_per_timestamp[timestamp]["sat_normal.png"]).astype(
                np.float32
            )
            dem_normal = open_image_array(files_per_timestamp[timestamp]["dem_normal.png"]).astype(
                np.float32
            )
            target_colour = np.array([2, 5, 20], dtype=np.float32)
            tolerance = 20.0
            water_distance = np.abs(
                np.dstack(
                    [
                        sat_normal[:, :, 0] - target_colour[0],
                        sat_normal[:, :, 1] - target_colour[1],
                        sat_normal[:, :, 2] - target_colour[2],
                    ]
                )
            )
            # Check if each channel is within tolerance
            within_tolerance = (water_distance <= tolerance).all(axis=2)

            mean_distance = water_distance.mean(axis=2)
            ocean_weight = (1.0 - mean_distance / tolerance).clip(0.0, 1.0) * 255
            # ocean_weight[dem_normal == 0] = 255
            ocean_weight[~within_tolerance] = 0
            ocean_weight = cv2.resize(
                ocean_weight, fx=2, fy=2, dsize=None, interpolation=cv2.INTER_CUBIC
            )
            # land_weight = 255 - ocean_weight
            # land_weight = cv2.GaussianBlur(land_weight, (7, 7), sigmaX=None)
            # ocean_weight = np.maximum(land_weight, ocean_weight) - land_weight
            ocean_weight = cv2.GaussianBlur(ocean_weight, (9, 9), sigmaX=None)

            h, w = ocean_weight.shape

            full_gray_rivers = cv2.resize(
                gray_rivers, (w, h), interpolation=cv2.INTER_NEAREST
            )

            full_gray_rivers = cv2.dilate(
                full_gray_rivers.astype(np.float32),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            )
            full_gray_rivers = cv2.GaussianBlur(full_gray_rivers, (9, 9), sigmaX=None)
            ocean_weight[full_gray_rivers > ocean_weight] = full_gray_rivers[full_gray_rivers > ocean_weight]
            Image.fromarray(ocean_weight.clip(0, 255).astype(np.uint8)).save(
                os.path.join(this_output_dir, f"{timestamp}_oceanmask.png")
            )

        if do_river_viz:
            if "river_weights.tif" in files_per_timestamp[timestamp]:
                river_weights = open_image_array(files_per_timestamp[timestamp]["river_weights.tif"])
                river_weights = river_weights / max(river_weights.max(), 1.0) * 255
            weights_viz = river_weights.copy()

            sat_viz = sat_normal.copy()
            H, W = sat_viz.shape[:2]

            weights_viz = cv2.resize(weights_viz, (W, H), interpolation=cv2.INTER_NEAREST)
            h, w = gray_rivers.shape[:2]
            if h < H:
                gray_rivers = cv2.resize(gray_rivers, (W, H), interpolation=cv2.INTER_NEAREST)
                gray_rivers = cv2.erode(
                    gray_rivers.astype(np.float32), np.ones((3, 3), np.uint8), iterations=1
                ).astype(np.uint8)
                normal_rivers = cv2.resize(
                    normal_rivers.clip(0, 255).astype(np.uint8), (W, H), interpolation=cv2.INTER_AREA
                )
            elif h > H:
                gray_rivers = cv2.dilate(
                    gray_rivers.astype(np.float32), np.ones((3, 3), np.uint8), iterations=1
                ).astype(np.uint8)
                gray_rivers = cv2.resize(gray_rivers, (W, H), interpolation=cv2.INTER_NEAREST)
                normal_rivers = cv2.resize(
                    normal_rivers.clip(0, 255).astype(np.uint8), (W, H), interpolation=cv2.INTER_AREA
                )

            sat_viz[gray_rivers > 0] = normal_rivers[gray_rivers > 0]
            dem_viz = np.dstack([dem_normal] * 3)  # TODO Maybe make this a different cmap
            weights_viz = np.dstack([weights_viz] * 3)
            dem_viz[gray_rivers > 0] = normal_rivers[gray_rivers > 0]
            weights_viz[gray_rivers > 0] = normal_rivers[gray_rivers > 0]
            Image.fromarray(sat_viz.clip(0, 255).astype(np.uint8)).save(
                os.path.join(this_output_dir, f"{timestamp}_sat_viz.png")
            )
            Image.fromarray(dem_viz.clip(0, 255).astype(np.uint8)).save(
                os.path.join(this_output_dir, f"{timestamp}_dem_viz.png")
            )
            Image.fromarray(weights_viz.clip(0, 255).astype(np.uint8)).save(
                os.path.join(this_output_dir, f"{timestamp}_weights_viz.png")
            )
            dem_viridis = np_rgb(dem_normal, cmap="viridis")
            Image.fromarray(dem_viridis.clip(0, 255).astype(np.uint8)).save(
                os.path.join(this_output_dir, f"{timestamp}_dem_viridis.png")
            )
