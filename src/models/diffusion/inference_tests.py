from copy import deepcopy
from dataclasses import replace
import os
import lpips
import torch
from time import time
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers.utils.torch_utils import randn_tensor
import math

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import wandb
from src.core.folder_dataset import load_args_config
from src.core.sketch_dataset import SketchDataset
from src.interface.utils import SketchArgs, derive_sketch_rivers
from src.models.diffusion.model import TerrainDiffusionPipeline

from planetAI.src.data.line_dataset import line_resizer
from planetAI.src.data.modal_sketch import ModalSketch
from src.models.diffusion.palette_dataset import PaletteDataset

from .inpaint_inference import InferenceInstance, InferenceArguments, get_device_generator
from .inference_controller import UpscalingInferenceController
from .inpaint_inference import run_inference_loop

from planetAI.src.data.utils import (
    PlanetConfig,
    image_grid,
    modal_resize,
    np_rgb,
    np_to_tensor,
    tensor_to_np,
)
from planetAI.src.data.rough_edge_mask import rough_edge_tile_mask
from planetAI.src.data.dataset import RAMDataset, array_variance
from planetAI.src.data.landcover_utils import gray_to_land


def get_random_tile(
    sketch: np.ndarray, shape: tuple[int, int], seed: int | None = None
) -> tuple[int, int]:
    """
    Generate a random tile from a sketch.
    """
    h, w = sketch.shape[:2]
    assert sketch.shape[0] >= shape[0] and sketch.shape[1] >= shape[1]
    while True:
        if seed is not None:
            np.random.seed(seed)
            seed = np.random.randint(0, 2**32)
        y = np.random.randint(0, sketch.shape[0] - shape[0])
        x = np.random.randint(0, sketch.shape[1] - shape[1])
        if w == 6 * h and x % h + shape[1] >= h:
            continue
        tile = sketch[y: y + shape[0], x: x + shape[1]]
        if np.sum(tile) > 0:
            break
    return y, x


def mse_loss(a: np.ndarray, b: np.ndarray) -> float:
    a_tensor = np_to_tensor(a).unsqueeze(0)
    b_tensor = np_to_tensor(b).unsqueeze(0)
    return torch.mean((a_tensor - b_tensor) ** 2).item()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_model = None


def lpips_distance(a: np.ndarray, b: np.ndarray) -> float:
    if any([x > 512 for x in a.shape]):
        return 0.0
    global lpips_model
    if lpips_model is None:
        lpips_model = lpips.LPIPS(net="vgg").to(device)
    a_tensor = np_to_tensor(a).unsqueeze(0).to(device)
    b_tensor = np_to_tensor(b).unsqueeze(0).to(device)
    with torch.no_grad():
        dist = lpips_model(a_tensor, b_tensor)
    return dist.item()


def format_losses(
    sat_mse: float,
    sat_lpips: float,
    dem_mse: float,
    dem_lpips: float,
    sat_only: bool = False,
    dem_only: bool = False,
) -> str:
    sat_loss = f"Sat MSE: {sat_mse:.4f}, LPIPS: {sat_lpips:.4f}"
    dem_loss = f"Dem MSE: {dem_mse:.4f}, LPIPS: {dem_lpips:.4f}"
    if sat_only:
        return sat_loss
    if dem_only:
        return dem_loss
    return f"{sat_loss} | {dem_loss}"


def print_losses(
    sat_generated: np.ndarray,
    downsat: np.ndarray,
    dem_generated: np.ndarray,
    downdem: np.ndarray,
) -> tuple[float, float, float, float]:
    sat_mse = mse_loss(sat_generated, downsat)
    dem_mse = mse_loss(dem_generated, downdem)
    sat_lpips = lpips_distance(sat_generated, downsat)
    dem_lpips = lpips_distance(dem_generated, downdem)
    print(format_losses(sat_mse, sat_lpips, dem_mse, dem_lpips))
    return sat_mse, sat_lpips, dem_mse, dem_lpips


def resize_river_tile(river_tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h1, w1 = shape
    if river_tile.shape[:2] == (h1, w1):
        return river_tile
    downrivers = line_resizer(river_tile, (w1, h1), thickness=1)
    rotation = 30
    r_h, r_w = downrivers.shape[:2]
    max_r_h = math.ceil(r_h * math.sqrt(2))
    max_r_w = math.ceil(r_w * math.sqrt(2))
    p_h = (max_r_h - r_h) // 2
    p_w = (max_r_w - r_w) // 2
    center = (max_r_w / 2, max_r_h / 2)
    downrivers = np.pad(downrivers, ((p_h, p_h), (p_w, p_w)))
    rot_mat = cv2.getRotationMatrix2D(center, rotation, 1.0)
    downrivers = cv2.warpAffine(downrivers, rot_mat, (max_r_w, max_r_h), flags=cv2.INTER_CUBIC)
    rot_mat = cv2.getRotationMatrix2D(center, -rotation, 1.0)
    downrivers = cv2.warpAffine(downrivers, rot_mat, (max_r_w, max_r_h), flags=cv2.INTER_CUBIC)
    downrivers = downrivers[p_h: p_h + r_h, p_w: p_w + r_w]
    return downrivers


def generate_and_save(
    model: TerrainDiffusionPipeline,
    batch: dict,
    timesteps: list[torch.Tensor],
    inference_instance: InferenceInstance,
    batch_size: int,
    shape: tuple[int, int],
    i: int,
    noisy_input: torch.Tensor,
    generator: torch.Generator,
    batch_time: float = 0.0,
):
    h, w = shape
    cond_images: torch.Tensor = batch["cond_image"]
    target_images: torch.Tensor = batch["target_image"]
    cond_images = cond_images.permute(0, 3, 1, 2) / 127.5 - 1.0
    target_images = target_images.permute(0, 3, 1, 2) / 127.5 - 1.0
    metadata = batch["metadata"]
    terrain_style = model.create_terrain_style(
        cond_images, cond_images, metadata
    )
    encoder_hidden_states = model._encode_terrain_style(
        terrain_style
    ).to(device)

    extra_step_kwargs = model.prepare_extra_step_kwargs(
        generator, 1.0
    )

    cond_images = cond_images.to(device)
    t1 = time()
    outputs, stats = run_inference_loop(
        model,
        noisy_input,
        cond_images,
        timesteps,
        guidance_channel=inference_instance.inference_args.guidance_channel,
        guidance_scale=inference_instance.inference_args.guidance_scale,
        encoder_hidden_states=encoder_hidden_states,
        extra_step_kwargs=extra_step_kwargs,
        batch_time=batch_time,
    )
    t2 = time()
    h, w = outputs.shape[-2:]
    full_stats = inference_instance.print_stat_summary(
        stats, total_time=t2 - t1 + batch_time, total_timesteps=len(timesteps), output_shape=(h, w)
    )
    output_display = inference_instance.planet_cfg.output_display(cond_images.cpu(), outputs.cpu(), target_images.cpu())
    for b in range(batch_size):
        images = [
            ("dem_output", outputs[b, 3]),
            ("sat_output", outputs[b, :3]),
            ("dem_real", target_images[b, 3]),
            ("sat_real", target_images[b, :3]),
            ("display", output_display[:, b * h: (b + 1) * h])
        ]
        inference_instance.planet_cfg.planet_seed = i + b

        name_suffix = None
        try:
            name_suffix = metadata.get("sample_name")
            if name_suffix is not None:
                name_suffix = name_suffix[b]
        except Exception:
            pass

        inference_instance.save_images(
            inference_instance.inference_args.output_folder,
            images,
            b,
            full_stats if b == 0 else {},
            name_suffix=name_suffix,
        )
    return outputs


class TileGenerator:
    def __init__(
        self,
        experiment_name: str,
        config: dict,
        output_size: int = 512,
        seed: int = 11,
        mars: bool = False,
        shuffle_landcover: bool = True,
        temp_variance: int = 100,
        do_upscaling: bool = True,
        num_images: int = 100,
        upload_results: bool = True,
        upload_only: bool = False,
        check_num_outputs: bool = True,
        size_offset: int = 0,
    ):
        self.experiment_name = experiment_name
        self.upload_results = upload_results
        self.upload_only = upload_only
        self.check_num_outputs = check_num_outputs

        # Configuration
        self.output_size = output_size
        self.seed = seed
        self.mars = mars
        self.shuffle_landcover = shuffle_landcover
        self.temp_variance = temp_variance
        self.do_upscaling = do_upscaling
        self.num_images = num_images
        self.use_previous = config["inference_args"]["use_previous"]
        self.resize_factor = config["experiment"]["resize_factor"]
        self.config = config

        self.planet_cfg = PlanetConfig(**config["planet_config"])
        self.upscale_cfg = PlanetConfig(**config["upscaling_config"])

        if size_offset != 0:
            self.planet_cfg = replace(
                self.planet_cfg,
                size=self.planet_cfg.size + size_offset,
                downscale_offset=self.planet_cfg.downscale_offset + size_offset,
            )
            self.upscale_cfg = replace(
                self.upscale_cfg,
                size=self.upscale_cfg.size + size_offset,
                downscale_offset=self.upscale_cfg.downscale_offset + size_offset,
            )

        self.inference_args = InferenceArguments(**config["inference_args"])
        self.upscaling_args = InferenceArguments(**config["upscaling_args"])

        # This makes sure that samples generated with different model versions are separated
        version_num = self.config["inference_args"].get("wandb_artifact_version")
        version_str = ""
        if version_num is not None:
            version_str = f"_{version_num}"

        self.inference_args = replace(
            self.inference_args, output_folder=self.inference_args.output_folder.rstrip("/") + version_str
        )
        self.upscaling_args = replace(
            self.upscaling_args, output_folder=self.upscaling_args.output_folder.rstrip("/") + version_str
        )

        self.output_shape = (self.output_size, self.output_size) if self.output_size is not None else None
        normal_size = round(
            int(self.output_size // self.upscale_cfg.delta / self.resize_factor)
        ) if self.output_size else None
        self.normal_shape = (
            normal_size,
            normal_size,
        ) if self.output_size is not None and self.do_upscaling else self.output_shape

        inference_instance = InferenceInstance(
            self.inference_args,
            self.planet_cfg,
            self.normal_shape,
        )
        upscaling_inference_instance = InferenceInstance(
            self.upscaling_args, self.upscale_cfg, output_shape=self.output_shape
        )

        self._dataset = None

        self.inference_controller = UpscalingInferenceController(
            inference_instance,
            upscaling_inference_instance,
        )

    @property
    def dataset(self):
        if self._dataset is not None:
            return self._dataset
        dataset_cfg = replace(self.upscale_cfg, image_mode="source")
        self._dataset = RAMDataset(
            dataset_cfg,
            dataset_cfg.output_channels(),
            dataset_cfg.input_channels(),
            normalise=True,
            tile_size=self.output_size,
            mode="test",
            do_transforms=False,
            shuffle_landcover=self.shuffle_landcover,
        )
        return self._dataset

    def generate_tiles(self):
        normal_shape = self.normal_shape
        if normal_shape is None:
            raise ValueError("output_size must be given when generating tiles")
        self.inference_args.tile_size = normal_shape[0]
        batch_size = self.upscaling_args.batch_size if self.do_upscaling else self.inference_args.batch_size

        generate_palette = self.config["experiment"].get("palette_test", False)

        dataset = PaletteDataset(
            self.planet_cfg,
            tile_size=normal_shape[0],
            do_transforms=False,
            samples_per_combination=8,
        ) if generate_palette else RAMDataset(
            self.planet_cfg,
            self.planet_cfg.output_channels(),
            self.planet_cfg.input_channels(),
            normalise=True,
            tile_size=normal_shape[0],
            mode="test",
            do_transforms=False,
            shuffle_landcover=self.shuffle_landcover,
        )

        if self.do_upscaling:
            upscaled_dataset = PaletteDataset(
                self.planet_cfg,
                tile_size=normal_shape[0],
                do_transforms=False,
                samples_per_combination=8,
            ) if generate_palette else RAMDataset(
                self.upscale_cfg,
                self.upscale_cfg.output_channels(),
                self.upscale_cfg.input_channels(),
                normalise=True,
                tile_size=self.output_size,
                mode="test",
                do_transforms=False,
                shuffle_landcover=self.shuffle_landcover,
            )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
        )
        inference_instance = self.inference_controller.normal_controller.inference_instance

        model = inference_instance.sat_pipeline

        generator = get_device_generator(None)

        timesteps = model.scheduler.timesteps

        already_generated = self.num_generated() if self.check_num_outputs else 0
        dataloader_iter = iter(dataloader)
        for i in tqdm(range(0, self.num_images, batch_size)):
            h, w = normal_shape
            noisy_input = randn_tensor(
                (batch_size, 4, h, w), device=device, generator=generator
            )
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)
            if i < already_generated:
                continue

            outputs = generate_and_save(
                model,
                batch,
                timesteps,
                inference_instance,
                batch_size,
                normal_shape,
                i,
                noisy_input,
                generator,
            )

            if self.do_upscaling:
                self.upscale_outputs(
                    batch_size, outputs, batch, upscaled_dataset, timesteps, i, generator, real_data=False
                )
                self.upscale_outputs(
                    batch_size, outputs, batch, upscaled_dataset, timesteps, i, generator, real_data=True
                )

    def upscale_outputs(
        self,
        batch_size: int,
        outputs: torch.Tensor,
        batch: dict,
        upscaled_dataset: RAMDataset,
        timesteps: list[torch.Tensor],
        i: int,
        generator: torch.Generator,
        real_data: bool,
    ):
        if self.output_shape is None:
            raise ValueError("output_size must be given when generating tiles")

        h, w = self.output_shape
        noisy_input = randn_tensor(
            (batch_size, 4, h, w), device=device, generator=generator
        )
        upscaling_instance = self.inference_controller.upscaling_controller.inference_instance
        upscaling_model = upscaling_instance.sat_pipeline
        next_batch: dict[str, list | dict] = {
            "cond_image": [],
            "target_image": [],
            "metadata": {}
        }
        for b, output in enumerate(outputs):
            meta = batch["metadata"]
            lat = meta["lat"][b]
            long = meta["long"][b]

            unbatched = upscaled_dataset.get_item_at_coords(
                float(lat),
                float(long),
                bool(meta["is_mars"][b]),
                bool(meta["is_summer"][b]),
                float(meta["angle"][b]),
                bool(meta["hflip"][b]),
                bool(meta["vflip"][b]),
            )
            cond_image = np_to_tensor(unbatched["cond_image"])
            if not real_data:
                output = F.interpolate(output.unsqueeze(0), scale_factor=self.resize_factor)[0]
                cond_image[:4] = F.interpolate(output.unsqueeze(0), scale_factor=self.upscale_cfg.delta)[0]
            cond_image = torch.tensor(tensor_to_np(cond_image))
            target_image = torch.tensor(unbatched["target_image"])
            next_batch["cond_image"].append(cond_image.unsqueeze(0))
            next_batch["target_image"].append(target_image.unsqueeze(0))
            for k in unbatched["metadata"]:
                next_batch["metadata"].setdefault(k, [])
                next_batch["metadata"][k].append(torch.tensor(unbatched["metadata"][k]).unsqueeze(0))
        next_batch["cond_image"] = torch.cat(next_batch["cond_image"])
        next_batch["target_image"] = torch.cat(next_batch["target_image"])
        for k in next_batch["metadata"]:
            next_batch["metadata"][k] = torch.cat(next_batch["metadata"][k])

        upscaling_instance.inference_args.real_data = real_data
        generate_and_save(
            upscaling_model,
            next_batch,
            timesteps,
            upscaling_instance,
            batch_size,
            self.output_shape,
            i,
            noisy_input,
            generator,
        )

    def generate_full(
        self,
        idx: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a random tile from real sketches.
        """
        item = self.dataset.get_item(idx % len(self.dataset))

        self.planet_cfg.planet_seed = idx
        self.upscale_cfg.planet_seed = idx

        condition = item["cond_image"]
        target = item["target_image"]
        # 'downland_sketch', 'downtemp_sketch', 'downsketch', 'river_mask'
        landcover = condition[:, :, 0].astype(np.uint8)
        temp = condition[:, :, 1]
        temp = array_variance(temp, self.temp_variance)
        sketch = condition[:, :, 2].astype(np.uint8)
        rivers = condition[:, :, 3]
        unique_vals = np.unique(sketch)
        unique_vals = unique_vals[unique_vals > 0]
        if len(unique_vals) > 0:
            sketch[((sketch == 0) & (landcover > 0))] = unique_vals[0]

        sat = target[:, :, :3]
        dem = target[:, :, 3]
        h2, w2 = sketch.shape[:2]
        if self.do_upscaling:
            h1 = int(h2 // self.upscale_cfg.delta / self.resize_factor)
            w1 = int(w2 // self.upscale_cfg.delta / self.resize_factor)
            h0 = h1 // self.planet_cfg.delta
            w0 = w1 // self.planet_cfg.delta
            full_delta = self.upscale_cfg.delta * self.planet_cfg.delta
        else:
            h0 = h2 // self.planet_cfg.delta
            w0 = w2 // self.planet_cfg.delta
            h1 = h2
            w1 = w2
            full_delta = self.planet_cfg.delta

        downsat = cv2.resize(sat, (w1, h1), interpolation=cv2.INTER_LANCZOS4)
        downdem = cv2.resize(dem, (w1, h1), interpolation=cv2.INTER_LANCZOS4)

        downland = modal_resize(landcover, full_delta)
        downtemp = cv2.resize(temp, (w0, h0), interpolation=cv2.INTER_LANCZOS4)
        downsketch = modal_resize(sketch, full_delta)

        downland = cv2.resize(downland, (w1, h1), interpolation=cv2.INTER_NEAREST)
        downtemp = cv2.resize(downtemp, (w1, h1), interpolation=cv2.INTER_NEAREST)
        downsketch = cv2.resize(downsketch, (w1, h1), interpolation=cv2.INTER_NEAREST)
        downrivers = resize_river_tile(rivers, (h1, w1))

        if self.mars:  # For now, TODO get mars sketches in get sketches
            landcover[:, :] = 255
            sketch[sketch == 0] = np.unique(sketch)[1]

        previous_mask = None
        if self.use_previous:
            h, w = self.output_shape
            previous_mask = np.ones((h, w), dtype=np.uint8)
            # Cut out a 256x256 tile from the center of the previous mask
            rough_mask = rough_edge_tile_mask((h - 256, w - 256), seed=idx)
            rough_mask = np.pad(rough_mask, 128)
            previous_mask[rough_mask] = 0
            self.inference_controller.normal_controller.inference_instance.previous_output = np_to_tensor(
                np.dstack([sat, dem])
            ).unsqueeze(
                0
            )

        # TODO Treat ocean as previous output

        current_guidance_scale = (
            self.inference_controller.normal_controller.inference_args.guidance_scale
        )
        self.inference_controller.normal_controller.inference_args.guidance_scale = 1.0
        sat_generated, dem_generated = self.inference_controller.generate_normal(
            SketchArgs(downsketch, downland, downtemp),
            rivers=downrivers,
            previous_mask=previous_mask,
        )

        self.inference_controller.normal_controller.inference_instance.save_images(
            self.inference_args.output_folder,
            [("sat_real", downsat), ("dem_real", downdem)]
        )

        sat_generated = tensor_to_np(sat_generated)
        dem_generated = tensor_to_np(dem_generated)

        losses = print_losses(sat_generated, downsat, dem_generated, downdem)

        self.inference_controller.normal_controller.inference_args.guidance_scale = (
            current_guidance_scale
        )

        guidance_sat_generated, guidance_dem_generated = (
            self.inference_controller.generate_normal(
                SketchArgs(downsketch, downland, downtemp),
                rivers=downrivers,
                previous_mask=previous_mask,
            )
        )

        self.inference_controller.normal_controller.inference_instance.save_images(
            self.inference_args.output_folder,
            [("sat_real", downsat), ("dem_real", downdem)]
        )

        guidance_sat_generated = tensor_to_np(guidance_sat_generated)
        guidance_dem_generated = tensor_to_np(guidance_dem_generated)

        guidance_losses = print_losses(
            guidance_sat_generated, downsat, guidance_dem_generated, downdem
        )

        # small_modal_sketch = ModalSketch(planet_cfg).get_sketch(downland, downtemp)
        # small_dem_sketch = downsketch
        up_sat = sat
        up_dem = dem

        top_row = [
            gray_to_land(condition[:, :, 0].astype(np.uint8)),
            gray_to_land(downland),
            downsketch,
            sat_generated,
            guidance_sat_generated,
            downsat,
        ]
        top_labels = [
            "Original landcover",
            "Shuffled Landcover sketch",
            "DEM sketch",
            f"Gen sat ({format_losses(*losses, sat_only=True)})",
            f"Gen sat (guidance) ({format_losses(*guidance_losses, sat_only=True)})",
            "Real small sat",
        ]
        bottom_row = [
            condition[:, :, 1],
            downtemp,
            rivers,
            dem_generated,
            guidance_dem_generated,
            downdem,
        ]
        bottom_labels = [
            "Original temperature",
            "Varied temperature",
            "Rivers",
            f"Gen dem ({format_losses(*losses, dem_only=True)})",
            f"Gen dem (guidance) ({format_losses(*guidance_losses, dem_only=True)})",
            "Real small dem",
        ]

        if self.do_upscaling:
            current_guidance_scale = (
                self.inference_controller.upscaling_controller.inference_args.guidance_scale
            )
            self.inference_controller.upscaling_controller.inference_args.guidance_scale = (
                1.0
            )

            up_sat, up_dem = self.inference_controller.upscale_last_output(
                SketchArgs(sketch, landcover, temp),
                rivers=rivers,
                previous_mask=previous_mask,
                resize_factor=self.resize_factor,
            )

            self.inference_controller.upscaling_controller.inference_instance.save_images(
                self.upscaling_args.output_folder,
                [("sat_real", sat), ("dem_real", dem)]
            )
            up_sat = tensor_to_np(up_sat)
            up_dem = tensor_to_np(up_dem)

            self.inference_controller.upscaling_controller.inference_args.guidance_scale = (
                current_guidance_scale
            )

            self.inference_controller.upscaling_controller.inference_args.real_data = False

            up_sat_guidance, up_dem_guidance = (
                self.inference_controller.upscale_last_output(
                    SketchArgs(sketch, landcover, temp),
                    rivers=rivers,
                    previous_mask=previous_mask,
                    resize_factor=self.resize_factor,
                )
            )
            up_sat_guidance = tensor_to_np(up_sat_guidance)
            up_dem_guidance = tensor_to_np(up_dem_guidance)

            upscale_losses = print_losses(up_sat, sat, up_dem, dem)

            upscale_guidance_losses = print_losses(
                up_sat_guidance, sat, up_dem_guidance, dem
            )

            self.inference_controller.upscaling_controller.inference_args.real_data = True

            real_up_sat, real_up_dem = self.inference_controller._upscale(
                downsat,
                downdem,
                SketchArgs(sketch, landcover, temp),
                rivers=rivers,
                previous_mask=previous_mask,
                resize_factor=self.resize_factor,
            )
            real_up_sat = tensor_to_np(real_up_sat)
            real_up_dem = tensor_to_np(real_up_dem)

            self.inference_controller.upscaling_controller.inference_instance.save_images(
                self.upscaling_args.output_folder,
                [("sat_real", sat), ("dem_real", dem)]
            )

            real_losses = print_losses(real_up_sat, sat, real_up_dem, dem)

            top_row += [
                up_sat,
                up_sat_guidance,
                real_up_sat,
                sat,
            ]
            top_labels += [
                f"Gen sat upscaled ({format_losses(*upscale_losses, sat_only=True)})",
                f"Gen sat upscaled (guidance) ({format_losses(*upscale_guidance_losses, sat_only=True)})",
                f"Real sat upscaled ({format_losses(*real_losses, sat_only=True)})",
                "Real sat",
            ]
            bottom_row += [
                up_dem,
                up_dem_guidance,
                real_up_dem,
                dem,
            ]
            bottom_labels += [
                f"Gen dem upscaled ({format_losses(*upscale_losses, dem_only=True)})",
                f"Gen dem upscaled (guidance) ({format_losses(*upscale_guidance_losses, dem_only=True)})",
                f"Real dem upscaled ({format_losses(*real_losses, dem_only=True)})",
                "Real dem",
            ]

        top_row = [
            cv2.resize(x, (w2, h2), interpolation=cv2.INTER_NEAREST) for x in top_row
        ]

        bottom_row = [
            np_rgb(
                cv2.resize(x, (w2, h2), interpolation=cv2.INTER_NEAREST),
                cmap="viridis" if i > 1 else "coolwarm",
            )
            for i, x in enumerate(bottom_row)
        ]

        pipeline_images = [
            Image.fromarray(x.clip(0, 255).astype(np.uint8))
            for x in top_row + bottom_row
        ]

        pipeline_grid = np.array(
            image_grid(
                pipeline_images,
                rows=2,
                cols=len(top_row),
                texts=top_labels + bottom_labels,
                size_multiplier=2,
                text_proportion=0.05,
            )
        )

        os.makedirs(
            os.path.join(self.planet_cfg.test_dir, "inference_pipelines"), exist_ok=True
        )
        self.inference_controller.normal_controller.inference_instance.save_images(
            os.path.join(self.planet_cfg.test_dir, "inference_pipelines"),
            [("upscaling_pipeline", pipeline_grid)],
        )

        modal_sketch = ModalSketch(self.planet_cfg).get_sketch(landcover, temp)
        dem_sketch = sketch
        return up_sat, up_dem, modal_sketch, dem_sketch

    def generate_real_sketches(
        self, folder: str, sketch_names: list[str], two_phase: bool, sketch_rivers: bool = True
    ) -> None:
        paths = [
            os.path.join(self.inference_controller.normal_controller.inference_args.output_folder, "final")
        ]
        if self.do_upscaling:
            paths.append(
                os.path.join(self.inference_controller.upscaling_controller.inference_args.output_folder, "final")
            )
        if self.upload_only:
            self.upload_artifact(paths)
            return

        dataset = SketchDataset(folder)

        use_filter = len(sketch_names) > 0
        found_names = set()

        for i in range(len(dataset)):
            sketches = dataset.__getitem__(i)
            name = sketches["name"]
            found_names.add(name)
            if name not in sketch_names and use_filter:
                print(f"Skipping {name}")
                continue
            print(f"Generating {name}")
            dem_sketch = sketches["dem_sketch"]
            # landcover_sketch = randomize_land(sketches["landcover_sketch"])
            landcover_sketch = sketches["landcover_sketch"]
            temperature_sketch = sketches["temperature_sketch"]
            river_sketch = sketches["river_sketch"]

            sketches = SketchArgs(
                dem_sketch, landcover_sketch, temperature_sketch
            )
            if river_sketch is None and sketch_rivers:
                river_sketch = derive_sketch_rivers(
                    sketches,
                    self.planet_cfg,
                    resolution_scale=0.5,
                )

            self.inference_controller.generate_normal(
                sketches,
                river_sketch,
            )
            if (river_sketch is None and self.do_upscaling) or two_phase:
                last_dem = self.inference_controller.last_normal_outputs[1]
                river_sketch = self.inference_controller._get_rivers(
                    last_dem,
                    sketches,
                    river_scaling_factor=1.0,
                )
            if two_phase:
                self.inference_controller.generate_normal(
                    sketches,
                    river_sketch,
                )

            if self.do_upscaling:
                self.inference_controller.upscale_last_output(
                    sketches,
                    river_sketch,
                    resize_factor=self.resize_factor,
                )

        missing_names = set(sketch_names) - found_names
        if len(missing_names) > 0:
            print("The following sketches were not found:", ",".join(missing_names))


        if self.upload_results:
            self.upload_artifact(paths)

    def num_generated(self) -> int:
        output_folder = os.path.join(self.inference_args.output_folder, "final")
        os.makedirs(output_folder, exist_ok=True)
        files = os.listdir(output_folder)
        json_files = list(filter(lambda x: ".json" in x, files))
        json_files = sorted(json_files)

        if len(json_files) == 0:
            return 0
        args_dict = load_args_config(os.path.join(output_folder, json_files[-1]))

        # This is technically not completely correct but is good enough if we keep the seed fixed
        return int(args_dict["planet_cfg"].get("planet_seed", 0)) - self.seed

    def upload_artifact(self, paths: list[str]):
        artifact = wandb.Artifact(name=f"{self.experiment_name}_inference", type="dataset")
        for path in paths:
            artifact.add_dir(path)
        wandb.log_artifact(artifact)
        artifact.wait()
        print("Done uploading")

    def generate_grid(self) -> None:
        already_generated = self.num_generated() if self.check_num_outputs else 0
        if already_generated >= self.num_images:
            return

        if not self.upload_only:
            if (
                self.inference_args.tile_size == self.output_size
                or (self.do_upscaling and self.upscaling_args.tile_size == self.output_size)
            ):
                self.generate_tiles()

            else:
                print("Generating full sized tile because tile size does not match output size")
                for i in tqdm(range(already_generated, self.num_images)):
                    self.generate_full(self.seed + i)
        paths = [
            os.path.join(self.inference_controller.normal_controller.inference_args.output_folder, "final")
        ]
        if self.do_upscaling:
            paths.append(
                os.path.join(self.inference_controller.upscaling_controller.inference_args.output_folder, "final")
            )
        if self.upload_results or self.upload_only:
            self.upload_artifact(paths)


def merge_config(default_config: dict, experiment_config: dict) -> dict:
    default_config = deepcopy(default_config)
    for cfg_type in experiment_config:
        for k in experiment_config[cfg_type]:
            default_config[cfg_type][k] = experiment_config[cfg_type][k]
    return default_config


def recursive_merge(default_config: dict, all_configs: dict, experiment_name: str) -> dict:

    current_config = all_configs[experiment_name]
    parent_name = current_config.get("experiment", {}).get("parent")

    if parent_name is None or parent_name == experiment_name:
        return merge_config(default_config, current_config)

    parent_config = recursive_merge(default_config, all_configs, parent_name)

    return merge_config(parent_config, current_config)


def _generate_configs(
    experiment_config: dict[str, str | float | bool | int | list[str | float | bool | int]]
) -> list[dict]:
    """
    Recurse through a config and create new configs with the lists unrolled.
    """
    if not any([isinstance(val, list) for val in experiment_config.values()]):
        return [experiment_config]
    to_return = []
    for key, val in experiment_config.items():
        if isinstance(val, list):
            for item in val:
                next_dict = experiment_config.copy()
                next_dict[key] = item
                to_return.extend(_generate_configs(next_dict))
            break
    return to_return


def generate_configs(
    experiment_config: dict[str, dict[str, str | float | bool | int | list[str | float | bool | int]]]
) -> list[dict]:
    """
    Recurse through a config and create new configs with the lists unrolled.
    """
    configs = [{}]
    for key, val in experiment_config.items():
        vals = _generate_configs(val)
        next_configs = []
        for cfg in configs:
            for next_val in vals:
                next_cfg = cfg.copy()
                next_cfg[key] = next_val
                next_configs.append(next_cfg)
        configs = next_configs
    return configs


test_cfgs = _generate_configs({
    "key1": [0, 1],
    "key2": [0, 1],
    "key3": 0
})
assert test_cfgs == [
    {"key1": 0, "key2": 0, "key3": 0},
    {"key1": 0, "key2": 1, "key3": 0},
    {"key1": 1, "key2": 0, "key3": 0},
    {"key1": 1, "key2": 1, "key3": 0},
]

big_test_cfgs = generate_configs({
    "key1": {"key1": [0, 1]},
    "key2": {"key2": [0, 1]},
    "key3": {"key3": 0}
})
assert big_test_cfgs == [
    {"key1": {"key1": 0}, "key2": {"key2": 0}, "key3": {"key3": 0}},
    {"key1": {"key1": 0}, "key2": {"key2": 1}, "key3": {"key3": 0}},
    {"key1": {"key1": 1}, "key2": {"key2": 0}, "key3": {"key3": 0}},
    {"key1": {"key1": 1}, "key2": {"key2": 1}, "key3": {"key3": 0}},
]
