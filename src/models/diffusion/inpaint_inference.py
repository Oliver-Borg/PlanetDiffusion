import json
from dataclasses import dataclass, field, asdict, replace
import os
from threading import Thread
import datetime
import logging
import gc
import sys
from typing import Callable, Optional
import psutil


import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from diffusers.schedulers import DDIMScheduler
from tqdm import tqdm
from PIL import Image
from skimage.morphology import skeletonize


from PIL import Image as img

from planetAI.src.data.uncertainty_sketch import UncertaintySketcher

if os.getenv("MEM_PROFILE", "0") == "1":
    try:
        from memory_profiler import profile as profile
    except Exception:
        from planetAI.src.data.utils import profile as profile
else:
    try:
        from line_profiler import profile
    except Exception:
        from planetAI.src.data.utils import profile


import cv2

from .model import TerrainDiffusionPipeline
from ...core.dataclass_argparser import CustomArgumentParser
from .quad_tree_mask import QuadTreeMaskWrapper

from planetAI.src.data.utils import (
    PlanetConfig,
    np_rgb,
    timing,
    tensor_to_np,
    MaskStore,
)
from planetAI.src.data.landcover_utils import (
    LandcoverClasses,
    translate_land,
    gray_to_land,
)
from planetAI.src.data.dataset import (
    NormaliseTransform,
    _simple_encode,
    EncoderOverride,
    upscale_regularization,
)
from planetAI.src.data.sphere_mapping import QuadSphere
from planetAI.src.data.sketch_gen import get_strahler_orders
from planetAI.src.data.modal_sketch import ModalSketch
from planetAI.src.data.river_modal_sketch import RiverModalSketch
from planetAI.src.data.rough_edge_mask import jigsaw_piece_mask, rough_edge_tile_mask
from diffusers.utils.torch_utils import randn_tensor
from random import randint
from ...labelling.encoding import PlanetEncoder


from time import time, sleep

logging.basicConfig(level=logging.INFO)
live_save_enabled = True

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("Cuda is not available. Using cpu.")


@dataclass
class GenerationArguments:
    diffusion_model_dir: str = field(
        default="/mnt/e/models/Sat-no-inpaint",
        metadata={"help": "Directory of diffusion model"},
    )
    wandb_artifact_version: Optional[int] = field(
        default=None, metadata={"help": "Version for the artifact"}
    )

    seed: int = field(
        default=None,
        metadata={
            "help": "Seed for reproducibility",
            "mutable": True,
            "min": 0,
            "max": 1000,
        },
    )
    timesteps: int = field(
        default=25,
        metadata={
            "help": "Number of timesteps used for sampling",
            "mutable": True,
            "min": 5,
            "max": 1000,
            "step": 5,
        },
    )
    guidance_scale: float = field(
        default=1.0,
        metadata={
            "help": "Guidance scale for classifier-free guidance",
            "mutable": True,
            "min": 0.0,
            "max": 5.0,
            "step": 0.1,
        },
    )
    guidance_channel: Optional[int] = field(
        default=None,
        metadata={
            "help": "Guidance channel for guidance",
            "mutable": False,  # TODO Think about doing this better
        },
    )

    use_fp16: bool = field(
        default=False, metadata={"help": "Run in fp16 mode", "mutable": True}
    )

    enable_attention_slicing: bool = field(
        default=False,
        metadata={
            "help": "Whether to enable attention slicing (more memory efficient, but slower)",
            "mutable": False,  # TODO Implement this
        },
    )

    attention_slice_size: str = field(
        default="auto",
        metadata={"help": "If `enable_attention_slicing`, use this as the slice_size"},
    )

    normalise_outputs: bool = field(
        default=False,
        metadata={
            "help": "Whether to normalise outputs",
            "mutable": False,  # TODO Implement this
        },
    )

    batch_size: int = field(
        default=8,
        metadata={
            "help": "Batch size for inference",
            "mutable": True,
            "min": 1,
            "max": 16,
        },
    )


@dataclass
class InferenceArguments(GenerationArguments):
    input_folder: str = field(
        default="./data/evaluation/inputs", metadata={"help": "Path to input folder"}
    )
    output_folder: str = field(
        default="./data/evaluation/outputs", metadata={"help": "Path to output folder"}
    )
    filename: str = field(
        default="sketch.png", metadata={"help": "Filename of the sketch"}
    )
    real_data: bool = field(
        default=False, metadata={"help": "Whether to use real data"}
    )
    loss_threshold: float = field(default=1.0, metadata={"help": "Threshold for loss"})
    use_water_mask: bool = field(
        default=True,
        metadata={
            "help": "Whether to use water mask to provide extra context",
            "mutable": False,
        },
    )
    erode_masks: bool = field(
        default=True,
        metadata={
            "help": "Whether to erode the water and previous masks",
            "mutable": False,
        },
    )  # TODO Add more options for number of erosion steps and kernel size
    compile_model: bool = field(
        default=False,
        metadata={
            "help": "Compile the model before running inference",
            "mutable": sys.platform == "linux",  # Not supported on windows
        },
    )
    sketch_injection: bool = field(
        default=True,
        metadata={"help": "Use noised sketch as starting point", "mutable": True},
    )
    guide_tiles: int = field(
        default=0,
        metadata={"help": "Pre-generate a given number of tiles to use as guidance"},
    )
    grid_align: int = field(
        default=1,
        metadata={
            "help": "Grid alignment for generation",
            "mutable": True,
            "min": 1,
            "max": 192,
            "step": 1,
        },
    )
    tile_size: int = field(
        default=256,
        metadata={
            "help": "Tile size for generation",
            "mutable": True,
            "min": 64,
            "max": 1024,
            "step": 64,
        },
    )
    use_previous: bool = field(
        default=False,
        metadata={"help": "Use previous mask for generation", "mutable": True},
    )
    do_upscaling: bool = field(
        default=False,
        metadata={"help": "Use upscaling model on previous output.", "mutable": False},
    )
    use_jigsaw_grid: bool = field(
        default=False, metadata={"help": "Use jigsaw grid for generation"}
    )
    use_rough_edge_mask: bool = field(
        default=True,
        metadata={"help": "Use rough edge mask for generation", "mutable": True},
    )
    statistics_dir: str = field(
        default=".", metadata={"help": "Directory to save statistics"}
    )
    save_outputs: bool = field(
        default=True,
        metadata={"help": "Save outputs after generation", "mutable": True},
    )
    dual_model_upscale_weight: float = field(
        default=1.0,
        metadata={
            "help": "The weight to use for the upscaling model if dual models is on"
        },
    )
    dual_model_dir: str = field(
        default="/mnt/e/models/Planet-size-5-fine-tuned",
        metadata={"help": "Directory of diffusion model for dual model inference"},
    )

    def __str__(self) -> str:
        string = "GenerationArguments:\n"
        fields = self.__dataclass_fields__

        max_key_length = 0
        max_value_legth = 0
        for name, f in fields.items():
            value = getattr(self, name)
            max_key_length = max(max_key_length, len(name))
            max_value_legth = max(max_value_legth, len(str(value)))

        for name, f in fields.items():
            # Print the name, value and if it has been modified from the default
            value = getattr(self, name)
            default = f.default
            name_str = f"{name:<{max_key_length}}"
            value_str = f"{str(value):<{max_value_legth}}"
            if value != default:
                string += (
                    f"  {name_str}: {value_str} (modified from default {default})\n"
                )
            else:
                string += f"  {name_str}: {value_str}\n"
        return string


input_transform = transforms.Compose([transforms.ToTensor(), NormaliseTransform()])


class InferenceDataset(Dataset):
    """
    This class returns a valid latitude, longitude pair and quad tile coordinates
    for a tile that has not been generated yet.
    It uses a generated mask and a sphere mapping to find out if the tile has been generated.
    """

    @profile
    def __init__(
        self,  # TODO Remove fake optional params
        generated_mask: np.ndarray,
        current_output: torch.Tensor,
        stacked_sketch: np.ndarray,
        planet_cfg: PlanetConfig,
        inference_args: InferenceArguments,
        device="cuda",
        river_mask: np.ndarray = None,
        water_mask: np.ndarray = None,
        scheduler: DDIMScheduler | None = None,
        t: int = 0,
        first_timestep: bool = False,
        seed: int = 0,
        grid_align: int = 1,
        tile_width: int = 256,
        encoder_override: EncoderOverride = EncoderOverride(),
        previous_output: torch.Tensor = None,
        previous_mask: np.ndarray = None,
        downsat: np.ndarray | None = None,
        downdem: np.ndarray | None = None,
    ):
        # TODO: Maintain down and up versions of quad generated mask and atlas generated
        self.generated_mask = generated_mask
        self.H = generated_mask.shape[0]
        self.W = generated_mask.shape[1]
        self.delta: int = 2**planet_cfg.downscale_offset
        self.current_output = current_output
        self.previous_output = previous_output
        self.previous_mask = previous_mask
        self.inference_args = inference_args
        self.do_upscaling = inference_args.do_upscaling
        self.use_previous = previous_mask is not None and previous_output is not None
        if self.do_upscaling and (downsat is None or downdem is None):
            raise ValueError("do_upscaling is True but downsat is None or downdem is None")
        if self.do_upscaling:
            assert downsat.shape[:2] == (self.H, self.W), (
                f"downsat shape {downsat.shape} must match generated mask shape {(self.H, self.W)}"
            )
            assert downdem.shape[:2] == (self.H, self.W), (
                f"downdem shape {downdem.shape} must match generated mask shape {(self.H, self.W)}"
            )
            self.downsat = downsat
            self.downdem = downdem

        if self.previous_output is None:
            self.previous_output = self.current_output
        if self.previous_mask is None:
            self.previous_mask = np.zeros_like(self.generated_mask)
        assert self.previous_output.shape == self.current_output.shape
        assert self.previous_mask.shape == self.generated_mask.shape
        self.generated_mask |= self.previous_mask > 0
        self.generated_mask_wrapper = QuadTreeMaskWrapper(
            self.generated_mask,
            num_levels=(
                None
                if self.use_previous or (self.H <= 4096 and self.W <= 4096)
                else planet_cfg.size + 2
            ),
        )
        self.sketch = stacked_sketch[:, :, 0]
        self.land_sketch = stacked_sketch[:, :, 1]
        self.temp_sketch = stacked_sketch[:, :, 2]
        self.tile_width = tile_width
        self.encoder_override = encoder_override
        self.use_rough_edge_mask = (
            self.inference_args.use_rough_edge_mask
            and not (self.tile_width, self.tile_width) == self.generated_mask.shape
        )

        self.planet_cfg = planet_cfg
        self.modal_sketch = ModalSketch(self.planet_cfg)
        self.river_modal_sketch = RiverModalSketch(replace(self.planet_cfg, size=5))
        self.device = device
        self.first_timestep = first_timestep
        self.noise_tiling = True

        self.river_mask = river_mask

        if self.river_mask.shape != self.generated_mask.shape:
            self.river_mask = cv2.resize(self.river_mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        if self.planet_cfg.discrete_rivers:
            mask = self.river_mask > 5
            mask = skeletonize(mask)
            self.river_mask[~mask] = 0

        self.grid_align = grid_align
        self.mask_changed = False

        self.ocean_tile = get_full_ocean_tensor((self.tile_width, self.tile_width, 4))
        self.scheduler = scheduler
        self.water_mask = water_mask
        self.seed = seed
        self.noise_tile = self.get_noise_tile(
            self.tile_width, self.tile_width, self.seed
        )
        self.t = t
        # For some reason this produces really poor results for unknown reasons
        # if self.first_timestep and not self.noise_tiling:
        #     current_output = self.get_noise_tile(self.H, self.W, self.seed)

    @timing
    def get_noise_tile(self, h: int, w: int, seed: int | None = None):
        return randn_tensor(
            (1, 4, h, w), device=None, generator=get_cpu_generator(seed)
        )

    def inject_sketches(
        self, modal_sketch: np.ndarray, sketch: np.ndarray, current_output: torch.Tensor
    ) -> torch.Tensor:
        # TODO Move this to be on the first step in dataloader too
        full_tensor = input_transform(
            np.dstack([modal_sketch, sketch]) / 255
        ).unsqueeze(0)
        # TODO: This is maybe just supposed to be timesteps[0]
        current_output = self.scheduler.add_noise(
            full_tensor, current_output, torch.tensor(999)
        )
        return current_output[0]

    def __len__(self):
        return 100000000

    @profile
    def get_coords(self):
        attempts = 0
        valid_tile = False
        reinit_count = 0
        while not valid_tile:
            u_i, v_i = self.generated_mask_wrapper.get_false_coord()
            if u_i is None or v_i is None:
                if reinit_count > 0:
                    us, vs = np.where(self.generated_mask == 0)
                    u_i = us[0]
                    v_i = vs[0]
                elif not self.generated_mask.all():
                    # TODO Investigate why the quad mask says we are done when the actual mask is not done
                    reinit_count += 1
                    print("Having to reinitialise mask")
                    self.generated_mask_wrapper.init_levels()
                    self.generated_mask_wrapper.init_tiles()
                    continue
                else:
                    u_i, v_i = self.H // 2, 3 * self.H // 2
            if (
                attempts < 100
            ):  # Allow the last few tiles to be non grid aligned to make sure we don't get stuck forever
                u_i = u_i // self.grid_align * self.grid_align
                v_i = v_i // self.grid_align * self.grid_align
            u_i = u_i % self.H
            v_i = v_i % self.W
            try:
                us, vs = self._get_coords((u_i, v_i))
            except NotImplementedError:
                continue
            valid_tile = True
        if attempts > 10:
            logging.warning("Attempts:", attempts)
        mask = np.ones_like(us, dtype=bool)
        if self.use_rough_edge_mask and self.t <= 100:
            edge_mask = rough_edge_tile_mask((self.tile_width, self.tile_width))
            if (edge_mask & ~self.generated_mask[us, vs]).any():
                mask = edge_mask

        return us, vs, mask

    def _get_coords(self, center: tuple[int, int]):
        u_i, v_i = center

        # Shift point to other corners
        # if u_i > self.H // 2:
        #     u_i -= 255
        # if v_i > self.W // 2:
        #     v_i -= 255

        if self.W == self.H * 6:
            raise ValueError(
                "Quadsphere atlas detected. Use QuadInferenceDataset instead"
            )
        f_w = self.H
        face = v_i // f_w
        h_t_w = self.tile_width // 2
        do_clip = (self.tile_width, self.tile_width) == self.generated_mask.shape
        if do_clip:
            u_i = np.clip(u_i, h_t_w, self.H - h_t_w)
            v_i = np.clip(v_i % f_w, h_t_w, f_w - h_t_w) + face * f_w

        # For now we are just wrapping around the image
        us = np.arange(u_i - h_t_w, u_i + h_t_w) % self.H
        vs = np.arange(v_i - h_t_w, v_i + h_t_w) % self.W
        vs, us = np.meshgrid(vs, us)
        return us, vs

    @profile
    def __getitem__(self, idx):
        # We don't actually use idx
        us, vs, new_pixel_mask = self.get_coords()
        ys = us.astype(np.int32)
        xs = vs.astype(np.int32)
        previous_mask = self.previous_mask[ys, xs]
        self.mask_changed = True
        # x, y, z = self.quad_sphere.quad_coords_to_surface_coords(u_i, v_i)
        # lat, long = self.quad_sphere.surface_coords_to_coords(x, y, z)

        # TODO Why don't we use seed?
        # seed = self.seed + idx + int(self.t) if self.seed else None

        if self.first_timestep and self.noise_tiling:
            _noise = self.get_noise_tile(self.tile_width, self.tile_width, seed=None)
        else:
            _noise = self.current_output[
                :, :, ys, xs
            ]  # .clone() TODO Check if this is necessary

        # Don't use previous output for the last step to avoid artifacts
        if previous_mask.any() and self.t > 0:
            _previous_output = self.previous_output[:, :, ys, xs]
            # Add t steps of noise to the previous output
            _previous_noisy: torch.tensor = self.scheduler.add_noise(
                _previous_output, self.noise_tile, torch.tensor(int(self.t))
            ).to(_noise.dtype)
            _ys, _xs = np.where(previous_mask)
            _noise[:, :, _ys, _xs] = _previous_noisy[:, :, _ys, _xs]

        _water_mask = self.water_mask[ys, xs]
        _mask = np.zeros((self.tile_width, self.tile_width), np.uint8)
        _mask += 255
        dts = self.tile_width // self.delta
        # _downland_sketch = modal_resize(self.land_sketch[ys, xs], self.delta)
        # _downtemp_sketch = modal_resize(self.temp_sketch[ys, xs], self.delta)
        # _downsketch = modal_resize(self.sketch[ys, xs], self.delta)

        # We don't actually need to modal resize because it has already been resized
        _downland_sketch = cv2.resize(
            self.land_sketch[ys, xs], (dts, dts), interpolation=cv2.INTER_NEAREST
        )
        _downland_sketch = translate_land(_downland_sketch)
        _rgb_downland_sketch = gray_to_land(_downland_sketch)
        _downtemp_sketch = cv2.resize(
            self.temp_sketch[ys, xs], (dts, dts), interpolation=cv2.INTER_NEAREST
        )
        _downsketch = cv2.resize(
            self.sketch[ys, xs], (dts, dts), interpolation=cv2.INTER_NEAREST
        )
        _downmodal = self.modal_sketch.get_sketch(
            _downland_sketch, _downtemp_sketch, mars_mask=(_downland_sketch == 255)
        )

        # Resize these to self.tile_widthxself.tile_width
        _downland_sketch = cv2.resize(
            _downland_sketch,
            (self.tile_width, self.tile_width),
            interpolation=cv2.INTER_NEAREST,
        )
        _rgb_downland_sketch = cv2.resize(
            _rgb_downland_sketch,
            (self.tile_width, self.tile_width),
            interpolation=cv2.INTER_NEAREST,
        )
        _downtemp_sketch = cv2.resize(
            _downtemp_sketch,
            (self.tile_width, self.tile_width),
            interpolation=cv2.INTER_NEAREST,
        )
        _downsketch = cv2.resize(
            _downsketch,
            (self.tile_width, self.tile_width),
            interpolation=cv2.INTER_NEAREST,
        )
        _downmodal = cv2.resize(
            _downmodal,
            (self.tile_width, self.tile_width),
            interpolation=cv2.INTER_NEAREST,
        )
        _rivers = np.zeros_like(_downsketch)
        if self.river_mask is not None:
            _rivers = self.river_mask[ys, xs]
            # if _rivers.any():
            #     _downmodal_rivers = self.river_modal_sketch.get_sketch(_downland_sketch, _downtemp_sketch, _rivers)
            #     _downmodal[_rivers > 0] = _downmodal_rivers[_rivers > 0]

        cond_images: dict[str, np.ndarray] = {
            "downmodal": _downmodal,
            "downsketch": _downsketch,
            "downland_sketch": (
                _rgb_downland_sketch
                if self.planet_cfg.rgb_landcover
                else _downland_sketch
            ),
            "downtemp_sketch": _downtemp_sketch,
            "river_upa": _rivers,
        }

        if self.do_upscaling:
            cond_images["downsat"] = upscale_regularization(
                self.downsat[ys, xs], self.planet_cfg, self.tile_width, self.delta
            )
            cond_images["downdem"] = upscale_regularization(
                self.downdem[ys, xs], self.planet_cfg, self.tile_width, self.delta
            )

        metadata = {
            "tile_y": 0,
            "tile_x": 0,
            "k": 0,
            "hflip": 0,
            "vflip": 0,
            "tile_size": self.tile_width,
            "mask_channel": 4,
            "delta": self.delta,
            "angle": 0,
        }
        metadata["embedding"] = _simple_encode(
            _downsketch,
            _downland_sketch,
            _downtemp_sketch,
            self.planet_cfg,
            self.planet_cfg.delta,
            rivers=_rivers,
            sat=_downmodal,
            encoder_override=self.encoder_override,
        )
        noisy_input = _noise[0]

        if self.first_timestep and self.inference_args.sketch_injection:
            noisy_input = self.inject_sketches(
                cond_images["downmodal"], cond_images["downsketch"], noisy_input
            ).to(_noise.dtype)
        cond_images = {
            k: input_transform(v.astype(np.float32) / 255.0)
            for k, v in cond_images.items()
        }
        # TODO It can cause issues having the noisy ocean tile here because
        # when it is eroded, pieces that were previously ocean (and therefore not generated)
        # will then become parts of the "land", meaning that they will be generated.
        # The problem is that the new "land" has not been denoised at all yet
        # so later steps won't be able to fully denoise it
        _noisy_ocean_tile = self.scheduler.add_noise(
            self.ocean_tile, self.noise_tile, torch.tensor(int(self.t))
        )
        noisy_input[0:4, _water_mask] = _noisy_ocean_tile.to(noisy_input.dtype)[0, 0:4, _water_mask]
        # metadata['embedding'] = np.zeros_like(metadata['embedding'])
        new_pixels = (new_pixel_mask & ~self.generated_mask[ys, xs]).sum()
        self.generated_mask[ys, xs] |= new_pixel_mask
        return {
            "cond_images": cond_images,
            "noisy_input": noisy_input,
            "metadata": metadata,
            "ys": ys,
            "xs": xs,
            "new_pixel_mask": new_pixel_mask,
            "new_pixels": new_pixels,
        }


def get_model_condition(
    cond_images: dict[str, torch.Tensor],
    input_channels: list[str],
    inject_rivers: bool = False,
) -> torch.Tensor:
    if inject_rivers:
        river_upa = cond_images["river_upa"]
        for image_name, cond_image in cond_images.items():
            if "dem" in image_name or image_name == "downsketch" or image_name == "sketch":
                cond_image[river_upa > 0] = river_upa[river_upa > 0]
    images = [cond_images[k] for k in input_channels]
    return torch.cat(images, dim=1)


class QuadInferenceDataset(InferenceDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quad_sphere = QuadSphere(shape=(self.planet_cfg.H, self.planet_cfg.W))

    def _get_coords(self, center: tuple[int, int]):
        u_i, v_i = center
        us, vs = self.quad_sphere.get_quad_tile_mapping(
            uv=(u_i, v_i), tile_size=self.tile_width
        )
        return us, vs


class FastInferenceDataset(InferenceDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.down_generated_mask = (
            cv2.resize(
                self.generated_mask.astype(np.uint8) * 255,
                (self.W // self.delta, self.H // self.delta),
                interpolation=cv2.INTER_AREA,
            )
            == 255
        )

    @profile
    def get_coords(self):
        # WIP Use a hierarchical approach
        # Use small generated mask
        # Randomly select ungenerated coord y0, x0
        # Y0 = y0 * delta
        # X0 = x0 * delta
        # Then we can extract this mask from the full generated mask and select a Y0 and X0 from this mask
        # If mask spans Y0 to Y1 and X0 to X1
        # and if y0 = Y0 // delta etc
        # Then we know for sure that y0 + 1 to y1 - 1 and x0 + 1 to x1 - 1 are all generated

        # 1. Get a random ungenerated tile from the downsampled mask
        while True:
            ys, xs = np.where(self.down_generated_mask == 0)
            idx = np.random.choice(len(ys))
            y, x = ys[idx], xs[idx]
            Y = y * self.delta
            X = x * self.delta
            try:
                us, vs = self._get_coords((Y, X))
            except NotImplementedError:
                continue
            generated_mask_tile = self.generated_mask[us, vs]
            ys, xs = np.where(generated_mask_tile == 0)
            ys += Y - 128
            xs += X - 128
            if len(ys) == 0:
                down_us, down_vs = us // self.delta, vs // self.delta
                down_us = np.clip(down_us, down_us.min() + 1, down_us.max() - 1)
                down_vs = np.clip(down_vs, down_vs.min() + 1, down_vs.max() - 1)
                self.down_generated_mask[down_us, down_vs] = True
                continue
            idx = np.random.choice(len(ys))
            y, x = ys[idx], xs[idx]
            us, vs = self._get_coords((y, x))
            break

        return us, vs, np.ones_like(us, dtype=bool)


class JigsawGridInferenceDataset(InferenceDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.valid_coords = []
        self.grid_spacing = self.grid_align / self.tile_width
        t_w = self.tile_width
        self.grid_offset = np.random.randint(0, (t_w - self.grid_align - 20) // 2, 2)
        # TODO Fix this
        # self.grid_offset = (0, 0)
        oy, ox = self.grid_offset
        for x in range(t_w // 2 + ox, self.H + t_w // 2 + ox, self.grid_align):
            for y in range(t_w // 2 + oy, self.W + t_w // 2 + oy, self.grid_align):
                us, vs, edge_mask = self.get_jigsaw_coords((y, x))
                if (~self.generated_mask[us, vs] & edge_mask).any():
                    self.valid_coords.append((y, x))
        self.length = len(self.valid_coords)

    def __len__(self):
        return self.length

    def get_jigsaw_coords(self, center: tuple[int, int]):
        y, x = center
        us, vs = self._get_coords((x, y))
        edge_mask = jigsaw_piece_mask(
            (y, x),
            (self.H, self.W),
            self.tile_width,
            grid_spacing=self.grid_spacing,
            grid_offset=self.grid_offset,
            seed=int(self.t),
        )
        # TODO: This should be included but breaks the water outlines for some reason
        # (~self.generated_mask[us, vs].copy()) &
        new_pixel_mask = edge_mask
        return us, vs, new_pixel_mask

    def get_coords(self):
        if len(self.valid_coords) == 0:
            if self.generated_mask.any():
                valid_tiles = np.where(self.generated_mask == 0)
                y, x = valid_tiles[0][0], valid_tiles[1][0]
            else:
                y, x = self.tile_width // 2, self.tile_width // 2
        else:
            y, x = self.valid_coords.pop()
        return self.get_jigsaw_coords((y, x))

    def _get_coords(self, center):
        h_t_w = self.tile_width // 2
        us = np.arange(center[0] - h_t_w, center[0] + h_t_w) % self.W
        vs = np.arange(center[1] - h_t_w, center[1] + h_t_w) % self.H
        vs, us = np.meshgrid(vs, us)
        return us, vs


def process_batch(
    batch, fixed_size: bool = False
) -> dict[str, torch.Tensor | int | dict]:
    cond_images: dict[str, torch.Tensor] = batch["cond_images"]
    noisy_input = batch["noisy_input"]
    metadata: dict[str, str | int] = batch["metadata"]
    new_pixels: torch.Tensor = batch["new_pixels"]
    ys = batch["ys"]
    xs = batch["xs"]
    new_mask = batch["new_pixel_mask"]

    valid_tiles = new_pixels > 0
    this_batch_size = int(valid_tiles.sum().item())
    if not fixed_size:
        cond_images = {k: v[valid_tiles] for k, v in cond_images.items()}
        noisy_input = noisy_input[valid_tiles]
        for k in metadata:
            metadata[k] = metadata[k][valid_tiles]
        ys = ys[valid_tiles]
        xs = xs[valid_tiles]
        new_mask = new_mask[valid_tiles]
        new_pixels = new_pixels[valid_tiles]

    return {
        "cond_images": cond_images,
        "noisy_input": noisy_input,
        "metadata": metadata,
        "ys": ys,
        "xs": xs,
        "new_pixel_mask": new_mask,
        "new_pixels": new_pixels,
        "this_batch_size": this_batch_size,
    }


def create_batch_list(
    batches: list,
    dataloader: DataLoader,
    signal_container: list[bool],
    fixed_size: bool = False,
    max_len: int = 5,
):
    for batch in dataloader:
        if signal_container[0]:
            return
        while max_len > 0 and len(batches) > max_len:
            if signal_container[0]:
                return
            sleep(0.01)
        processed_batch = process_batch(batch, fixed_size)
        this_batch_size = processed_batch["this_batch_size"]
        if this_batch_size == 0:
            gen_mask: np.ndarray = dataloader.dataset.generated_mask
            if gen_mask.all():
                # We are actually done generating here
                break
            else:
                logging.warning("No valid tiles in batch but not finished generating")
                # Set grid align to 1 to finish generating
                dataloader.dataset.grid_align = 1
                continue
        batches.append(processed_batch)
    signal_container[1] = True
    while len(batches) > 0:
        if signal_container[0]:
            return
        sleep(0.01)


def erode_mask(
    mask: np.ndarray, num_timesteps: int, timestep_i: int, num_erosions: int = 25
):
    if timestep_i % max(num_timesteps // num_erosions, 1):
        # Only erode every n steps
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.erode(mask.astype(np.uint8) * 255, kernel, iterations=1) > 0
    return mask


def get_sketches(
    real_data: bool, input_folder: str, filename: str, planet_cfg: PlanetConfig
):
    if not real_data:
        full_sketch = np.array(img.open(input_folder + "/" + filename))
        downsketch = full_sketch[:, :, 0]
        downland_sketch = full_sketch[:, :, 1]
        downtemp_sketch = full_sketch[:, :, 2]
    else:
        data_dir = planet_cfg.data_dir
        downland_sketch = np.array(Image.open(data_dir + "/downland_sketch.png"))
        downtemp_sketch = np.array(Image.open(data_dir + "/downtemp_sketch.png"))
        downsketch = np.array(Image.open(data_dir + "/downsketch.png"))
    return downsketch, downland_sketch, downtemp_sketch


def get_quad_sketches(
    downsketch: np.ndarray, downland_sketch: np.ndarray, downtemp_sketch: np.ndarray
):
    down_sphere = QuadSphere(downsketch, discrete=True)
    downsketch = down_sphere.quad_sphere_atlas
    downland_sphere = QuadSphere(downland_sketch, discrete=True)
    downland_sketch = downland_sphere.quad_sphere_atlas
    downtemp_sphere = QuadSphere(downtemp_sketch, discrete=True)
    downtemp_sketch = downtemp_sphere.quad_sphere_atlas
    return downsketch, downland_sketch, downtemp_sketch


sat_pipeline_cache = {}


def load_pipeline(
    sat_model_path: str, wandb_artifact_name: str | None = None
) -> TerrainDiffusionPipeline:
    if sat_model_path in sat_pipeline_cache:
        return sat_pipeline_cache[sat_model_path]
    if not os.path.exists(sat_model_path) or (os.listdir(sat_model_path) == 0):
        if wandb_artifact_name is None:
            raise ValueError(f"No model found at {sat_model_path}")
        import wandb

        try:
            wandb.login()
        except Exception:
            if "WANDB_API_KEY" in os.environ:
                wandb.login(key=os.environ["WANDB_API_KEY"])
            else:
                raise Exception(
                    "WANDB_API_KEY not found in environment variables. https://wandb.ai/authorize"
                )
        run = wandb.run or wandb.init(project="PlanetAI", job_type="download")
        artifact = run.use_artifact(wandb_artifact_name)
        artifact.download(sat_model_path)
        # Only finish the run if we created it
        if run != wandb.run:
            run.finish()

    torch.serialization.add_safe_globals([PlanetEncoder, PlanetConfig, MaskStore])
    sat_pipeline = TerrainDiffusionPipeline.from_pretrained(sat_model_path).to(device)
    # We shouldn't be loading many models so we don't ever invalidate the cache
    sat_pipeline_cache[sat_model_path] = sat_pipeline
    return sat_pipeline


def prepare_sat_pipeline(
    sat_model_path: str,
    compile_model: bool,
    num_timesteps: int,
    wandb_artifact_name: str | None = None,
) -> tuple[TerrainDiffusionPipeline, torch.Tensor]:
    sat_pipeline = load_pipeline(sat_model_path, wandb_artifact_name)
    if compile_model:
        sat_pipeline.unet = torch.compile(sat_pipeline.unet)
    sat_pipeline.scheduler.set_timesteps(num_timesteps, device=device)
    return sat_pipeline, sat_pipeline.scheduler.timesteps


def get_device_generator(seed: int | None):
    if seed is None:
        seed = randint(0, 2**32 - 1)
    generator = torch.Generator(device).manual_seed(seed)
    return generator


def get_cpu_generator(seed: int | None):
    if seed is None:
        seed = randint(0, 2**32 - 1)
    generator = torch.Generator().manual_seed(seed)
    return generator


def get_full_ocean_tensor(shape: tuple[int, int, int]):
    # Create a full ocean tensor
    # We will add noise to this to get the ocean mask for each denoising step
    # full_ocean = torch.zeros((1, 4, H_quad, W_quad), dtype=dtype)
    water_colours = [2, 5, 20, 0]
    # for i, c in enumerate(water_colours):
    #     full_ocean[0, i] = c/127.5 - 1.0
    full_ocean = np.zeros(shape, dtype=np.uint8)
    # Set full_ocean to water_colours
    for i, c in enumerate(water_colours):
        full_ocean[:, :, i] = c
    return torch.from_numpy(full_ocean).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0


def resize_sketches(sketches: list[np.ndarray], shape: tuple[int, int]):
    return [
        cv2.resize(sketch, shape, interpolation=cv2.INTER_NEAREST)
        for sketch in sketches
    ]


@dataclass
class StatItem:
    batch_time: float
    generation_time: float
    batch_size: int
    step: int
    remaining_step_pixels: int
    total_generated_pixels: int
    progress_pcnt: float
    step_pixels: int
    gpu_vram_bytes: int
    ram_bytes: int


def get_total_ram_usage() -> int:
    """Calculates RAM usage for current process and all children in bytes."""
    try:
        current_process = psutil.Process(os.getpid())
        total_ram = current_process.memory_info().rss
        for child in current_process.children(recursive=True):
            try:
                total_ram += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return total_ram
    except Exception:
        return 0


def run_inference_loop(
    model: TerrainDiffusionPipeline,
    noisy_input: torch.Tensor,
    cond_images: torch.Tensor,
    timesteps: list[torch.Tensor],
    guidance_channel: int | None = None,
    guidance_scale: float = 1.0,
    encoder_hidden_states: torch.Tensor | None = None,
    extra_step_kwargs: dict = {},
    batch_time: float = 0.0,
) -> tuple[torch.Tensor, list[StatItem]]:
    stats = []
    for i, t in enumerate(tqdm(timesteps)):
        t1 = time()
        stepped = model.step(
            noisy_input,
            cond_images,
            t,
            {"encoder_hidden_states": encoder_hidden_states},
            extra_step_kwargs,
            guidance_channel=guidance_channel,
            guidance_scale=guidance_scale,
        )
        t2 = time()
        noisy_input = stepped.prev_sample
        step_pixels = int(np.prod(list(noisy_input.shape)) // noisy_input.shape[1])
        peak_step_vram = torch.cuda.max_memory_reserved(device=None)
        torch.cuda.reset_peak_memory_stats()
        current_ram = get_total_ram_usage()
        stats.append(
            StatItem(
                batch_time=batch_time if i == 0 else 0.0,
                generation_time=t2-t1,
                batch_size=cond_images.shape[0],
                step=i,
                remaining_step_pixels=0,
                total_generated_pixels=step_pixels * (i + 1),
                progress_pcnt=(i + 1) / len(timesteps),
                step_pixels=step_pixels,
                gpu_vram_bytes=peak_step_vram,
                ram_bytes=current_ram,
            )
        )
    return noisy_input, stats


class InferenceInstance:
    @profile
    def __init__(
        self,
        inference_args: InferenceArguments,
        planet_cfg: PlanetConfig,
        output_shape: tuple[int, int] = None,
        encoder_override: EncoderOverride = EncoderOverride(),
        output_callback: Callable[[torch.Tensor], None] = lambda x: None,
        show_prediction: bool = False,
    ):
        self.inference_args = inference_args
        self.planet_cfg = planet_cfg
        self.encoder_override = encoder_override
        self.use_previous = inference_args.use_previous
        self.use_jigsaw_grid = inference_args.use_jigsaw_grid
        self.output_callback = output_callback
        self.show_prediction = show_prediction
        self.uncertainty_sketcher = UncertaintySketcher(self.planet_cfg)

        self.quadsphere = QuadSphere(shape=(planet_cfg.H, planet_cfg.W))
        if output_shape is None:
            self.H = self.quadsphere.face_width
            self.W = 6 * self.quadsphere.face_width
            self.dataset_cls = QuadInferenceDataset
        else:
            self.set_output_shape(output_shape)
        self.current_output: torch.Tensor = None

        self.num_timesteps = inference_args.timesteps
        self.compile_model = inference_args.compile_model
        self.diffusion_model_dir = inference_args.diffusion_model_dir
        self.use_dual_model = (
            self.inference_args.dual_model_upscale_weight != 1.0
            and self.inference_args.do_upscaling
        )
        self.init_model()
        self.seed = inference_args.seed
        self.batch_size = inference_args.batch_size

        self.init_extra_step_kwargs()
        self.modal_sketch = ModalSketch(self.planet_cfg)
        self.river_modal_sketch = RiverModalSketch(replace(self.planet_cfg, size=5))
        self.stopped = False
        self.signal_container = [False, False]
        self.rivers = None
        self.river_sketch = None
        self.progress_pcnt = 100.0
        self.current_output = torch.zeros(
            (1, 4, self.H, self.W),
            dtype=torch.float32 if not self.inference_args.use_fp16 else torch.float16,
            device=None,
        )
        self.previous_output = (
            self.current_output.clone() if self.use_previous else None
        )
        self.current_prediction = (
            self.current_output.clone() if self.show_prediction else None
        )
        self.use_rough_edge_mask = inference_args.use_rough_edge_mask
        self.fully_generated = False

    def set_previous_output(
        self, previous_sat: torch.Tensor, previous_dem: torch.Tensor
    ) -> None:
        # previous_sat will be HxWxC, but we want CxHxW
        if self.previous_output is None:
            self.previous_output = self.current_output.clone()
        self.previous_output[0, :3, :, :] = previous_sat.permute(2, 0, 1)
        self.previous_output[0, 3, :, :] = previous_dem
        self.current_output = self.previous_output.clone()

    def check_for_config_change(self):
        to_update = list(self.inference_args.__dict__.keys())
        to_update.remove("timesteps")  # Always remove this because it conflicts
        if (
            self.compile_model != self.inference_args.compile_model
            or self.diffusion_model_dir != self.inference_args.diffusion_model_dir
        ):
            self.compile_model = self.inference_args.compile_model
            self.diffusion_model_dir = self.inference_args.diffusion_model_dir
            to_update.remove("compile_model")
            to_update.remove("diffusion_model_dir")
            self.init_model()
        if self.num_timesteps != self.inference_args.timesteps:
            self.set_timesteps(self.inference_args.timesteps)
        if self.seed != self.inference_args.seed:
            to_update.remove("seed")
            self.set_seed(self.inference_args.seed)
        if self.use_previous != self.inference_args.use_previous:
            to_update.remove("use_previous")
            self.use_previous = self.inference_args.use_previous
            self.previous_output = (
                self.current_output.clone() if self.use_previous else None
            )
        for key in to_update:
            setattr(self, key, getattr(self.inference_args, key))

    def init_model(self):
        ver = self.inference_args.wandb_artifact_version
        self.sat_pipeline, self.timesteps = prepare_sat_pipeline(
            os.path.join(self.diffusion_model_dir, "final"),
            self.compile_model,
            self.num_timesteps,
            f"ailand/PlanetAI/PlanetAI-Model:v{ver}" if ver is not None else None,
        )
        if self.use_dual_model:
            self.other_pipeline, _ = prepare_sat_pipeline(
                os.path.join(self.inference_args.dual_model_dir, "final"),
                self.compile_model,
                self.num_timesteps,
            )

    def init_extra_step_kwargs(self):
        # Send necessary components to GPU
        gpu_generator = get_device_generator(self.seed)
        self.extra_step_kwargs = self.sat_pipeline.prepare_extra_step_kwargs(
            gpu_generator, 1.0
        )

    @timing
    def init_output(self):
        # TODO Check if this has repeating patterns and maybe initialise tile-wise
        self.current_output = torch.zeros(
            (1, 4, self.H, self.W),
            dtype=torch.float32 if not self.inference_args.use_fp16 else torch.float16,
            device=None,
        )
        return
        self.current_output = self.sat_pipeline.prepare_inputs(
            batch_size=1,
            num_noise_channels=4,
            unet_input_height=self.H,
            unet_input_width=self.W,
            dtype=torch.float32 if not self.inference_args.use_fp16 else torch.float16,
            device=None,
            generator=get_cpu_generator(self.seed),
        )

    def reset_output(self):
        # Set output to ocean
        ocean_piece = get_full_ocean_tensor((1, 1, 4))
        for c in range(4):
            self.current_output[:, c, :, :] = ocean_piece[0, c, 0]

    def set_output_shape(self, shape: tuple[int, int]):
        self.H, self.W = shape
        self.dataset_cls = (
            JigsawGridInferenceDataset if self.use_jigsaw_grid else InferenceDataset
        )

    @property
    def current_sat_output(self):
        if self.current_prediction is not None:
            return self.current_prediction[0, :3].permute(1, 2, 0)
        if self.current_output is None:
            return None
        return self.current_output[0, :3].permute(1, 2, 0)

    @property
    def current_dem_output(self):
        if self.current_prediction is not None:
            return self.current_prediction[0, 3]
        if self.current_output is None:
            return None
        return self.current_output[0, 3]

    def stop_generation(self):
        self.stopped = True
        self.signal_container[0] = True

    def get_rivers(
        self,
        quad_dem: np.ndarray,
        downland_sketch: np.ndarray,
        downtemp_sketch: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        rivers = np.zeros_like(quad_dem)
        full_landcover_sketch = cv2.resize(
            downland_sketch, (self.W, self.H), interpolation=cv2.INTER_NEAREST
        )
        full_temp_sketch = cv2.resize(
            downtemp_sketch, (self.W, self.H), interpolation=cv2.INTER_NEAREST
        )
        if quad_dem.dtype != np.uint8:
            quad_dem = (quad_dem * 127.5 + 127.5).astype(np.uint8)
        oceanmask = (
            (quad_dem == 0)
            | (full_landcover_sketch == LandcoverClasses.ANTARCTICA.gray_colour)
            | (full_landcover_sketch == LandcoverClasses.OPEN_WATER.gray_colour)
        ).astype(np.uint8) * 255
        face_width = self.quadsphere.face_width
        for face in range(6):
            slicer = slice(face * face_width, (face + 1) * face_width)
            _dem = quad_dem[:, slicer]
            rivers[:, slicer] = get_strahler_orders(
                _dem, oceanmask[:, slicer], oceanmask[:, slicer], self.planet_cfg
            )
        _land = cv2.erode(
            downland_sketch,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        _land = cv2.resize(_land, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        rivers[_land == 0] = 0
        orders = self.planet_cfg.top_n_orders
        rivers[rivers < (rivers.max() - orders)] = 0
        rivers[rivers > 0] -= rivers[rivers > 0].min()
        # rivers = process_rivers(rivers, orders + 1, [0, 0, 1, 2, 2], full_landcover_sketch)
        rivers = cv2.dilate(rivers, np.ones((5, 5)), iterations=1)
        river_sketch = self.river_modal_sketch.get_sketch(
            full_landcover_sketch, full_temp_sketch, rivers
        )
        return rivers, river_sketch

    def set_timesteps(self, timesteps: int):
        self.sat_pipeline.scheduler.set_timesteps(timesteps, device=device)
        self.num_timesteps = timesteps
        self.timesteps = self.sat_pipeline.scheduler.timesteps

    def set_seed(self, seed: int):
        self.seed = seed
        self.init_extra_step_kwargs()

    def two_phase_generate(
        self,
        downsketch: np.ndarray,
        downland_sketch: np.ndarray,
        downtemp_sketch: np.ndarray,
        timestep_ratio: float = 0.25,
    ):
        """Generate and save the images"""
        first_timesteps = int(self.num_timesteps * timestep_ratio)
        if first_timesteps == 0:
            first_timesteps = 1
        second_time_steps = self.num_timesteps

        self.set_timesteps(first_timesteps)
        stats = self._generate(downsketch, downland_sketch, downtemp_sketch)
        self.save_images(self.inference_args.output_folder, [], stats_dict=stats)
        dem = self.current_dem_output
        self.set_timesteps(second_time_steps)
        self.generate(downsketch, downland_sketch, downtemp_sketch, dem)

    def generate(
        self,
        downsketch: np.ndarray,
        downland_sketch: np.ndarray,
        downtemp_sketch: np.ndarray,
        dem: (
            np.ndarray | None
        ) = None,  # Output from previous generation to derive rivers from
        rivers: np.ndarray | None = None,
        previous_mask: np.ndarray | None = None,
        downsat: np.ndarray | None = None,
        downdem: np.ndarray | None = None,
    ):
        """
        Generate and save the images

        previous_mask (np.ndarray) - A mask of the sketch pixels that have been modified since the last generation.
        """
        self.check_for_config_change()
        stats = self._generate(
            downsketch, downland_sketch, downtemp_sketch, dem, rivers, previous_mask, downsat, downdem
        )
        gc.collect()
        # Save one more time to make sure the final outputs are saved
        combined_sketch = (
            self.downland_sketch.astype(np.uint32) * 256**2
            + self.downtemp_sketch.astype(np.uint32) * 256
            + self.downsketch.astype(np.uint32)
        )
        images = [
            ("sat_output", self.current_output[0, :3]),
            ("dem_output", self.current_output[0, 3]),
            ("land_sketch", gray_to_land(self.downland_sketch)),
            ("temp_sketch", self.downtemp_sketch),
            ("sketch", self.downsketch),
            ("modal_sketch", self.down_modal_sketch),
            (
                "uncertainty_sketch",
                self.uncertainty_sketcher.get_uncertainty_sketch(combined_sketch)
            )
            # ('water_mask', (water_mask*255).astype(np.uint8))
        ]
        if self.previous_output is not None:
            images.extend(
                [
                    ("previous_sat_output", self.previous_output[0, :3]),
                    ("previous_dem_output", self.previous_output[0, 3]),
                ]
            )
        if self.rivers is not None:
            images.append(
                ("rivers", np_rgb(self.rivers, cmap="viridis").astype(np.uint8))
            )
        path = self.inference_args.output_folder
        if not self.stopped and self.inference_args.save_outputs:
            logging.info(f"Saving images to {path}")
            self.save_images(path, images, stats_dict=stats)
            logging.info("Quad sphere images saved")
            if self.W == 6 * self.H:
                logging.info("Converting to normal")
                normal_image = self.quadsphere.get_normal_atlas(
                    tensor_to_np(self.current_output[0, :4])
                )
                extra_images = [
                    ("sat_normal", normal_image[:, :, :3]),
                    ("dem_normal", normal_image[:, :, 3]),
                ]
                self.save_images(path, extra_images)
                logging.info("Normal images saved")
        elif self.stopped:
            logging.info("Generation stopped. Not saving images")
        elif not self.inference_args.save_outputs:
            logging.info("Save Outputs is False. Not saving images")

        self.stopped = True
        self.signal_container[0] = True
        return path, images

    @profile
    def _generate(
        self,
        downsketch: np.ndarray,
        downland_sketch: np.ndarray,
        downtemp_sketch: np.ndarray,
        dem: (
            np.ndarray | None
        ) = None,  # Output from previous generation to derive rivers from
        rivers: np.ndarray | None = None,
        previous_mask: np.ndarray | None = None,
        downsat: np.ndarray | None = None,
        downdem: np.ndarray | None = None,
    ) -> dict[str, float | int | str]:

        self.progress_pcnt = 0
        self.stopped = False
        self.signal_container[0] = False

        self.rivers = None
        self.river_sketch = None
        if self.stopped:
            return
        if dem is not None and rivers is None:
            assert dem.shape == (self.H, self.W)
            self.rivers, self.river_sketch = self.get_rivers(
                dem, downland_sketch, downtemp_sketch
            )
        elif rivers is not None:
            self.rivers = rivers
            # self.river_sketch = self.river_modal_sketch.get_sketch(downland_sketch, downtemp_sketch, rivers)

        # Create modal sketch of the entire sketch
        if self.stopped:
            return
        down_modal_sketch = self.modal_sketch.get_sketch(
            downland_sketch, downtemp_sketch, mars_mask=(downland_sketch == 255)
        )
        self.down_modal_sketch = down_modal_sketch
        self.downsketch = downsketch
        self.downland_sketch = downland_sketch
        self.downtemp_sketch = downtemp_sketch
        if self.stopped:
            return
        full_modal_sketch = cv2.resize(
            down_modal_sketch, (self.W, self.H), interpolation=cv2.INTER_NEAREST
        )
        if self.rivers is not None:
            # full_modal_sketch[self.rivers > 0] = self.river_sketch[self.rivers > 0]
            self.river_sketch = full_modal_sketch

        if self.stopped:
            return
        logging.info("Initialising output")
        self.init_output()
        self.fully_generated = False

        # TODO Remove need for full size sketches
        if self.stopped:
            return
        sketch, land_sketch, temp_sketch = resize_sketches(
            [downsketch, downland_sketch, downtemp_sketch], (self.W, self.H)
        )
        if previous_mask is not None:
            prev_h, prev_w = previous_mask.shape
            if 2 * prev_h == prev_w and self.W == 6 * self.H:
                previous_mask = QuadSphere(
                    atlas=previous_mask, discrete=True
                ).quad_sphere_atlas
            previous_mask = resize_sketches(
                [previous_mask.astype(np.uint8)], (self.W, self.H)
            )[0]
            previous_mask = previous_mask > 0

        logging.info("Creating water mask")
        self.water_mask = ((downland_sketch == 0) & (downsketch == 0)).astype(
            np.uint8
        ) * 255
        # TODO Maybe do area resize so that the water mask is less blocky
        self.water_mask = cv2.resize(
            self.water_mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST
        )
        self.water_mask = self.water_mask > 0

        logging.info("Creating generation mask")
        self.generated_mask = np.zeros((self.H, self.W), dtype=bool)

        generation_times = []
        lookback_steps = 5
        total_tiles_generated = 0
        start_time = time()
        statistics = []
        total_pixels = 0
        total_generated_pixels = 0
        if self.stopped:
            return
        logging.info("Calculating total pixels")
        _water_mask = self.water_mask.copy()
        _previous_mask = previous_mask.copy() if previous_mask is not None else None
        for i, t in enumerate(self.timesteps):
            step_pixels = self.H * self.W
            skip_mask = np.zeros((self.H, self.W), dtype=bool)
            if self.inference_args.use_water_mask:
                if self.inference_args.erode_masks:
                    _water_mask = erode_mask(_water_mask, len(self.timesteps), i)
                    skip_mask |= _water_mask
                else:
                    skip_mask |= _water_mask
            if self.use_previous and _previous_mask is not None:
                if self.inference_args.erode_masks:
                    _previous_mask = erode_mask(_previous_mask, len(self.timesteps), i)
                    skip_mask |= _previous_mask
                else:
                    skip_mask |= _previous_mask
            step_pixels -= skip_mask.sum()
            total_pixels += step_pixels
        if self.stopped:
            return
        logging.info("Starting generation with args")
        logging.info(str(self.inference_args))

        with tqdm(total=total_pixels) as pbar:
            for i, t in enumerate(self.timesteps):
                # gc.collect()
                if self.stopped:
                    break
                self.generated_mask[:, :] = False
                self.actual_generated_mask = self.generated_mask.copy()
                if self.inference_args.use_water_mask:
                    # Always erode the last step regardless of the flag
                    if self.inference_args.erode_masks or t == 0:
                        self.water_mask = erode_mask(
                            self.water_mask, len(self.timesteps), i
                        )
                    self.generated_mask[self.water_mask] = True
                    ocean_piece = get_full_ocean_tensor((1, 1, 4))
                    for c in range(4):
                        self.current_output[:, c, self.water_mask] = ocean_piece[
                            0, c, 0
                        ]
                        if self.current_prediction is not None:
                            self.current_prediction[:, c, self.water_mask] = ocean_piece[
                                0, c, 0
                            ]
                if previous_mask is not None and self.use_previous:
                    if self.inference_args.erode_masks or t == 0:
                        _previous_mask = erode_mask(
                            previous_mask, len(self.timesteps), i
                        )
                    ys, xs = np.where(_previous_mask)
                    self.generated_mask[ys, xs] = True
                    self.actual_generated_mask[ys, xs] = True
                    self.current_output[:, :, ys, xs] = self.previous_output[
                        :, :, ys, xs
                    ]
                    previous_mask = _previous_mask

                # TODO Fix not all tiles generated error
                remaining_step_pixels = (~self.generated_mask).sum()

                dataset = self.dataset_cls(
                    generated_mask=self.generated_mask,
                    current_output=self.current_output.clone(),
                    stacked_sketch=np.dstack([sketch, land_sketch, temp_sketch]),
                    planet_cfg=self.planet_cfg,
                    inference_args=self.inference_args,
                    device=device,
                    river_mask=self.rivers,
                    water_mask=self.water_mask,
                    scheduler=self.sat_pipeline.scheduler,
                    t=t,
                    first_timestep=(i == 0),
                    seed=self.seed,
                    grid_align=self.inference_args.grid_align,
                    tile_width=self.inference_args.tile_size,
                    encoder_override=self.encoder_override,
                    previous_mask=previous_mask if self.use_previous else None,
                    previous_output=(
                        self.previous_output
                        if self.use_previous
                        else None
                    ),
                    downsat=downsat,
                    downdem=downdem,
                )
                dataloader = DataLoader(
                    dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
                )
                compiled = self.inference_args.compile_model
                batch_list = []
                self.signal_container = [self.stopped, False]
                num_batch_threads = 1
                batch_threads = [
                    Thread(
                        target=create_batch_list,
                        args=(batch_list, dataloader, self.signal_container, compiled),
                    )
                    for _ in range(num_batch_threads)
                ]
                for batch_thread in batch_threads:
                    batch_thread.start()
                try:
                    total_generated_pixels, total_tiles_generated = self.inference_step(
                        previous_mask,
                        remaining_step_pixels,
                        total_pixels,
                        total_generated_pixels,
                        start_time,
                        generation_times,
                        total_tiles_generated,
                        statistics,
                        pbar,
                        batch_list,
                        lookback_steps,
                        i,
                        self.current_output,
                        t,
                    )

                    self.output_callback(
                        self.current_prediction
                        if self.current_prediction is not None
                        else self.current_output
                    )
                finally:
                    for batch_thread in batch_threads:
                        batch_thread.join(5.0)
                    del dataset
                    if not np.all(self.generated_mask):
                        logging.warning(
                            f"Error: Not all tiles generated at timestep {i}"
                        )
        if self.current_prediction is not None:
            self.current_prediction = self.current_output.clone()
        self.output_callback(self.current_output)
        logging.info("Generation complete")
        logging.info("Filling ocean")
        if self.inference_args.use_water_mask:
            ocean_piece = get_full_ocean_tensor((1, 1, 4))
            for c in range(4):
                # TODO Using actual_generated_mask here is nice because it means that we don't have a blocky water mask
                # but it also means we have to do a hack to fill the ocean each step.
                # Consider fixing it by instead making water mask more natural.
                self.current_output[:, c, ~self.actual_generated_mask] = ocean_piece[
                    0, c, 0
                ]
                if self.current_prediction is not None:
                    self.current_prediction[:, c, ~self.actual_generated_mask] = ocean_piece[
                        0, c, 0
                    ]
            self.generated_mask[self.water_mask] = True
            self.actual_generated_mask[self.water_mask] = True

        if previous_mask is not None and self.use_previous:
            ys, xs = np.where(previous_mask)
            self.generated_mask[ys, xs] = True
            self.actual_generated_mask[ys, xs] = True
            self.current_output[:, :, ys, xs] = self.previous_output[:, :, ys, xs]
        if not self.stopped:
            assert self.generated_mask.all()
            assert self.actual_generated_mask.all()
            self.fully_generated = True
        if self.use_previous:
            self.previous_output = self.current_output.clone()

        total_time = time() - start_time
        h, w = self.generated_mask.shape
        stats_dict = self.print_stat_summary(statistics, total_time, len(self.timesteps), (h, w))
        with open(
            os.path.join(self.inference_args.statistics_dir, "statistics.json"), "w"
        ) as f:
            statistics = [stat.__dict__ for stat in statistics]
            json.dump(statistics, f)
        return stats_dict

    def print_stat_summary(
        self, statistics: list[StatItem], total_time: float, total_timesteps: int, output_shape: tuple[int, int]
    ) -> dict[str, float | int | str]:
        if len(statistics) == 0:
            return {}
        total_batch_time = sum([stat.batch_time for stat in statistics])
        total_generation_time = sum([stat.generation_time for stat in statistics])
        total_tiles = sum([stat.batch_size for stat in statistics])
        total_used_pixels = sum([stat.step_pixels for stat in statistics])
        total_generated_pixels = self.inference_args.tile_size**2 * total_tiles
        peak_vram = max([stat.gpu_vram_bytes for stat in statistics])
        peak_ram = max([stat.ram_bytes for stat in statistics])
        h, w = output_shape
        print("-" * 20)
        print("Statistics:")
        print(f"Shape: {w}x{h}")
        print(f"Timesteps: {total_timesteps}")
        print(f"Dataloader %: {total_batch_time / total_time * 100:.2f}%")
        print(f"GPU %: {total_generation_time / total_time * 100:.2f}%")
        print(f"Peak VRAM: {peak_vram / 1024 ** 3:.2f} GB")
        print(f"Peak RAM: {peak_ram / 1024 ** 3:.2f} GB")
        print(f"Total time: {total_time:.2f}s")
        print(f"Total tiles: {total_tiles}")
        print(f"Tiles/s: {total_tiles / total_time:.2f}")
        print(
            f"Pixel efficiency: {total_used_pixels / total_generated_pixels:.2%}"
        )
        print("-" * 20)
        return {
            "timesteps": total_timesteps,
            "dataloader_pcnt": total_batch_time / total_time * 100,
            "gpu_pcnt": total_generation_time / total_time * 100,
            "total_time": total_time,
            "total_tiles": total_tiles,
            "tiles_rate": total_tiles / total_time,
            "pixel_efficiency": total_used_pixels / total_generated_pixels,
            "gpu_peak_vram": peak_vram,
            "peak_ram": peak_ram,
            "output_height": h,
            "output_width": w,
            "device": torch.cuda.get_device_name(),
        }

    def get_batch(self, batch_list: list[list[torch.Tensor]]) -> list[torch.Tensor]:
        while True:
            if (self.signal_container[1] and len(batch_list) == 0) or self.stopped:
                return None
            try:
                return batch_list.pop()
            except Exception:
                sleep(0.01)
                continue

    @profile
    def inference_step(
        self,
        previous_mask: np.ndarray,
        remaining_step_pixels: int,
        total_pixels: int,
        total_generated_pixels: int,
        start_time: float,
        generation_times: list[float],
        total_tiles_generated: int,
        statistics: list[StatItem],
        pbar: tqdm,
        batch_list: list,
        lookback_steps: int,
        i: int,
        next_output: torch.Tensor,
        t: int,
    ) -> tuple[int, int]:
        while len(batch_list) > 0 or not self.signal_container[1]:
            bt1 = time()
            batch = self.get_batch(batch_list)
            if batch is None:
                break
            batch_time = time() - bt1
            if batch_time > 1.0:
                logging.warning(f"Batch time: {batch_time:.2f}")
            cond_images: dict[str, torch.Tensor] = batch["cond_images"]
            noisy_input: torch.Tensor = batch["noisy_input"]
            metadata: dict[str, str | int] = batch["metadata"]
            ys: torch.Tensor = batch["ys"]
            xs: torch.Tensor = batch["xs"]
            new_mask: torch.Tensor = batch["new_pixel_mask"]
            this_batch_size: int = batch["this_batch_size"]

            cond_image = get_model_condition(
                cond_images,
                self.planet_cfg.input_types,
                inject_rivers=self.planet_cfg.river_upa_mode == "injected"
            )

            size_5_cond_image = get_model_condition(
                cond_images, replace(self.planet_cfg, image_mode="planet").input_types
            )

            terrain_style = self.sat_pipeline.create_terrain_style(
                cond_image, cond_image, metadata
            )
            encoder_hidden_states = self.sat_pipeline._encode_terrain_style(
                terrain_style
            ).to(device)
            t1 = time()
            noise_pred = self.sat_pipeline._step(
                noisy_input.to(device),
                cond_image.to(device),
                t,
                {"encoder_hidden_states": encoder_hidden_states},
                self.extra_step_kwargs,
                guidance_channel=self.inference_args.guidance_channel,  # River channel
                guidance_scale=self.inference_args.guidance_scale,
            )

            upscale_weight = self.inference_args.dual_model_upscale_weight
            if self.use_dual_model:
                size_5_noise_pred = self.other_pipeline._step(
                    noisy_input.to(device),
                    size_5_cond_image.to(device),
                    t,
                    {"encoder_hidden_states": encoder_hidden_states},
                    self.extra_step_kwargs,
                    guidance_channel=3,  # River channel
                    guidance_scale=self.inference_args.guidance_scale,
                )
                noise_pred = noise_pred * upscale_weight + size_5_noise_pred * (
                    1 - upscale_weight
                )

            stepped = self.sat_pipeline.scheduler.step(
                noise_pred, t, noisy_input.to(device), **self.extra_step_kwargs
            )
            diffusion_outputs: torch.Tensor = stepped.prev_sample

            output_display = self.planet_cfg.output_display(
                cond_image.cpu(), diffusion_outputs.cpu(), noisy_input.cpu()
            )

            save_steps = False
            # This is useful for debugging
            if save_steps:
                os.makedirs("inference_steps", exist_ok=True)

                timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                if t == 0:
                    Image.fromarray(tensor_to_np(output_display)).save(
                        f"./inference_steps/{timestamp}-{total_generated_pixels}.png"
                    )

                if total_generated_pixels == 0 and self.inference_args.do_upscaling:
                    full_outputs, _ = run_inference_loop(
                        self.sat_pipeline,
                        noisy_input.to(device),
                        cond_image.to(device),
                        self.timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        extra_step_kwargs=self.extra_step_kwargs,
                    )
                    output_display = self.planet_cfg.output_display(
                        cond_image.cpu(), full_outputs.cpu(), noisy_input.cpu()
                    )

                    os.makedirs("inference_steps", exist_ok=True)

                    Image.fromarray(tensor_to_np(output_display)).save(
                        f"./inference_steps/{timestamp}-full.png"
                    )
                    pass

            diffusion_prediction: torch.Tensor = stepped.pred_original_sample
            if t == 0:
                diffusion_outputs = self.sat_pipeline.postprocess(
                    stepped.prev_sample.clone(),
                    cond_images=cond_image,
                    normalise_output=self.inference_args.normalise_outputs,
                )
            peak_step_vram = torch.cuda.max_memory_reserved(device=None)
            torch.cuda.reset_peak_memory_stats()
            current_ram = get_total_ram_usage()
            t2 = time()

            # Set outputs
            to_add = diffusion_outputs.clone().cpu().to(next_output.dtype)
            to_add_pred = diffusion_prediction.clone().cpu().to(next_output.dtype)
            batch_new_pixels = 0
            for b in range(len(ys)):
                batch_ys = ys[b]
                batch_xs = xs[b]
                to_add_pixels = new_mask[b].sum()
                if to_add_pixels == 0:
                    continue
                # This doesn't really help
                # if t == 0:
                #     # Blend DEM in last step only
                #     existing_mask = ~new_mask[b]
                #     existing = next_output[0][3, batch_ys, batch_xs]
                #     to_add[b, 3][existing_mask] = to_add[b, 3][existing_mask] / 2 + existing[existing_mask] / 2
                # TODO mask with to_add_pixels
                new_mask_b = (
                    new_mask[b].numpy()
                    # TODO Try and fix this
                    # & ~self.water_mask[batch_ys, batch_xs]
                    # & ~previous_mask[batch_ys, batch_xs]
                )
                to_add[b, 0:4, ~new_mask_b] = 0
                next_output[0, 0:4, batch_ys, batch_xs] = (
                    next_output[0, 0:4, batch_ys, batch_xs] * ~new_mask_b
                    + to_add[b, 0:4]
                )
                if self.current_prediction is not None:
                    to_add_pred[b, 0:4, ~new_mask_b] = 0
                    self.current_prediction[0, 0:4, batch_ys, batch_xs] = (
                        self.current_prediction[0, 0:4, batch_ys, batch_xs] * ~new_mask_b
                        + to_add_pred[b, 0:4]
                    ).clip(-1.0, 1.0)
                to_add_pixels = (
                    ~self.actual_generated_mask[batch_ys, batch_xs] & new_mask_b
                ).sum()
                batch_new_pixels += to_add_pixels
                self.actual_generated_mask[batch_ys, batch_xs] |= new_mask_b
                # self.generated_mask[batch_ys, batch_xs] |= new_mask_b  # This is already done above
                total_generated_pixels += to_add_pixels
                remaining_step_pixels -= to_add_pixels
                pbar.update(int(to_add_pixels))
                # self.progress_pcnt = total_generated_pixels / total_pixels * 100
            if remaining_step_pixels < 0:
                remaining_step_pixels = (~self.generated_mask).sum()
            pixel_efficiency = batch_new_pixels / (
                ys.shape[0] * ys.shape[1] * ys.shape[2]
            )

            # Collect statistics
            generation_time = t2 - t1
            generation_times.append(generation_time)
            lb = min(lookback_steps, len(generation_times))
            avg_time = sum(generation_times[-lb:]) / lb
            tile_speed = this_batch_size / avg_time
            total_tiles_generated += this_batch_size
            total_step_time = time() - bt1
            gpu_overhead_pcnt = generation_time / total_step_time * 100

            pbar.set_description(
                f"Step {i+1}/{len(self.timesteps)} | {tile_speed:.2f} tps "
                + f"| {pixel_efficiency:.2%} eff | {total_tiles_generated} tiles | {gpu_overhead_pcnt:.1f}% GPU"
            )
            # Update statistics
            statistics.append(
                StatItem(
                    batch_time=float(batch_time),
                    generation_time=float(t2 - t1),
                    batch_size=int(this_batch_size),
                    step=int(i),
                    remaining_step_pixels=int(remaining_step_pixels),
                    total_generated_pixels=int(total_generated_pixels),
                    progress_pcnt=float(total_generated_pixels / total_pixels * 100),
                    step_pixels=int(batch_new_pixels),
                    gpu_vram_bytes=peak_step_vram,
                    ram_bytes=current_ram,
                )
            )
            bt1 = time()
        # TODO change back to pixel progress
        self.progress_pcnt = (1000 - int(t)) / 1000 * 100
        if not self.stopped:
            assert self.generated_mask.all()
        return total_generated_pixels, total_tiles_generated

    def save_images(
        self,
        path: str,
        images: list[tuple[str, np.ndarray]],
        batch_num: int | None = None,
        stats_dict: dict[str, float | int | str] = {},
        name_suffix: str | None = None,
    ):
        date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if batch_num is not None:
            date = f"{date}_{batch_num}"
        if name_suffix is not None:
            date = f"{date}_{name_suffix}"
        for name, image in images:
            if isinstance(image, torch.Tensor):
                if "dem" in name:
                    image = ((image.cpu().detach().numpy() + 1) * 65535 / 2).round().clip(0, 65535).astype(np.uint16)
                else:
                    image = tensor_to_np(image)
            else:
                image = image.round().clip(0, 255).astype(np.uint8)
            os.makedirs(path, exist_ok=True)
            os.makedirs(os.path.join(path, "final"), exist_ok=True)
            img.fromarray(image).save(os.path.join(path, f"{name}.png"))
            img.fromarray(image).save(os.path.join(path, "final", f"{date}_{name}.png"))

        with open(os.path.join(path, "args.json"), "w") as f:
            data = {
                "inference_args": self.inference_args.__dict__,
                "planet_cfg": asdict(self.planet_cfg),
                "stats": stats_dict,
            }
            json.dump(data, f, indent=2)

        with open(os.path.join(path, "final", f"{date}_args.json"), "w") as f:
            data = {
                "inference_args": self.inference_args.__dict__,
                "planet_cfg": asdict(self.planet_cfg),
                "stats": stats_dict,
            }
            json.dump(data, f, indent=2)


def live_save_outputs(
    current_output: torch.tensor,
    generated_mask: np.ndarray,
    path: str,
    delay: int = 100,
):
    while live_save_enabled:
        t1 = time()
        save_outputs(current_output, generated_mask, path, shape=(6 * 512, 512))
        t2 = time()
        save_time = t2 - t1
        sleep_time = max(0, delay / 1000 - save_time)
        sleep(sleep_time)


def save_outputs(
    current_output: torch.tensor, generated_mask: np.ndarray, path: str, shape=None
):
    np_current_output = (current_output[0] * 127.5 + 127.5).round().clamp(0, 255)
    np_current_output = (
        np_current_output.detach().cpu().numpy().transpose((1, 2, 0)).astype(np.uint8)
    )
    sat_output = np_current_output[:, :, :3]
    dem_output = np_current_output[:, :, 3]
    generated_mask = generated_mask.astype(np.uint8) * 255
    if shape is not None:
        sat_output = cv2.resize(sat_output, shape, interpolation=cv2.INTER_NEAREST)
        dem_output = cv2.resize(dem_output, shape, interpolation=cv2.INTER_NEAREST)
        generated_mask = cv2.resize(
            generated_mask, shape, interpolation=cv2.INTER_NEAREST
        )
    # Save temp outputs and then rename them to prevent flashing
    img.fromarray(sat_output).save(path + "/temp_sat_output.png")
    img.fromarray(dem_output).save(path + "/temp_dem_output.png")
    img.fromarray(generated_mask).save(path + "/temp_generated_mask.png")
    os.rename(path + "/temp_sat_output.png", path + "/sat_output.png")
    os.rename(path + "/temp_dem_output.png", path + "/dem_output.png")
    os.rename(path + "/temp_generated_mask.png", path + "/generated_mask.png")


def main():
    parser = CustomArgumentParser(
        (
            InferenceArguments,
            PlanetConfig,
        ),
        description="Run inference on a diffusion model",
    )

    args: tuple[InferenceArguments, PlanetConfig] = parser.parse_args_into_dataclasses()
    inference_args, planet_cfg = args

    downsketch, downland_sketch, downtemp_sketch, river_mask = get_sketches(
        inference_args.real_data,
        inference_args.input_folder,
        inference_args.filename,
        planet_cfg,
    )
    dem = None

    # For debugging only generate small area
    # ------------------------------------
    h, w = downsketch.shape
    slicer = slice(3 * h // 8, 5 * h // 8), slice(3 * w // 8, 5 * w // 8)
    mask = np.zeros_like(downsketch, bool)
    mask[slicer] = True
    downsketch = downsketch[slicer]
    downland_sketch = downland_sketch[slicer]
    downtemp_sketch = downtemp_sketch[slicer]
    # dem = np.array(Image.open(inference_args.input_folder + '/dem_output.png'))
    # dem = np.load(planet_cfg.data_dir + f'/quad_sat_dem{planet_cfg.W}x{planet_cfg.H}.npy')[:, :, 3]
    # ------------------------------------
    h, w = downsketch.shape
    delta = planet_cfg.delta
    inference_instance = InferenceInstance(
        inference_args, planet_cfg, output_shape=(h * delta, w * delta)
    )
    path, images = inference_instance.generate(
        downsketch, downland_sketch, downtemp_sketch, dem=dem
    )


if __name__ == "__main__":
    main()
