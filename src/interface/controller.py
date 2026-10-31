import os
from threading import Thread
from typing import TYPE_CHECKING, Callable
from dataclasses import replace

import numpy as np
import torch
import cv2
from PIL import Image as img
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QFileDialog

from planetAI.src.data.modal_sketch import ModalSketch
from planetAI.src.data.river_modal_sketch import RiverModalSketch
from planetAI.src.data.landcover_utils import gray_to_land, LandcoverClasses, translate_land
from planetAI.src.data.uncertainty_sketch import UncertaintySketcher
from planetAI.src.data.utils import get_brush_kernel, np_rgb, tensor_to_np, PlanetConfig, profile
from planetAI.src.data.sketch_gen import dilate_paint
from planetAI.src.data.sphere_mapping import QuadSphere, quad_to_normal
from planetAI.src.data.dataset import EncoderOverride
from planetAI.src.data.noise_settings import NoiseSettings
from planetAI.src.data.mean_precip import PrecipSketch
from src.interface.panels.Icons import add_icons_to_temperature_sketch
from src.interface.widgets.WaitCursor import wait_cursor

if TYPE_CHECKING:
    from .view import View

from .interface_types import (
    DEMSketchEventMetadata,
    EventMetadata,
    DEMEventMetadata,
    LandCoverEventMetadata,
    RiverDerivationSettings,
    TempPresetEventMetadata,
    RiverEventMetadata,
    RiverDisplayType,
    EventTypeEnum,
    RegenEventMetadata,
    AtlasType,
    ToolEnum,
    EditTypeEnum,
    modal_stack_display_func,
)
from .utils import (
    derive_sketch_rivers,
    get_line,
    get_lasso_points,
    get_paint_points,
    create_temp_preset,
    get_temp_folder,
    noise_atlas,
    temperature_paint,
    derive_landcover,
)
from ..models.diffusion.inpaint_inference import (
    InferenceInstance,
    InferenceArguments,
)
from ..models.diffusion.inference_controller import InferenceController, SketchArgs
from .atlas_storage import ATLAS_STORAGE
from planetAI.src.data.atlas_loader import AtlasLoader
from .tasks import RepeatingTask, OnceOffTask
import traceback


class StartupTask:
    def __init__(self, target: Callable, args: list = [], kwargs: dict = {}):
        self.target = target
        self.args = args
        self.kwargs = kwargs


class Controller:
    def __init__(
        self,
        view: "View",
        shape: tuple[int, int],
        output_shape: tuple[int, int],
        modal_sketch: ModalSketch,
        river_modal_sketch: RiverModalSketch,
        inference_arguments: InferenceArguments,
        planet_cfg: PlanetConfig,
        encoder_override: EncoderOverride,
    ):
        self.encoder_override = encoder_override
        self.view = view
        self.dem = np.zeros(shape, dtype=np.float32)
        self.selection_mask = np.zeros(shape, dtype=bool)
        self.cursor_shape = shape
        self.cursor_mask = np.zeros(self.cursor_shape, dtype=bool)
        self.filtered_selection_mask = np.zeros(shape, dtype=np.float32)
        self.landcover = np.zeros(shape, dtype=np.uint8)
        self.temperature = np.zeros(shape, dtype=np.uint8)
        self.temperature_preset = np.zeros(shape, dtype=np.uint8)
        self._modal_sketch_stack = None
        self._uncertainty_sketch = None
        self.uncertainty_sketcher = UncertaintySketcher(planet_cfg)
        self.modal_sketch = modal_sketch
        self.river_modal_sketch = river_modal_sketch
        self.cursor_ys = None
        self.cursor_xs = None
        self.quad_sphere = QuadSphere(shape=output_shape, discrete=True)
        qh, qw = self.quad_sphere.quad_shape
        self._sat_output = torch.zeros((qh, qw, 3), dtype=torch.uint8)
        self._dem_output = torch.zeros((qh, qw), dtype=torch.uint8)
        self.current_display_image: np.ndarray = np.zeros(shape, dtype=np.uint8)
        self.current_display_type = EventTypeEnum.DEM
        self.last_image_type = None
        self.last_x = None
        self.last_y = None
        self.lasso_points = []
        self.planet_cfg = planet_cfg
        self.planet_cfg.gaussian_blur = 0
        self.inference_arguments = inference_arguments
        self.inference_instance = None
        self.inference_controller = None
        self.inference_thread = None
        self.is_exiting = False
        self.shape = shape
        self.output_shape = output_shape
        self.noise_overlay_enabled = False
        self.noise_preview_settings = None
        self.noise_preview = np.zeros(shape, dtype=np.uint8)
        self.preview_coords = (0, 0)
        self.view_coords = (0, 0)
        self.align_noise = False
        self.noise_format_func = lambda noise: noise
        self.land_mult = np.ones(shape, dtype=np.float32)
        self.land_mask = np.zeros(shape, dtype=bool)
        self.normal_river_upa = np.zeros(output_shape, dtype=np.float32)
        self.quad_river_upa = np.zeros((qh, qw), dtype=np.float32)
        self.river_stack = np.zeros((*output_shape, 7), dtype=np.float32)
        self.last_river_meta: RiverEventMetadata | None = None
        self.river_upa_max = 1.0
        self.river_norm_max = 255
        self.river_norm_min = 0
        self.atlas_loader = AtlasLoader(self.planet_cfg)
        self.precip_sketcher = PrecipSketch(self.planet_cfg, exclude_dem=True)
        self.regen_mask = np.zeros(
            shape, dtype=bool
        )  # For now use a normal atlas. TODO Use a quad atlas
        self.temp_steepness = 2
        self.temp_factor = 4
        self.dem_max = 175  # This seems to be the max value that we can generate for the dem so use this for scaling

        self._landcover_modified = False
        self._temperature_modified = False
        self._dem_modified = False
        self._river_modified = False
        self._modal_modified = True  # Make sure initial update fires
        self._uncertainty_modified = True

        self.update_images_enabled = False

        self.update_images_thread = RepeatingTask(
            target=self.update_inference_images, stop_condition=lambda: self.is_exiting
        )
        self.update_images_thread.start()

        self.update_display_timer = QTimer()
        self.update_display_timer.timeout.connect(self.display_current_outputs)
        self.update_display_timer.start(100)
        self.update_progress_timer = QTimer()
        self.update_progress_timer.timeout.connect(self.update_progress)
        self.update_progress_timer.start(100)

        # TODO: These might still cause issues. We should only enable generation and
        # the update threads once the startup tasks are done
        self.startup_tasks = [
            StartupTask(self.load_on_open),
            StartupTask(self.init_inference_instance),
            StartupTask(self.update_atlas_storage),
            StartupTask(ATLAS_STORAGE.load_earth_data, args=[self.planet_cfg.data_dir]),
            StartupTask(
                self.view.update_globes, args=[[t for t in AtlasType]]
            ),  # Just tell the view that we have updated every type
        ]

    def startup(self):
        Thread(target=self.do_startup_tasks).start()

    def do_startup_tasks(self):
        for task in self.startup_tasks:
            try:
                task.target(*task.args, **task.kwargs)
            except Exception as e:
                print(f"Error in startup task {task.target.__name__}: {e}")
                traceback.print_exc()

    def set_noise_settings(self, noise_settings: NoiseSettings):
        self.noise_preview_settings = noise_settings
        self.update_noise_preview()
        self.display_current_input()

    def update_noise_preview(self):
        self.noise_preview = noise_atlas(
            self.shape,
            self.noise_preview_settings,
            self.preview_coords,
            self.view_coords,
        )

    def set_view_coords(self, coords: tuple[float, float]):
        if not self.align_noise:
            return
        updated = self.view_coords != coords
        self.view_coords = coords
        if updated:
            self.update_noise_preview()

    def set_preview_coords(self, coords: tuple[float, float]):
        if not self.align_noise:
            return
        updated = self.preview_coords != coords
        self.preview_coords = coords
        if updated:
            self.update_noise_preview()

    def set_both_coords(
        self, view_coords: tuple[float, float], preview_coords: tuple[float, float]
    ):
        if not self.align_noise:
            return
        updated = (
            self.view_coords != view_coords or self.preview_coords != preview_coords
        )
        self.view_coords = view_coords
        self.preview_coords = preview_coords
        if updated:
            self.update_noise_preview()

    def set_align_noise(self, align_noise: bool):
        if not align_noise:
            self.view_coords = (0, 0)
            self.preview_coords = (0, 0)
        self.align_noise = align_noise
        self.update_noise_preview()

    def set_noise_format_func(self, func: Callable[[np.ndarray], np.ndarray]) -> None:
        """Set the function to format the sketch image"""
        self.noise_format_func = func

    def toggle_noise_overlay(self):
        self.noise_overlay_enabled = not self.noise_overlay_enabled
        self.display_current_input()

    def save_images(self):
        folder_path = QFileDialog.getExistingDirectory(
            None, "Select Folder to Save Images", ""
        )
        if folder_path:
            self._save_images(os.path.join(folder_path, self.save_filename))

    def load_images(self):
        folder_path = QFileDialog.getExistingDirectory(
            None, "Select Folder to Load Images", ""
        )
        if folder_path:
            self._load_images(os.path.join(folder_path, self.save_filename))

    @property
    def save_filename(self):
        return f"tmp_{self.shape[1]}x{self.shape[0]}_{self.output_shape[1]}x{self.output_shape[0]}"

    def save_filepath(self):
        return os.path.join(
            get_temp_folder(),
            self.save_filename
        )

    def save_on_close(self):
        filepath = self.save_filepath()
        self._save_images(filepath)

    def load_on_open(self):
        filepath = self.save_filepath()
        self._load_images(filepath)

    def _save_images(self, filename: str):
        img.fromarray(self.dem.clip(0, 255).astype(np.uint8)).save(
            filename + "_dem.png"
        )
        img.fromarray(self.landcover).save(filename + "_landcover.png")
        img.fromarray(self.combined_temperature).save(filename + "_temperature.png")
        img.fromarray(self.modal_sketch_stack[:, :, :3]).save(filename + "_modal.png")
        torch.save(self.dem_output, filename + "_dem_output.pt")
        torch.save(self.sat_output, filename + "_sat_output.pt")
        np.save(filename + "_quad_river_upa.npy", self.quad_river_upa)
        np.save(filename + "_normal_river_upa.npy", self.normal_river_upa)
        img.fromarray(
            np.dstack([self.dem_sketch, self.landcover, self.combined_temperature])
        ).save(filename + "_combined.png")

    def _load_images(self, filename: str):
        if os.path.exists(filename + "_dem.png"):
            self.dem = (
                np.array(img.open(filename + "_dem.png").convert("L"))
                .clip(0, 255)
                .astype(np.uint8)
            )
            self._dem_modified = True
            self._uncertainty_modified = True
        if os.path.exists(filename + "_landcover.png"):
            self.landcover = (
                np.array(img.open(filename + "_landcover.png").convert("L"))
                .clip(0, 255)
                .astype(np.uint8)
            )
            self._landcover_modified = True
            self._modal_modified = True
        if os.path.exists(filename + "_dem_output.pt"):
            self._dem_output = torch.load(filename + "_dem_output.pt")
        if os.path.exists(filename + "_sat_output.pt"):
            self._sat_output = torch.load(filename + "_sat_output.pt")
        # if os.path.exists(filename + "_river_sketch.npy"):
        #     self._river_sketch = np.load(filename + "_river_sketch.npy")
        if os.path.exists(filename + "_quad_river_upa.npy"):
            self.quad_river_upa = np.load(filename + "_quad_river_upa.npy")
            self.river_upa_max = self.quad_river_upa.max()
        # TODO: Add proper temperature editing support and then re-enable this
        # self.temperature = np.array(
        #     img.open(filename + "_temperature.png").convert("L")
        # ).clip(0, 255).astype(np.uint8)
        shape = self.river_stack.shape[:2][::-1]
        self.river_stack[:, :, 0] = cv2.resize(
            self.landcover, shape, interpolation=cv2.INTER_NEAREST
        )
        self.river_stack[:, :, 1] = cv2.resize(
            self.combined_temperature, shape, interpolation=cv2.INTER_NEAREST
        )
        self.river_stack[:, :, 2] = cv2.resize(
            self.dem_sketch, shape, interpolation=cv2.INTER_NEAREST
        )
        if os.path.exists(filename + "_normal_river_upa.npy"):
            self.normal_river_upa = np.load(filename + "_normal_river_upa.npy")
        elif self.river_upa_max > 0:
            self.normal_river_upa = quad_to_normal(self.quad_river_upa)
            np.save(filename + "_normal_river_upa.npy", self.normal_river_upa)
        self.river_stack[:, :, 3] = cv2.dilate(
            self.normal_river_upa,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        )
        self.river_stack[:, :, 4] = 0.0
        self.river_stack[:, :, 5] = 1.0
        self.set_display_image(self.dem_sketch, EventTypeEnum.DEM)

    def select_image(
        self,
        metadata: EventMetadata,
        display_controls_callback: callable,
        noise_preview_func: Callable[[np.ndarray], np.ndarray],
    ):
        self.noise_format_func = noise_preview_func
        image_type = metadata.type
        if image_type == self.last_image_type:
            return
        self.process_event(metadata, final=False)
        self.last_image_type = image_type
        display_controls_callback()

    @property
    def combined_temperature(self) -> np.ndarray:
        combined = self.temperature_preset.copy()
        # Make DEM contribute to temperature
        factor = np.ceil(
            ((self.dem.astype(np.float32) / 255) ** self.temp_steepness)
            * self.temp_factor
        ).astype(np.uint8)
        combined[factor > 0] = combined[factor > 0] // factor[factor > 0]
        combined[self.temperature > 0] = self.temperature[self.temperature > 0]
        self._modal_modified = True
        combined = np.clip(combined, 0, 255).astype(np.uint8)
        combined[combined == 0] = 1
        combined = temperature_paint(combined, self.planet_cfg)
        return combined

    @property
    def modal_sketch_stack(self):
        if self._modal_modified or self._modal_sketch_stack is None:
            self._modal_modified = False
            modal_sketch_img = self.modal_sketch.get_sketch(
                self.landcover,
                self.combined_temperature,
                mars_mask=self.landcover == 255,
            )
            self._modal_sketch_stack = np.dstack(
                (
                    modal_sketch_img,
                    self.landcover,
                    self.combined_temperature,
                )
            )
        return self._modal_sketch_stack

    @property
    def uncertainty_sketch(self):
        if self._uncertainty_modified or self._uncertainty_sketch is None:
            self._uncertainty_modified = False
            combined_sketch = (
                self.landcover.astype(np.uint32) * 256**2
                + self.combined_temperature.astype(np.uint32) * 256
                + self.dem_sketch.astype(np.uint32)
            )
            self._uncertainty_sketch = self.uncertainty_sketcher.get_uncertainty_sketch(combined_sketch)
        return self._uncertainty_sketch

    @property
    def dem_sketch(self) -> np.ndarray:
        dem = self.filtered_dem.clip(0, 255).astype(np.uint8)
        return dilate_paint(dem, self.planet_cfg)

    @property
    def filtered_dem(self):
        filtered = self.dem * self.land_mult
        filtered[self.dem == 0] = 0
        filtered[(self.dem > 0) & (filtered < 1)] = 1
        return filtered

    def init_inference_instance(self):
        if self.inference_instance is None:
            self.inference_instance = InferenceInstance(
                replace(self.inference_arguments, do_upscaling=False, tile_size=256),
                self.planet_cfg,
                encoder_override=self.encoder_override,
                show_prediction=False,
            )

            # upscaling_inference_arguments = replace(
            #     self.inference_arguments,
            #     diffusion_model_dir='/mnt/e/models/Sat-upscaling-2-5',
            #     do_upscaling=True,
            #     sketch_injection=False,
            #     guidance_channel=4,
            # )

            # upscale_cfg = replace(
            #     self.planet_cfg,
            #     size=1,
            #     downscale_offset=1,
            #     image_mode="sat-upscaling",
            #     river_upa_mode="channel",
            # )
            # upscaling_instance = InferenceInstance(
            #     upscaling_inference_arguments,
            #     upscale_cfg,
            #     encoder_override=self.encoder_override,
            #     show_prediction=False,
            # )
            self.inference_controller = InferenceController(self.inference_instance)
            # TODO Check if these are the actual previous outputs or the loaded outputs
            self.inference_instance.set_previous_output(
                self._sat_output, self._dem_output
            )

    def init_inference(self, two_phase: bool = False, use_rivers: bool = False, do_upscaling: bool = False):
        self.init_inference_instance()
        assert self.inference_controller is not None
        previous_mask = ~self.regen_mask
        kwargs = {
            "sketches": SketchArgs(
                self.dem_sketch, self.landcover, self.combined_temperature
            ),
        }
        use_previous = previous_mask.any() and self.inference_arguments.use_previous

        if use_previous:
            kwargs.update(
                {
                    "previous_mask": previous_mask,
                    "previous_sat": self._sat_output,
                    "previous_dem": self._dem_output,
                }
            )
        if not two_phase:
            if use_rivers and self.quad_river_upa is not None:
                norm_rivers = self.quad_river_upa / self.river_upa_max * self.river_norm_max
                norm_rivers[norm_rivers < self.river_norm_min] = 0.0
                kwargs["rivers"] = norm_rivers

        # TODO list - upscaling
        # 1. Provide UI for generating low res, generating high res and doing both.
        # 2. Allow partial regeneration in both low and high res
        # 3. Create image types for low res, high res and current outputs
        # 4. Provide better UI and image types for high and low res rivers
        # 5. Improve command line args and allow different inference args
        # 6. Bounding box inference for normal generation

        # TODO list - river upa
        # 1. Add sliders for landcover classes
        # 2. Add sliders for temperature classes
        # 3. Add matrix to show landcover vs temperature contribution

        self.inference_controller.inference_args.use_previous = use_previous
        self.inference_controller.inference_instance.inference_args.use_previous = use_previous

        # self.inference_controller.partial_regenerate(**kwargs)

        if self.inference_thread is None:
            # NOTE: If this breaks then use a OnceOffTask
            self.inference_thread = OnceOffTask(
                target=(
                    # TODO Implement partial regenerate for upscaling
                    self.inference_controller.partial_regenerate
                    if use_previous
                    else self.inference_controller.generate
                    # self.inference_controller.upscale_last_output
                    # if do_upscaling
                    # else self.inference_controller.generate_normal
                ),
                kwargs=kwargs,
            )

    def update_inference_images(
        self, check_stopped: bool = True
    ):
        if (
            self.inference_controller is None
            or self.is_exiting
            or not self.update_images_enabled
        ):
            return
        if not check_stopped or not self.inference_controller.stopped:
            self.update_current_outputs()

    def update_progress(self):
        if (
            self.inference_controller is not None
            and not self.inference_controller.stopped
        ):
            progress = float(self.inference_controller.progress_pcnt)
            if progress == 0:
                self.view.set_progress(None)
            else:
                self.view.set_progress(progress)
        else:
            self.view.set_progress(100)

    def start_generation(self, two_phase: bool = False, use_rivers: bool = False, do_upscaling: bool = False):
        # Check if thread is started
        self.update_images_enabled = True
        if self.inference_thread is not None:
            self.stop_generation()
        self.init_inference(two_phase, use_rivers, do_upscaling)
        self.inference_thread.start()

    def stop_generation(self):
        if self.inference_controller is not None:
            self.inference_controller.stop_generation()
        if self.inference_thread is not None:
            self.inference_thread.stop()
        self.inference_thread = None

    @property
    def sat_output(self):
        if self.inference_controller is None:
            return self._sat_output
        if self.inference_controller.current_sat_output is not None:
            self._sat_output = self.inference_controller.current_sat_output
        return self._sat_output

    @property
    def dem_output(self):
        if self.inference_controller is None:
            return self._dem_output
        if self.inference_controller.current_dem_output is not None:
            self._dem_output = self.inference_controller.current_dem_output
        return self._dem_output

    def display_current_outputs(self):
        ATLAS_STORAGE.set(AtlasType.SAT, self._sat_output)
        ATLAS_STORAGE.set(AtlasType.DEM, self._dem_output)
        self.view.update_globes([AtlasType.SAT])
        self.view.update_globes([AtlasType.DEM])

    def update_current_outputs(self):
        self.dem_output
        self.sat_output

    @profile
    def process_event(self, metadata: EventMetadata, final: bool = False):
        event_type = metadata.type
        x = metadata.x
        y = metadata.y
        self.process_cursor_event(metadata)
        if event_type == EventTypeEnum.LANDCOVER:
            self.process_landcover_event(metadata, final)
        elif event_type == EventTypeEnum.TEMP:
            self.process_temperature_event(metadata, final)
        elif event_type == EventTypeEnum.TEMP_PRESET:
            self.process_temperature_preset_event(metadata, final)
        elif event_type == EventTypeEnum.DEM:
            self.process_dem_event(metadata, final)
        elif event_type == EventTypeEnum.DEM_SKETCH:
            self.process_dem_sketch_event(metadata, final)
        elif event_type == EventTypeEnum.RIVER:
            self.process_river_event(metadata, final)
        elif event_type == EventTypeEnum.MODAL:
            self.process_modal_event(metadata, final)
        elif event_type == EventTypeEnum.REGEN:
            self.process_regen_event(metadata, final)
        else:
            raise ValueError(f"Unknown event type {event_type}")
        if x is not None and y is not None:
            self.last_x = x
            self.last_y = y
        self.update_atlas_storage()

    def update_atlas_storage(self):
        ATLAS_STORAGE.set(AtlasType.DEM_SKETCH, self.dem)
        ATLAS_STORAGE.set(AtlasType.LANDCOVER_SKETCH, self.landcover)
        ATLAS_STORAGE.set(AtlasType.TEMPERATURE_SKETCH, self.temperature_preset)
        ATLAS_STORAGE.set(AtlasType.MODAL_STACK, self.modal_sketch_stack)
        ATLAS_STORAGE.set(AtlasType.UNCERTAINTY_MASK, self.uncertainty_sketch)
        ATLAS_STORAGE.set(AtlasType.RIVERS, self.normal_river_upa)
        ATLAS_STORAGE.set(AtlasType.DEM, self.dem_output)
        ATLAS_STORAGE.set(AtlasType.SAT, self.sat_output)
        ATLAS_STORAGE.set(AtlasType.SELECTION_MASK, self.selection_mask > 0)
        ATLAS_STORAGE.set(
            AtlasType.FILTERED_SELECTION_MASK, self.filtered_selection_mask * 255
        )

    def reset_sketch(self, metadata: EventMetadata):
        # We use slicers here so that the current image also gets modifed instead of creating a new array
        event_type = metadata.type
        if event_type == EventTypeEnum.LANDCOVER:
            self.landcover[:, :] = 0
            self._landcover_modified = True
            self._modal_modified = True
            self._uncertainty_modified = True
        elif (
            event_type == EventTypeEnum.TEMP or event_type == EventTypeEnum.TEMP_PRESET
        ):
            self.temperature[:, :] = 0
            self._temperature_modified = True
            self._modal_modified = True
            self._uncertainty_modified = True
        elif event_type == EventTypeEnum.DEM or event_type == EventTypeEnum.DEM_SKETCH:
            self.dem[:, :] = 0
            self._dem_modified = True
            self._uncertainty_modified = True
        elif event_type == EventTypeEnum.RIVER:
            self.quad_river_upa[:, :] = 0.0
            self.normal_river_upa[:, :] = 0.0
            self.river_stack[:, :, 3] = 0
        elif event_type == EventTypeEnum.REGEN:
            self.regen_mask[:, :] = False
        else:
            raise ValueError(f"Unknown event type {event_type}")
        self.process_event(metadata)

    def set_display_image(
        self,
        image: np.ndarray,
        image_type: EventTypeEnum,
        display_post_process: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    ):
        self._set_post_process(display_post_process)
        self.current_display_image = image
        self.current_display_type = image_type
        atlas_types = []
        if image_type == EventTypeEnum.DEM:
            atlas_types = [AtlasType.DEM_SKETCH]
        elif image_type == EventTypeEnum.DEM_SKETCH:
            atlas_types = [
                AtlasType.DEM_SKETCH,
                AtlasType.FILTERED_SELECTION_MASK,
                AtlasType.SELECTION_MASK,
                AtlasType.UNCERTAINTY_MASK,
            ]
        elif image_type == EventTypeEnum.TEMP:
            atlas_types = [AtlasType.TEMPERATURE_SKETCH, AtlasType.MODAL_STACK, AtlasType.UNCERTAINTY_MASK]
        elif image_type == EventTypeEnum.TEMP_PRESET:
            atlas_types = [AtlasType.TEMPERATURE_SKETCH, AtlasType.MODAL_STACK, AtlasType.UNCERTAINTY_MASK]
        elif image_type == EventTypeEnum.LANDCOVER:
            atlas_types = [
                AtlasType.LANDCOVER_SKETCH,
                AtlasType.MODAL_STACK,
                AtlasType.FILTERED_SELECTION_MASK,
                AtlasType.SELECTION_MASK,
                AtlasType.UNCERTAINTY_MASK,
            ]
        elif image_type == EventTypeEnum.RIVER:
            atlas_types = [AtlasType.RIVERS]
        elif image_type == EventTypeEnum.MODAL:
            atlas_types = [AtlasType.MODAL_STACK]
        self.update_atlas_storage()
        if atlas_types:
            self.view.update_globes(atlas_types)
        self.display_current_input()

    def refresh_display(self):
        self.display_current_input()

    def display_current_input(self):
        if self.current_display_image is None:
            return
        self.view.set_image(self.current_display_image)

    def _set_post_process(self, post_process_func: Callable[[np.ndarray], np.ndarray]):
        self.view.set_input_post_process(post_process_func)

    def process_cursor_event(self, metadata: EventMetadata):
        x = metadata.x
        y = metadata.y
        brush_size = metadata.brush_size
        lasso_points = np.array(self.lasso_points)
        h, w = (
            self.current_display_image.shape[:2]
            if self.current_display_image is not None
            else self.shape
        )
        if len(lasso_points) > 0:
            brush_size = 0
        ys = None
        xs = None
        if y is not None and x is not None:
            kernel = get_brush_kernel(brush_size)
            # Pad the kernel with a border of 1
            kernel = cv2.copyMakeBorder(
                kernel, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0
            )
            dilate_kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
            kernel = cv2.dilate(kernel, dilate_kernel, iterations=1) - kernel
            ys, xs = np.where(kernel > 0)
            ys -= brush_size + 1
            xs -= brush_size + 1
            ys += y
            xs += x
            ys = np.clip(ys, 0, h - 1)
            xs = xs % w

        if len(lasso_points) > 0:
            lasso_ys = lasso_points[:, 1]
            lasso_xs = lasso_points[:, 0]
            lasso_ys = np.clip(lasso_ys, 0, h - 1)
            lasso_xs = lasso_xs % w
            # Add lasso points to cursor points
            if ys is not None and xs is not None:
                ys = np.concatenate([ys, lasso_ys])
                xs = np.concatenate([xs, lasso_xs])
            else:
                ys = lasso_ys
                xs = lasso_xs
        self.set_cursor(ys, xs)

    def set_cursor(self, ys: np.ndarray | None, xs: np.ndarray | None):
        ys_equal = (
            isinstance(ys, np.ndarray)
            and isinstance(self.cursor_ys, np.ndarray)
            and np.all(ys == self.cursor_ys)
        ) or (ys is None and self.cursor_ys is None)
        xs_equal = (
            isinstance(xs, np.ndarray)
            and isinstance(self.cursor_xs, np.ndarray)
            and np.all(xs == self.cursor_xs)
        ) or (xs is None and self.cursor_xs is None)
        has_changed = not (ys_equal and xs_equal)
        if has_changed:
            self.cursor_shape = (
                self.current_display_image.shape[:2]
                if self.current_display_image is not None
                else self.shape
            )
            if self.cursor_mask.shape != self.cursor_shape:
                new_cursor_mask = np.zeros(self.cursor_shape, dtype=bool)
                if ys is not None and xs is not None:
                    new_cursor_mask[ys, xs] = True
                self.cursor_mask = new_cursor_mask
            else:
                if self.cursor_ys is not None and self.cursor_xs is not None:
                    self.cursor_mask[self.cursor_ys, self.cursor_xs] = False
                if ys is not None and xs is not None:
                    self.cursor_mask[ys, xs] = True
            self.view.cursor_mask_update(self.cursor_mask)
            self.cursor_ys = ys
            self.cursor_xs = xs

    def finish_draw(self, metadata: EventMetadata):
        self.process_event(metadata, final=True)
        self.last_x = None
        self.last_y = None

    def import_dem(self, filename: str | None):
        if filename is None:
            return
        dem = np.array(img.open(filename))
        if len(dem.shape) > 2:
            dem = dem[:, :, 0]
        dem = (dem / dem.max() * 255).clip(0, 255)
        h, w = dem.shape
        # desired shape is so that it fits on a single face
        desired_h = self.shape[0]
        desired_w = self.shape[1]
        # This DEM can be any shape so we want to resize it so
        # that one of the dimensions matches the shape of the current DEM.
        # Then we can just put it in the center of the current DEM
        if 2 * h > w:
            # Too tall so resize to fit shape
            new_h = desired_h
            new_w = int(w * new_h / h)
        else:
            # Too wide so resize to fit shape
            new_w = desired_w
            new_h = int(h * new_w / w)
        dem = cv2.resize(dem, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4).clip(0, 255)
        h, w = dem.shape
        x = self.shape[1] // 2 - w // 2
        y = self.shape[0] // 2 - h // 2
        self.dem[y: y + h, x: x + w] = dem
        self.derive_landcover()
        self.land_mult = np.ones(self.shape, dtype=np.float32)
        self._dem_modified = True
        self._uncertainty_modified = True
        self.set_display_image(self.dem_sketch, EventTypeEnum.DEM)

    def import_landcover(self, filename: str | None):
        if filename is None:
            return
        landcover = np.array(img.open(filename).convert("L"))
        if landcover.shape != self.shape:
            landcover = cv2.resize(
                landcover, (self.shape[1], self.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        self.landcover = translate_land(landcover)
        self._landcover_modified = True
        self._modal_modified = True
        self._uncertainty_modified = True
        self.set_display_image(self.landcover, EventTypeEnum.LANDCOVER)

    def derive_landcover(self):
        # TODO Move this to a proper button and improve control
        self.landcover = derive_landcover(self.dem_sketch, self.combined_temperature, self.noise_preview / 255)
        self._landcover_modified = True
        self._modal_modified = True
        self._uncertainty_modified = True
        self.update_atlas_storage()

    def derive_dem(self):
        """
        Derive a dem sketch from the generated dem.
        Basically just converts the quad dem to a normal dem and resizes it.
        """
        small_quad_sphere = QuadSphere(shape=self.shape, discrete=True)
        _qh, _qw = small_quad_sphere.quad_shape
        dem_output = tensor_to_np(self._dem_output)
        dem_output = cv2.resize(
            dem_output, (_qw, _qh), interpolation=cv2.INTER_LANCZOS4
        )
        dem = dem_output.clip(0, self.dem_max).astype(np.uint8)
        dem = (dem / self.dem_max * 255).astype(np.uint8)
        self.dem = small_quad_sphere.get_normal_atlas(dem)
        self.landcover[(self.dem > 0) & (self.landcover == 0)] = (
            LandcoverClasses.TREE_COVER.gray_colour
        )
        self.landcover[(self.dem == 0) & (self.landcover > 0)] = 0
        self.land_mult = np.ones(self.shape, dtype=np.float32)
        self.set_display_image(self.dem_sketch, EventTypeEnum.DEM)

    def derive_rivers(self, derivation_settings: RiverDerivationSettings):
        with wait_cursor():
            self._derive_rivers(derivation_settings)

    def _derive_rivers(self, derivation_settings: RiverDerivationSettings):
        manual_weights = self.river_stack[:, :, 4]

        river_weights = self.river_stack[:, :, 6].copy()
        river_weights[manual_weights > 0] = manual_weights[manual_weights > 0]

        river_efficiency = self.river_stack[:, :, 5]
        river_efficiency = cv2.resize(river_efficiency, (manual_weights.shape[::-1]), interpolation=cv2.INTER_NEAREST)

        self.quad_river_upa, self.normal_river_upa = derive_sketch_rivers(
            SketchArgs(
                self.dem_sketch,
                self.landcover,
                self.combined_temperature,
            ),
            self.planet_cfg,
            river_weights,
            river_efficiency,
            noise=None,
            noise_factor=0.5,
            normalize=False,
            return_both=True,
            resolution_scale=0.5,
        )
        self.river_upa_max = self.quad_river_upa.max()
        self.river_stack[:, :, 3] = self.normal_river_upa[:, :]
        self.refresh_display()
        return

    def dem_sketch_postprocess_function(
        self,
        # TODO Remove these
        threshold: float,
        min_value: float,
        max_value: float,
    ) -> Callable[[np.ndarray], np.ndarray]:
        def dem_sketch_postprocess(preview: np.ndarray) -> np.ndarray:
            if len(preview.shape) <= 2 or preview.shape[2] != 4:
                print("Incorrect image passed to postprocess function")
                return preview
            dem = preview[:, :, 0]
            selection = (preview[:, :, 1] * 255).astype(np.uint8)
            dilated = cv2.dilate(selection, np.ones((3, 3), dtype=np.uint8))
            edge = dilated - selection
            noise = preview[:, :, 2]
            mean_filtered = preview[:, :, 3]
            new_preview = self.process_dem_sketch(
                dem, noise, mean_filtered
            )
            painted_preview = dilate_paint(
                new_preview.clip(0, 255).astype(np.uint8), self.planet_cfg
            )
            painted_preview[edge > 0] = (new_preview[edge > 0] < 128) * 255
            return painted_preview

        return dem_sketch_postprocess

    def process_dem_sketch(
        self,
        dem: np.ndarray,  # 0.0 -> 255.0
        noise: np.ndarray,  # 0.0 -> 255.0
        mean_filtered: np.ndarray,  # 0.0 -> 1.0
    ) -> np.ndarray:
        if mean_filtered.max() == mean_filtered.min():
            normalised_mean_filter = np.ones_like(mean_filtered) * mean_filtered.min()
        else:
            normalised_mean_filter = (mean_filtered - mean_filtered.min()) / (
                mean_filtered.max() - mean_filtered.min()
            )
        assert normalised_mean_filter.min() >= 0.0
        assert normalised_mean_filter.max() <= 1.0
        new_noise = normalised_mean_filter + noise.astype(np.float32) / 255  # 0.0 -> 2.0
        assert new_noise.min() >= 0.0
        assert new_noise.max() <= 2.0
        new_noise = np.maximum(new_noise - 1.0, 0.0)  # 0.0 -> 2.0
        assert new_noise.min() >= 0.0
        assert new_noise.max() <= 1.0
        new_noise = (1.0 - normalised_mean_filter) * dem.astype(
            np.float32
        ) + normalised_mean_filter * new_noise * 255  # 0.0 -> 255.0
        return new_noise

    def process_dem_sketch_event(self, metadata: DEMSketchEventMetadata, final: bool):
        x = metadata.x
        y = metadata.y
        brush_size = metadata.brush_size
        lasso = metadata.tool == ToolEnum.LASSO
        roughness = metadata.roughness
        lock_ocean = metadata.lock_ocean
        edge_distance_settings = metadata.edge_distance_settings
        threshold = metadata.threshold
        min_value = metadata.min_value
        max_value = metadata.max_value
        edit_type = metadata.edit_type

        if edit_type == EditTypeEnum.DRAW:
            if lasso and not final:
                self.add_lasso_points(x, y, wrap_around_width=self.shape[1])
                return

            if x is None or y is None:
                mask = np.zeros_like(self.dem)
            else:
                if metadata.tool == ToolEnum.FILL:
                    ys, xs = np.where(np.zeros_like(self.dem) == 0)
                else:
                    ys, xs = self.get_paint_points(
                        x,
                        y,
                        lasso,
                        final,
                        brush_size,
                        *self.dem.shape,
                        wrap_around_width=self.shape[1],
                    )
                mask = self.get_modification_mask(
                    ys,
                    xs,
                    brush_size,
                    np.ones_like(self.dem),
                    roughness,
                    lock_ocean,
                    land_mask=self.dem,
                )

            self.selection_mask[mask > 0] = not metadata.erase
            self.filtered_selection_mask = edge_distance_settings.apply_filter(
                self.selection_mask
            )
        elif edit_type == EditTypeEnum.CONFIRM:
            self.dem = (
                self.process_dem_sketch(
                    self.dem,
                    self.noise_preview,
                    edge_distance_settings.apply_filter(self.selection_mask),
                )
                .clip(0, 255)
                .astype(np.uint8)
            )
            self.selection_mask[:, :] = False
            new_landcover = self.landcover.copy()
            new_landcover[(self.dem > 0) & (self.landcover == 0)] = (
                LandcoverClasses.TREE_COVER.gray_colour
            )
            new_landcover[(self.dem == 0) & (self.landcover > 0)] = 0
            modified = new_landcover != self.landcover
            self.landcover = new_landcover
            self.regen_mask[modified] = True
            self._dem_modified = True
            self._uncertainty_modified = True
            self._landcover_modified = modified.any()
            self._modal_modified |= self._landcover_modified
            self.filtered_selection_mask = np.zeros_like(self.dem)
        elif edit_type == EditTypeEnum.CANCEL:
            self.selection_mask[:, :] = False
            self.filtered_selection_mask = np.zeros_like(self.dem)
        elif edit_type == EditTypeEnum.OTHER_CHANGE:
            self.filtered_selection_mask = edge_distance_settings.apply_filter(
                self.selection_mask
            )
        else:
            EditTypeEnum(edit_type)

        self.set_display_image(
            np.dstack(
                [
                    self.dem.astype(np.float32),
                    self.selection_mask.astype(np.float32),
                    self.noise_preview.astype(np.float32),
                    self.filtered_selection_mask.astype(np.float32),
                ]
            ),
            EventTypeEnum.DEM_SKETCH,
            self.dem_sketch_postprocess_function(threshold, min_value, max_value),
        )

    def process_dem_event(self, metadata: DEMEventMetadata, final: bool):
        x = metadata.x
        y = metadata.y
        brush_size = metadata.brush_size
        lasso = metadata.lasso
        roughness = metadata.roughness
        lock_ocean = metadata.lock_ocean
        display_sketch = metadata.display_sketch
        mean_filter = metadata.mean_filter
        self.set_both_coords(metadata.view_coords, metadata.preview_coords)

        if lasso and not final:
            self.add_lasso_points(x, y, wrap_around_width=self.shape[1])
            self.refresh_display()
            return
        if x is not None and y is not None:

            if metadata.fill:
                ys, xs = np.where(np.zeros_like(self.dem) == 0)
            else:
                ys, xs = self.get_paint_points(
                    x,
                    y,
                    lasso,
                    final,
                    brush_size,
                    *self.dem.shape,
                    wrap_around_width=self.shape[1],
                )
            mask = self.get_modification_mask(
                ys,
                xs,
                brush_size,
                np.ones_like(self.dem),
                roughness,
                lock_ocean,
                land_mask=self.dem,
            )

            noise = self.noise_preview
            self.dem[mask] = noise[mask]
            self.regen_mask[mask] = True
            land_mask = self.dem > 0
            if final:
                self.land_mask = land_mask
                self.land_mult = mean_filter.apply_filter(self.dem > 0)
            else:
                self.land_mult[mask] = 1.0
            new_landcover = self.landcover.copy()
            new_landcover[(self.dem > 0) & (self.landcover == 0)] = (
                LandcoverClasses.TREE_COVER.gray_colour
            )
            new_landcover[(self.dem == 0) & (self.landcover > 0)] = 0
            modified = new_landcover != self.landcover
            self.landcover = new_landcover
            self.regen_mask[modified] = True
            self._dem_modified = True
            self._uncertainty_modified = True
            self._landcover_modified = modified.any()
            self._modal_modified |= self._landcover_modified
        if display_sketch:
            self.set_display_image(self.dem_sketch, EventTypeEnum.DEM)
        else:
            self.set_display_image(self.filtered_dem, EventTypeEnum.DEM)

    def process_temperature_event(self, metadata, final: bool):
        pass

    def create_temperature_preset(self, metadata: TempPresetEventMetadata):
        self._temperature_modified = True
        self._modal_modified = True
        self._uncertainty_modified = True
        new_temperature_preset = create_temp_preset(
            metadata, self.temperature_preset.shape
        )
        # TODO Consider adding this back but it causes too many updates at the moment
        # modify_mask = (new_temperature_preset != self.temperature_preset)
        # self.regen_mask[modify_mask] = True
        self.temperature_preset = new_temperature_preset

    def process_temperature_preset_event(
        self, metadata: TempPresetEventMetadata, final: bool
    ):
        # TODO Process these events
        # when dem changes to make sure the height-temperature relationship is preserved

        # If these are both equal, then it is something important that changed
        if (
            self.temp_steepness == metadata.steepness
            and self.temp_factor == metadata.factor
        ):
            # Don't recreate the preset if the steepness or factor change as it's expensive
            # Steepness and factor is used in combined_temperature instead
            self.create_temperature_preset(metadata)
        self.temp_steepness = metadata.steepness
        self.temp_factor = metadata.factor
        self._modal_modified = True
        self._uncertainty_modified = True
        if metadata.modal_view:
            self.set_display_image(self.modal_sketch_stack, EventTypeEnum.TEMP_PRESET, modal_stack_display_func)
        else:
            self.set_display_image(
                self.combined_temperature, EventTypeEnum.TEMP_PRESET, add_icons_to_temperature_sketch
            )

    def process_landcover_sketch(
        self,
        landcover: np.ndarray,  # 0.0 -> 255.0
        noise: np.ndarray,  # 0.0 -> 255.0
        mean_filtered: np.ndarray,  # 0.0 -> 1.0
        metadata: LandCoverEventMetadata,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert mean_filtered.min() >= 0.0
        assert mean_filtered.max() <= 1.0
        new_noise = noise.astype(np.float32) / 255  # 0.0 -> 1.0
        assert new_noise.min() >= 0.0
        assert new_noise.max() <= 2.0
        if new_noise.max() == new_noise.min():
            normalised = np.full_like(new_noise, new_noise.min())
        else:
            normalised = (new_noise - new_noise.min()) / (
                new_noise.max() - new_noise.min()
            )  # 0.0 -> 1.0
        if mean_filtered.max() == mean_filtered.min():
            normalised_mean_filter = np.ones_like(mean_filtered) * mean_filtered.min()
        else:
            normalised_mean_filter = (mean_filtered - mean_filtered.min()) / (
                mean_filtered.max() - mean_filtered.min()
            )
        new_noise = normalised_mean_filter + normalised  # 0.0 -> 2.0
        threshold = 0.75  # metadata.threshold
        original_mask = (new_noise < threshold) | (normalised_mean_filter < 0.001)

        primary_mask = np.zeros(landcover.shape, dtype=bool)
        secondary_mask = np.zeros(landcover.shape, dtype=bool)

        secondary_mask[normalised > metadata.primary_ratio] = True
        primary_mask[
            (normalised <= metadata.primary_ratio) & (normalised_mean_filter > 0)
        ] = True
        primary_mask[original_mask] = False
        secondary_mask[original_mask] = False

        return primary_mask, secondary_mask

    def landcover_sketch_postprocess_function(
        self,
        metadata: LandCoverEventMetadata,
    ) -> Callable[[np.ndarray], np.ndarray]:
        def landcover_sketch_postprocess(preview: np.ndarray) -> np.ndarray:
            landcover = preview[:, :, 0]
            selection = (preview[:, :, 1] * 255).astype(np.uint8)
            dilated = cv2.dilate(selection, np.ones((3, 3), dtype=np.uint8))
            edge = dilated - selection
            noise = preview[:, :, 2]
            mean_filtered = preview[:, :, 3]
            temperature = preview[:, :, 4]
            primary_mask, secondary_mask = self.process_landcover_sketch(
                landcover, noise, mean_filtered, metadata
            )
            new_preview = landcover.copy().astype(np.uint8)
            if metadata.primary_class is not None:
                new_preview[primary_mask] = metadata.primary_class.colour
            if metadata.secondary_class is not None:
                new_preview[secondary_mask] = metadata.secondary_class.colour
            if metadata.modal_view:
                painted_preview = self.modal_sketch.get_sketch(
                    new_preview, temperature, mars_mask=new_preview == 255
                )
                painted_preview = modal_stack_display_func(np.dstack([painted_preview, landcover, temperature]))
            else:
                painted_preview = gray_to_land(new_preview)
            edge_ys, edge_xs = np.where(edge)
            edge_pixels = painted_preview[edge_ys, edge_xs]
            if len(edge_pixels) > 0:
                painted_preview[edge_ys, edge_xs] = np.dstack(
                    [(edge_pixels.mean(axis=1) < 128) * 255] * 3
                )
            return painted_preview

        return landcover_sketch_postprocess

    def process_landcover_event(self, metadata: LandCoverEventMetadata, final: bool):
        noise = metadata.noise
        roughness = metadata.roughness
        # TODO Cache noise better
        if metadata.primary_subclass is not None:
            temp_colour = metadata.primary_subclass.colour
        else:
            temp_colour = None
        x = metadata.x
        y = metadata.y
        lasso = metadata.tool == ToolEnum.LASSO
        self.set_both_coords(metadata.view_coords, metadata.preview_coords)
        # preset = format_landcover_noise(self.noise_preview, metadata)
        edit_type = metadata.edit_type
        edge_distance_settings = metadata.edge_distance_settings

        if edit_type == EditTypeEnum.DRAW:
            if x is not None and y is not None:
                brush_size = metadata.brush_size
                if lasso and not final:
                    self.add_lasso_points(x, y, wrap_around_width=self.shape[1])
                    self.refresh_display()
                    return
                if metadata.tool == ToolEnum.FILL:
                    # TODO Implement max_difference
                    ys, xs = np.where(np.zeros_like(self.landcover) == 0)
                else:
                    ys, xs = self.get_paint_points(
                        x,
                        y,
                        lasso,
                        final,
                        brush_size,
                        *self.landcover.shape,
                        wrap_around_width=self.shape[1],
                    )
                mask = self.get_modification_mask(
                    ys,
                    xs,
                    brush_size,
                    noise,
                    roughness,
                    metadata.lock_ocean,
                    land_mask=self.landcover,
                )
            self.selection_mask[mask > 0] = not metadata.erase
            self.filtered_selection_mask = edge_distance_settings.apply_filter(
                self.selection_mask
            )
        elif edit_type == EditTypeEnum.CONFIRM:
            mask = self.selection_mask
            primary_mask, secondary_mask = self.process_landcover_sketch(
                self.landcover,
                self.noise_preview,
                self.filtered_selection_mask,
                metadata,
            )

            self.landcover[primary_mask] = metadata.primary_class.colour
            if metadata.secondary_class is not None:
                self.landcover[secondary_mask] = metadata.secondary_class.colour
            self.regen_mask[mask] = True
            if temp_colour is not None:
                self.temperature[mask] = temp_colour
                self._temperature_modified = True
                self._modal_modified = True
                self._uncertainty_modified = True
            modify_mask = (self.landcover == 0) & (self.dem > 0)
            self.dem[modify_mask] = 0
            self.regen_mask[modify_mask] = True
            modify_mask = (self.landcover > 0) & (self.dem == 0)
            self.dem[modify_mask] = 64
            self.regen_mask[modify_mask] = True
            self._landcover_modified = True
            self._modal_modified = True
            self._dem_modified = True
            self._uncertainty_modified = True
            self.selection_mask[:, :] = False
            self.filtered_selection_mask[:, :] = 0.0
        elif edit_type == EditTypeEnum.CANCEL:
            self.selection_mask[:, :] = False
            self.filtered_selection_mask[:, :] = 0.0
        elif edit_type == EditTypeEnum.OTHER_CHANGE:
            self.filtered_selection_mask = edge_distance_settings.apply_filter(
                self.selection_mask
            )

        self.set_display_image(
            np.dstack(
                [
                    self.landcover,
                    self.selection_mask,
                    self.noise_preview.astype(np.float32),
                    self.filtered_selection_mask.astype(np.float32),
                    self.combined_temperature,
                ]
            ),
            EventTypeEnum.LANDCOVER,
            self.landcover_sketch_postprocess_function(metadata),
        )

    def add_lasso_points(self, x: int, y: int, wrap_around_width: int | None = None):
        if x is None or y is None:
            return
        if self.last_x is None or self.last_y is None:
            return
        points = get_line((self.last_x, self.last_y), (x, y), wrap_around_width)
        self.lasso_points.extend(points)

    def get_points(
        self,
        x: int,
        y: int,
        lasso: bool,
        final: bool,
        wrap_around_width: int | None = None,
    ) -> list[tuple[int, int]]:
        if self.last_x is None or self.last_y is None or x is None or y is None:
            points = [(x, y)] if x is not None and y is not None else []
        else:
            points = get_line((self.last_x, self.last_y), (x, y), wrap_around_width)
        if lasso and final and len(self.lasso_points) > 0:
            points.extend(
                get_line(self.lasso_points[0], self.lasso_points[-1], wrap_around_width)
            )
        return points

    @profile
    def get_paint_points(
        self,
        x: int,
        y: int,
        lasso: bool,
        final: bool,
        brush_size: int,
        h: int,
        w: int,
        wrap_around_width: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        points = self.get_points(x, y, lasso, final, wrap_around_width)
        ys, xs = get_paint_points(points, h, w, brush_size, wrap_around_width)
        ys = ys.astype(int)
        xs = xs.astype(int)
        if lasso and final and len(self.lasso_points) > 0:
            try:
                extra_ys, extra_xs = get_lasso_points(
                    self.lasso_points, (h, w), wrap_around_width
                )
                ys = np.concatenate([ys, extra_ys.astype(int)])
                xs = np.concatenate([xs, extra_xs.astype(int)])
            finally:
                self.lasso_points = []
        ys = np.clip(ys, 0, h - 1)
        xs = xs % w
        return ys, xs

    def get_modification_mask(
        self,
        ys: np.ndarray,
        xs: np.ndarray,
        brush_size: int,
        noise: np.ndarray,
        roughness: float,
        lock_ocean: bool,
        land_mask: np.ndarray = None,
    ) -> np.ndarray:
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        original_mask = np.zeros_like(noise, dtype=np.uint8)
        original_mask[ys, xs] = 1
        dilated_mask = original_mask.copy()
        # TODO Improve this brush roughness
        for i in range(brush_size * 2):
            dilated_mask = cv2.dilate(dilated_mask, kernel, iterations=1)
            dilated_mask[original_mask > 0] += 1
        if dilated_mask.max() > 0:
            dilated_mask = dilated_mask / dilated_mask.max()
        mask = (dilated_mask * (noise * roughness + 1.0)) >= 1.0
        if lock_ocean:
            assert land_mask is not None
            mask[land_mask == 0] = False
        return mask

    def river_post_process(self, display_type: RiverDisplayType):
        def _river_post_process(preview: np.ndarray) -> np.ndarray:
            """
            Postprocessing for the river.
            """
            landcover = preview[:, :, 0]
            temperature = preview[:, :, 1]
            dem_sketch = preview[:, :, 2]
            # dem_sketch = cv2.normalize(dem_sketch, None, 0, 255, cv2.NORM_MINMAX)
            river_upa = preview[:, :, 3]
            river_weights = preview[:, :, 4]
            river_efficiency = preview[:, :, 5]
            if display_type == RiverDisplayType.MODAL:
                tmp_river_sketch = self.river_modal_sketch.get_sketch(
                    landcover, temperature, (river_upa > 0).astype(np.uint8) * 255
                )
                full_sketch = self.modal_sketch.get_sketch(
                    landcover, temperature, mars_mask=landcover == 255
                )
                full_sketch[river_upa > 0] = tmp_river_sketch[river_upa > 0]
            elif display_type == RiverDisplayType.DEM:
                full_sketch = np.dstack([dem_sketch] * 3)
                full_sketch[river_upa > 0] = full_sketch[river_upa > 0] // 2
                full_sketch[:, :, 2][river_upa > 0] += 128
            elif display_type == RiverDisplayType.UPA:
                normalised_upa = np.round(
                    (river_upa / max(self.river_upa_max, 1.0) * self.river_norm_max)
                ).clip(0, 255).astype(np.uint8)
                normalised_upa[normalised_upa < self.river_norm_min] = 0.0
                viridis_rivers = np_rgb(normalised_upa, cmap='viridis')
                full_sketch = np.dstack([dem_sketch] * 3)
                full_sketch[normalised_upa > 0] = viridis_rivers[normalised_upa > 0]
            elif display_type == RiverDisplayType.WEIGHTS:
                manual_weights = preview[:, :, 4]
                river_weights = preview[:, :, 6].copy()
                river_weights[manual_weights > 0] = manual_weights[manual_weights > 0]
                normalised_weights = np_rgb((river_weights * 255).clip(0, 255), cmap="Blues")
                full_sketch = np.dstack([dem_sketch] * 3)
                normalised_weights[full_sketch == 0] = 0
                full_sketch = normalised_weights
            elif display_type == RiverDisplayType.EFFICIENCY:
                normalised_efficiency = (river_efficiency * 255).clip(0, 255).astype(np.uint8)
                full_sketch = np.dstack([dem_sketch] * 3)
                full_sketch[:, :, 1] = normalised_efficiency
            else:
                raise ValueError(f"Unknown display type {display_type}")
            return full_sketch

        return _river_post_process

    def sketch_precip(self, derivation_settings: RiverDerivationSettings):
        h, w = self.river_stack.shape[:2]
        self.river_stack[:, :, 6] = cv2.resize(
            self.precip_sketcher.get_sketch(
                self.dem_sketch,
                self.landcover,
                self.combined_temperature,
                derivation_settings.full_mapping,
            ) + derivation_settings.base_weight,
            (w, h), interpolation=cv2.INTER_NEAREST
        )
        self.view.update_globes([AtlasType.RIVERS])

    def process_river_event(self, metadata: RiverEventMetadata, final: bool):
        x = metadata.x
        y = metadata.y
        brush_size = metadata.brush_size
        erase = metadata.erase
        display_type = metadata.display_type

        if (
            self.last_river_meta is not None and
            self.last_river_meta.display_type != RiverDisplayType.WEIGHTS and
            metadata.display_type == RiverDisplayType.WEIGHTS
        ):
            self.sketch_precip(metadata.settings)

        self.river_norm_max = metadata.river_max
        self.river_norm_min = metadata.river_min
        if y is not None and x is not None:
            h, w = self.river_stack[:, :, 4].shape
            ys, xs = self.get_paint_points(
                x, y, False, final, brush_size, h, w
            )
            # self.river_stack[ys, xs, 3] = 0 if erase else 255
            self.river_stack[ys, xs, 4] = 0 if erase else metadata.weight_value
            self.river_stack[ys, xs, 5] = 0 if erase else metadata.efficiency_value
            # self.river_upa[ys, xs] = False if erase else True
            regen_h, regen_w = self.regen_mask.shape
            delta = h // regen_h
            self.regen_mask[ys // delta, xs // delta] = True
        _h, _w = self.river_stack.shape[:2]
        if self._landcover_modified:
            self.river_stack[:, :, 0] = cv2.resize(
                self.landcover, (_w, _h), interpolation=cv2.INTER_NEAREST
            )
            self._landcover_modified = False
        if self._temperature_modified:
            self.river_stack[:, :, 1] = cv2.resize(
                self.combined_temperature, (_w, _h), interpolation=cv2.INTER_NEAREST
            )
            self._temperature_modified = False
        if self._dem_modified:
            self.river_stack[:, :, 2] = cv2.resize(
                self.dem_sketch, (_w, _h), interpolation=cv2.INTER_NEAREST
            )
            self._dem_modified = False

        self.last_river_meta = metadata
        self.set_display_image(
            self.river_stack, EventTypeEnum.RIVER,
            self.river_post_process(display_type)
        )

    def process_modal_event(self, metadata, final: bool):
        # TODO Change all modal views to use a post process function
        modal_sketch = self.modal_sketch.get_sketch(
            self.landcover, self.combined_temperature, mars_mask=self.landcover == 255
        )
        self.set_display_image(modal_sketch, EventTypeEnum.MODAL)

    def process_regen_event(self, metadata: RegenEventMetadata, final: bool):
        x = metadata.x
        y = metadata.y
        if x is not None and y is not None:
            brush_size = metadata.brush_size
            if not final:
                self.add_lasso_points(x, y, wrap_around_width=self.shape[1])
                self.refresh_display()
                return
            ys, xs = self.get_paint_points(
                x,
                y,
                True,
                final,
                brush_size,
                *self.regen_mask.shape,
                wrap_around_width=self.shape[1],
            )
            if metadata.erase:
                self.regen_mask[ys, xs] = False
            else:
                self.regen_mask[ys, xs] = True
        display_mask = self.regen_mask.astype(np.uint8) * 128 + 127
        self.set_display_image(display_mask, EventTypeEnum.REGEN)
