import logging
from math import log2
from typing import Generator
import torch
import numpy as np
import cv2
from PIL import Image

from skimage.morphology import skeletonize
from planetAI.src.data.mean_precip import PrecipSketch
from planetAI.src.data.sketch_gen import accumulation
from planetAI.src.data.sphere_mapping import (
    QuadSphere,
    get_normal_shape,
    get_quad_shape,
    connectivity,
)
from planetAI.src.data.utils import get_bounds, tensor_to_np, PlanetConfig, timing
from ...interface.utils import SketchArgs

from .inpaint_inference import InferenceInstance


Image.MAX_IMAGE_PIXELS = 933120000


def can_generate_faces(faces: set[int]) -> bool:
    # TODO Actually check for pixels near the edges,
    # because we can generate objects as long as they are fully on separate faces
    if len(faces) == 1:
        return True
    if 0 in faces or 5 in faces:
        # Definitely can't do these with others
        return False
    if 1 in faces and 4 in faces:
        # Can't handle wrap around currently
        return False
    return True


def can_generate_edges(combined_edges: set[int]) -> bool:
    alt_connect = {}
    for (face1, face2), (edge1, edge2) in connectivity.items():
        alt_connect[(face1, edge1)] = (face2, edge2)
    for val in combined_edges:
        face = val // 4
        edge = val % 4
        connect = alt_connect.get((face, edge))
        if connect is None:
            continue
        if face == 0 or face == 5:
            return False
        if face == 1 and face2 == 4:
            return False
    return True


def get_tile_coords_to_generate(
    content_mask: np.ndarray,
    delta: int
) -> Generator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
    num_components, component_mask = cv2.connectedComponents(content_mask)
    quad_component_mask = QuadSphere(component_mask, discrete=True).quad_sphere_atlas
    for comp in range(num_components):
        if comp == 0:
            continue
        mask = (quad_component_mask == comp)
        ymin, xmin, ymax, xmax = get_bounds(mask)
        big_ymin = ymin * delta
        big_xmin = xmin * delta
        big_ymax = ymax * delta
        big_xmax = xmax * delta
        big_ys = np.arange(big_ymin, big_ymax)
        big_xs = np.arange(big_xmin, big_xmax)
        big_xs, big_ys = np.meshgrid(big_xs, big_ys)
        small_ys = np.arange(ymin, ymax)
        small_xs = np.arange(xmin, xmax)
        small_xs, small_ys = np.meshgrid(small_xs, small_ys)
        yield (big_ys, big_xs, small_ys, small_xs)


@timing
def quad_rivers(sketches: SketchArgs, planet_cfg: PlanetConfig, quad_dem: np.ndarray):
    precip_sketcher = PrecipSketch(planet_cfg)
    full_mapping = precip_sketcher.get_full_mapping()
    river_weights = precip_sketcher.get_sketch(
        sketches.quad_downsketch,
        sketches.quad_downland_sketch,
        sketches.quad_downtemp_sketch,
        full_mapping,
    )

    h, w = river_weights.shape
    assert w == 6 * h

    face_width, full_width = quad_dem.shape
    assert full_width == face_width * 6

    river_weights = cv2.resize(river_weights, (full_width, face_width), interpolation=cv2.INTER_LANCZOS4)
    rivers = np.zeros_like(river_weights)

    quad_dem_int = (quad_dem > 0 * 255).astype(np.uint8)
    top_components, top_component_mask = cv2.connectedComponents(quad_dem_int[:, :face_width])
    mid_components, mid_component_mask = cv2.connectedComponents(quad_dem_int[:, face_width:-face_width])
    bot_components, bot_component_mask = cv2.connectedComponents(quad_dem_int[:, -face_width:])

    for num_components, component_mask, x_offset in zip(
        (top_components, mid_components, bot_components),
        (top_component_mask, mid_component_mask, bot_component_mask),
        (0, face_width, 5 * face_width),
    ):
        if num_components > 20:
            big_ys, big_xs = np.where(component_mask >= 0)
            big_xs += x_offset
            big_ys = big_ys.reshape(component_mask.shape)
            big_xs = big_xs.reshape(component_mask.shape)
            river_face = accumulation(
                quad_dem[big_ys, big_xs], river_weights[big_ys, big_xs], None, planet_cfg
            )
            rivers[big_ys, big_xs] = river_face
            continue

        for i in range(num_components):
            mask = component_mask == i
            if mask.sum() < (face_width / 100) ** 2:
                continue
            ymin, xmin, ymax, xmax = get_bounds(mask)
            if ymin >= ymax or xmin >= xmax:
                continue
            big_ys = np.arange(ymin, ymax)
            big_xs = np.arange(xmin, xmax)
            big_xs, big_ys = np.meshgrid(big_xs, big_ys)
            big_xs += x_offset

            dem_tile = quad_dem[big_ys, big_xs]
            if dem_tile.max() == 0:
                continue
            weight_tile = river_weights[big_ys, big_xs]
            river_tile = accumulation(dem_tile, weight_tile, None, planet_cfg)
            rivers[big_ys, big_xs] = river_tile

    return rivers / rivers.max() * 255


class NonEquirectangularError(Exception):
    pass


class InferenceController:
    # TODO Consider reimplementing two phase generation
    def __init__(self, full_sized_instance: InferenceInstance):
        self.full_sized_instance = full_sized_instance
        self.inference_args = full_sized_instance.inference_args
        self.planet_cfg = full_sized_instance.planet_cfg
        self.encoder_override = full_sized_instance.encoder_override
        self.inference_instance = self.full_sized_instance

    def _generate_full(
        self,
        sketches: SketchArgs,
        rivers: np.ndarray | None = None,
        previous_mask: np.ndarray | None = None,
        previous_sat: torch.Tensor | None = None,
        previous_dem: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.inference_instance = self.full_sized_instance
        self.inference_instance.progress_pcnt = 0
        (
            quad_downsketch,
            quad_downland_sketch,
            quad_downtemp_sketch,
            quad_previous_mask,
            quad_rivers
        ) = self._prep_sketches(sketches, previous_mask, rivers)

        if previous_sat is not None and previous_dem is not None and self.inference_args.use_previous:
            self.full_sized_instance.set_previous_output(previous_sat, previous_dem)
        extra_kwargs = {}
        if self.inference_args.do_upscaling:
            if previous_sat is None or previous_dem is None:
                raise ValueError("do_upscaling is True but previous_sat or previous_dem is None")
            extra_kwargs["downsat"] = previous_sat.numpy() * 127.5 + 127.5
            extra_kwargs["downdem"] = previous_dem.numpy() * 127.5 + 127.5

        _, outputs = self.full_sized_instance.generate(
            downsketch=quad_downsketch,
            downland_sketch=quad_downland_sketch,
            downtemp_sketch=quad_downtemp_sketch,
            rivers=quad_rivers,
            previous_mask=quad_previous_mask,
            **extra_kwargs
        )
        return self.full_sized_instance.current_sat_output.clone(), self.full_sized_instance.current_dem_output.clone()

    def stop_generation(self):
        self.inference_instance.stop_generation()

    @property
    def stopped(self):
        return self.inference_instance.stopped

    @property
    def progress_pcnt(self):
        return self.inference_instance.progress_pcnt

    @property
    def current_sat_output(self):
        return self.full_sized_instance.current_sat_output

    @property
    def current_dem_output(self):
        return self.full_sized_instance.current_dem_output

    def _prep_sketches(
        self,
        sketches: SketchArgs,
        previous_mask: np.ndarray | None = None,
        rivers: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        masks = sketches.array + [previous_mask]
        max_w = 0
        max_h = 0
        for i, mask in enumerate(masks):
            if mask is None:
                continue
            h, w = mask.shape
            if w == 2 * h:
                masks[i] = QuadSphere(mask, discrete=True).quad_sphere_atlas
                h, w = masks[i].shape
            max_w = max(max_w, w)
            max_h = max(max_h, h)

        for i, mask in enumerate(masks):
            if mask is None:
                continue
            h, w = mask.shape
            if w == max_w:
                continue
            masks[i] = cv2.resize(mask, (max_w, max_h), interpolation=cv2.INTER_NEAREST)

        # 2. Get components
        quad_downsketch, quad_downland_sketch, quad_downtemp_sketch, quad_previous_mask = masks
        quad_rivers = None
        if rivers is not None:
            h, w = rivers.shape
            if w == 2 * h:
                quad_rivers = QuadSphere(rivers, discrete=True).quad_sphere_atlas
            else:
                quad_rivers = rivers
        return quad_downsketch, quad_downland_sketch, quad_downtemp_sketch, quad_previous_mask, quad_rivers

    def _bounding_box_generate(
        self,
        sketches: SketchArgs,
        previous_mask: np.ndarray | None = None,
        previous_sat: torch.Tensor | None = None,
        previous_dem: torch.Tensor | None = None,
        rivers: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        use_previous_mask = (
            previous_mask is not None
            and previous_sat is not None
            and previous_dem is not None
            and self.inference_args.use_previous
        )
        do_upscaling = self.inference_args.do_upscaling

        if sketches.downsketch.shape[1] != 2 * sketches.downsketch.shape[0]:
            raise NonEquirectangularError("Sketches must be equirectangular")

        delta = self.planet_cfg.delta
        # TODO: This will break around the edges of the faces of the quad mask
        generation_mask = (previous_mask == 0).astype(np.uint8) * 255 if use_previous_mask else sketches.downsketch
        if not use_previous_mask:
            previous_mask = np.zeros_like(sketches.downsketch)
        kernel_width = self.inference_args.tile_size // delta
        dilated = cv2.dilate(
            generation_mask,
            np.ones((kernel_width, kernel_width), dtype=np.uint8)
        )
        quad_generation_mask = QuadSphere(dilated, discrete=True).quad_sphere_atlas
        if previous_dem is not None and previous_sat is not None and do_upscaling:
            delta = int(2 ** round(log2(previous_dem.shape[0] / quad_generation_mask.shape[0])))
        faces = np.unique((np.where(quad_generation_mask > 0)[1] / quad_generation_mask.shape[1] * 6).astype(np.uint8))
        # 1. Prepare masks
        (
            quad_downsketch,
            quad_downland_sketch,
            quad_downtemp_sketch,
            quad_previous_mask,
            quad_rivers
        ) = self._prep_sketches(sketches, previous_mask, rivers)

        h, w = sketches.downsketch.shape

        H = h * delta
        W = w * delta
        q_H, q_W = get_quad_shape((H, W))
        if quad_rivers is not None and quad_rivers.shape != (q_H, q_W):
            r_H, r_W = quad_rivers.shape
            if quad_rivers.shape[0] > q_H:
                quad_rivers = cv2.dilate(
                    quad_rivers, np.ones((q_H // r_H, q_W // r_W), dtype=np.uint8)
                )
            quad_rivers = cv2.resize(quad_rivers, (q_W, q_H), interpolation=cv2.INTER_NEAREST)
            river_mask = skeletonize(quad_rivers > 0)
            quad_rivers[~river_mask] = 0
        elif quad_rivers is None:
            quad_rivers = np.zeros((q_H, q_W), dtype=np.float32)
        # For now we don't have a great solution to generating at edges so just generate full
        if not can_generate_faces(set(faces)):
            print(f"Can't generate faces {set(faces)}, generating full")
            return self._generate_full(sketches, quad_rivers, previous_mask, previous_sat, previous_dem)
        if use_previous_mask:
            self.full_sized_instance.set_previous_output(previous_sat.clone(), previous_dem.clone())
        else:
            self.full_sized_instance.reset_output()
        # 2. Get components

        def get_tile(sketch: np.ndarray, big_ys: np.ndarray, big_xs: np.ndarray) -> np.ndarray:
            sketch = cv2.resize(sketch, (q_W, q_H), interpolation=cv2.INTER_NEAREST)
            tile = sketch[big_ys, big_xs]
            return tile

        for big_ys, big_xs, _, _ in get_tile_coords_to_generate(dilated, delta):
            downsketch_tile = get_tile(quad_downsketch, big_ys, big_xs)
            downland_sketch_tile = get_tile(quad_downland_sketch, big_ys, big_xs)
            downtemp_sketch_tile = get_tile(quad_downtemp_sketch, big_ys, big_xs)
            previous_mask_tile = get_tile(quad_previous_mask.astype(np.uint8), big_ys, big_xs) > 0 if use_previous_mask else None
            river_tile = quad_rivers[big_ys, big_xs] if quad_rivers is not None else None
            output_shape = big_ys.shape

            def output_callback(current_output_tile: torch.Tensor) -> None:
                self.full_sized_instance.current_output[
                    :, :, big_ys, big_xs
                ] = current_output_tile
                if self.full_sized_instance.current_prediction is not None:
                    self.full_sized_instance.current_prediction[
                        :, :, big_ys, big_xs
                    ] = current_output_tile

            inference_instance = InferenceInstance(
                self.inference_args,
                self.planet_cfg,
                output_shape,
                self.encoder_override,
                output_callback,
                show_prediction=self.full_sized_instance.show_prediction
            )
            self.inference_instance = inference_instance

            extra_kwargs = {}
            if do_upscaling:
                extra_kwargs["downsat"] = previous_sat[big_ys, big_xs].numpy() * 127.5 + 127.5
                extra_kwargs["downdem"] = previous_dem[big_ys, big_xs].numpy() * 127.5 + 127.5
            if use_previous_mask:
                inference_instance.set_previous_output(
                    previous_sat[big_ys, big_xs],
                    previous_dem[big_ys, big_xs]
                )

            inference_instance.generate(
                downsketch_tile,
                downland_sketch_tile,
                downtemp_sketch_tile,
                rivers=river_tile,
                previous_mask=previous_mask_tile,
                **extra_kwargs
            )

        assert self.full_sized_instance.current_sat_output is not None
        assert self.full_sized_instance.current_dem_output is not None

        return self.full_sized_instance.current_sat_output.clone(), self.full_sized_instance.current_dem_output.clone()

    def generate(
        self,
        sketches: SketchArgs,
        rivers: np.ndarray | None = None,
        previous_sat: torch.Tensor | None = None,  # TODO Change these to np.ndarrays
        previous_dem: torch.Tensor | None = None,
    ):
        try:
            return self._bounding_box_generate(
                sketches, rivers=rivers, previous_sat=previous_sat, previous_dem=previous_dem
            )
        except NonEquirectangularError:
            logging.info("Falling back to full generation")
            return self._generate_full(
                sketches, rivers=rivers, previous_sat=previous_sat, previous_dem=previous_dem
            )

    def partial_regenerate(
        self,
        sketches: SketchArgs,
        previous_mask: np.ndarray,
        previous_sat: torch.Tensor,
        previous_dem: torch.Tensor,
        rivers: np.ndarray | None = None,
    ):
        return self._bounding_box_generate(
            sketches,
            previous_mask,
            previous_sat,
            previous_dem,
            rivers,
        )


class UpscalingInferenceController:
    def __init__(self, normal_instance: InferenceInstance, upscaling_instance: InferenceInstance):
        self.normal_controller = InferenceController(normal_instance)
        self.upscaling_controller = InferenceController(upscaling_instance)
        self.current_controller = self.normal_controller
        self.last_normal_outputs: tuple[torch.Tensor | None, torch.Tensor | None] = (None, None)
        self.normal_controller.inference_args.do_upscaling = False
        self.upscaling_controller.inference_args.do_upscaling = True

    def stop_generation(self):
        self.current_controller.stop_generation()

    @property
    def stopped(self):
        return self.current_controller.stopped

    @property
    def progress_pcnt(self):
        return self.current_controller.progress_pcnt

    @property
    def current_sat_output(self):
        return self.current_controller.full_sized_instance.current_sat_output

    @property
    def current_dem_output(self):
        return self.current_controller.full_sized_instance.current_dem_output

    def generate_normal(
        self,
        sketches: SketchArgs,
        rivers: np.ndarray | None = None,
        previous_mask: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.current_controller = self.normal_controller
        self.last_normal_outputs = self.normal_controller.generate(sketches, rivers)
        return self.last_normal_outputs

    def upscale_last_output(
        self,
        sketches: SketchArgs,
        rivers: np.ndarray | None = None,
        previous_mask: np.ndarray | None = None,
        resize_factor: float = 1.0,
    ) -> tuple[str, list[tuple[str, np.ndarray]]]:
        if (
            len(self.last_normal_outputs) == 0
            or not self.normal_controller.inference_instance.fully_generated
            or self.last_normal_outputs[0] is None
            or self.last_normal_outputs[1] is None
        ):
            logging.info("Last output is empty. Generating low resolution first")
            self.last_normal_outputs = self.generate_normal(sketches, rivers, previous_mask)

        sat = np.array(((self.last_normal_outputs[0] + 1.0) * 127.5).round().clip(0, 255), dtype=np.uint8)
        dem = np.array(((self.last_normal_outputs[1] + 1.0) * 127.5).round().clip(0, 255), dtype=np.uint8)

        return self._upscale(sat, dem, sketches, rivers, previous_mask, resize_factor)

    def generate_and_upscale(
        self,
        sketches: SketchArgs,
        rivers: np.ndarray | None = None,
        previous_mask: np.ndarray | None = None,
    ) -> tuple[str, list[tuple[str, np.ndarray]]]:
        self.generate_normal(sketches, rivers, previous_mask)
        return self.upscale_last_output(sketches, rivers, previous_mask)

    def _get_rivers(
        self,
        dem: torch.Tensor,
        sketches: SketchArgs,
        river_scaling_factor: float = 1.0,
    ):
        quad_dem = tensor_to_np(dem)
        return quad_rivers(sketches, self.current_controller.planet_cfg, quad_dem) * river_scaling_factor

    def _upscale(
        self,
        sat: np.ndarray,
        dem: np.ndarray,
        sketches: SketchArgs,
        rivers: np.ndarray | None = None,
        previous_mask: np.ndarray | None = None,
        resize_factor: float = 1.0,
    ):
        h, w = sat.shape[:2]
        h0 = int(round(h * resize_factor))
        w0 = int(round(w * resize_factor))
        sat = cv2.resize(sat, (w0, h0), interpolation=cv2.INTER_LANCZOS4)
        dem = cv2.resize(dem, (w0, h0), interpolation=cv2.INTER_LANCZOS4)
        H = h0 * self.upscaling_controller.full_sized_instance.planet_cfg.delta
        W = w0 * self.upscaling_controller.full_sized_instance.planet_cfg.delta
        sat = cv2.resize(sat, (W, H), interpolation=cv2.INTER_NEAREST)
        dem = cv2.resize(dem, (W, H), interpolation=cv2.INTER_NEAREST)
        rh, rw = rivers.shape[:2]
        if rw == rh * 6:
            rivers = cv2.resize(rivers, (W, H), interpolation=cv2.INTER_NEAREST)
        elif rw == rh * 2:
            r_H, r_W = get_normal_shape((H, W))
            rivers = cv2.resize(rivers, (r_W, r_H), interpolation=cv2.INTER_NEAREST)
        else:
            rivers = cv2.resize(rivers, (W, H), interpolation=cv2.INTER_NEAREST)
        sat = torch.tensor(sat, dtype=torch.float32) / 127.5 - 1.0
        dem = torch.tensor(dem, dtype=torch.float32) / 127.5 - 1.0
        self.current_controller = self.upscaling_controller
        return self.upscaling_controller.generate(sketches, rivers, sat, dem)
