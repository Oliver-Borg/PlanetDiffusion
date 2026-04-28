from PIL import Image
import numpy as np
import cv2
from enum import Enum
import os

from skimage.morphology import skeletonize

from .utils import PlanetConfig, get_data_image
from .sketch_gen import dilate_paint
from .sphere_mapping import QuadSphere


class BoundaryColours(Enum):
    DIVERENT = (255, 0, 0)
    CONVERGENT = (32, 255, 32)
    TRANSFORM = (190, 67, 255)
    # OTHER = (0, 79, 167)


class ModalColours(Enum):
    DIVERENT = 113
    CONVERGENT = 183
    TRANSFORM = 171
    OTHER = 0


class ClassColours(Enum):
    DIVERENT = 60
    CONVERGENT = 180
    TRANSFORM = 120
    OTHER = 0


def filter_arr(im: np.ndarray, col: tuple[int, int, int], delta: int = 100) -> np.ndarray:
    im = im.astype(np.uint16)  # Prevent overflow
    c0 = np.abs(im[:, :, 0] - col[0]) < delta
    c1 = np.abs(im[:, :, 1] - col[1]) < delta
    c2 = np.abs(im[:, :, 2] - col[2]) < delta
    return c0 & c1 & c2


def modal_colours(arr: np.ndarray, n: int = 5) -> np.ndarray:
    vals, counts = np.unique(arr, return_counts=True)
    return vals[np.argsort(counts)][-n:]


def modal_colour(arr: np.ndarray) -> tuple[int, int, int]:
    return modal_colours(arr, 1)[0]


def get_boundary_sketch(
    bathy: np.ndarray, dem: np.ndarray, boundaries: np.ndarray, use_modal: bool = False
) -> np.ndarray:
    boundary_sketch = np.zeros_like(bathy)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    for boundary, chosen_colour, modal_colour in zip(BoundaryColours, ClassColours, ModalColours):
        mask = filter_arr(boundaries, boundary.value) & (dem == 0)
        # This is to get the modal colours
        # erode_mask = cv2.erode(mask.astype(np.uint8) * 255, kernel, iterations=1) > 0
        # colours = bathy[erode_mask]
        # colour = modal_colour(colours)
        colour = chosen_colour if not use_modal else modal_colour

        print("Mapping", boundary, "to", colour.value)
        boundary_sketch[mask] = colour.value

    boundary_sketch = cv2.morphologyEx(boundary_sketch, cv2.MORPH_CLOSE, kernel)
    return boundary_sketch


def get_boundary_line_sketch(boundary_sketch: np.ndarray, dilate_size: int = 5, dilate_iters: int = 1) -> np.ndarray:
    boundary_sketch = boundary_sketch.copy()
    dilated = cv2.dilate(
        boundary_sketch,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=2
    )
    line_mask = skeletonize(dilated)
    line_mask = cv2.dilate(
        line_mask.astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size)),
        iterations=dilate_iters
    )
    boundary_sketch[line_mask == 0] = 0
    return boundary_sketch


if __name__ == "__main__":
    planet_cfg = PlanetConfig(size=2)

    H, W = planet_cfg.H, planet_cfg.W
    bathy = get_data_image(
        planet_cfg.data_dir,
        (H, W),
        "gebco_bathy.WxH.jpg",
        default_shape=(10801, 21601),
        interpolation=cv2.INTER_AREA,
    )
    quadsphere = QuadSphere(atlas=bathy)

    boundaries = get_data_image(
        planet_cfg.data_dir,
        (H, W),
        "bmplates_WxH.png",
        default_shape=(512, 1024),
        interpolation=cv2.INTER_NEAREST,
    )

    dem = get_data_image(
        planet_cfg.data_dir,
        (H, W),
        "World_DEM_WxH.png",
        default_shape=(8192, 16384),
        interpolation=cv2.INTER_LANCZOS4,
    )

    boundary_sketch = get_boundary_sketch(bathy, dem, boundaries, use_modal=False)
    boundary_line_sketch = get_boundary_line_sketch(boundary_sketch, dilate_iters=0)
    bathy_sketch = dilate_paint(bathy.copy(), planet_cfg)
    bathy_sketch[boundary_line_sketch > 0] = boundary_sketch[boundary_line_sketch > 0]

    Image.fromarray(boundary_sketch).save(os.path.join(planet_cfg.test_dir, f"boundary_sketch_{W}x{H}.png"))
    Image.fromarray(boundary_line_sketch).save(os.path.join(planet_cfg.test_dir, f"boundary_line_sketch_{W}x{H}.png"))

    quad_boundary_sketch = QuadSphere(atlas=boundary_sketch, discrete=True).get_quad_sphere_atlas()
    quad_boundary_line_sketch = get_boundary_line_sketch(quad_boundary_sketch, dilate_iters=0)

    quad_h, quad_w = quad_boundary_sketch.shape

    Image.fromarray(quad_boundary_sketch).save(
        os.path.join(planet_cfg.test_dir, f"quad_boundary_sketch_{quad_w}x{quad_h}.png")
    )
    Image.fromarray(quad_boundary_line_sketch).save(
        os.path.join(planet_cfg.data_dir, f"quad_boundary_line_sketch_{quad_w}x{quad_h}.png")
    )
