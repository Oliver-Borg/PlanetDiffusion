import numpy as np
import cv2
from .noise_funcs import noise2coords


def noise2(
    xs: np.ndarray,
    ys: np.ndarray,
    seed: int | None = None,
    frequency: float = 0.015,
    amplitude: float = 0.2,
    octaves: int = 5,
) -> np.ndarray:

    # 1. Get noise
    freq_mult = 0.5
    persistence = 0.7
    stacked_coords = np.dstack([xs, ys])
    total_noise = np.zeros_like(stacked_coords[:, :, 0])
    for _ in range(octaves):
        noise = (noise2coords(stacked_coords, seed) + 1) / 2
        total_noise += noise * amplitude
        frequency *= freq_mult
        amplitude *= persistence

    # 2. Normalize noise
    # This is done to prevent rare cases where it could far exceed the expected maximum
    total_noise = total_noise / total_noise.max()
    total_noise *= amplitude

    return total_noise


def rough_edge_tile_mask(
    shape: tuple[int, int],
    seed: int | None = None,
    frequency: float = 0.015,
    amplitude: int = 20,
) -> np.ndarray:
    h, w = shape
    return jigsaw_piece_mask(
        (h // 2, w // 2), shape, w, 0.75, (0, 0), seed, frequency, amplitude
    )


def circle_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (h, w)) > 0


def min_max_norm(image: np.ndarray) -> np.ndarray:
    return (image - image.min()) / (image.max() - image.min())


def directed_noise(
    xs: np.ndarray,
    ys: np.ndarray,
    seed: int | None = None,
    frequency: float = 0.015,
    amplitude: float = 0.2,
    octaves: int = 5,
) -> np.ndarray:
    noise = noise2(xs, ys, seed, frequency, amplitude, octaves)
    noise = min_max_norm(noise)
    noise = noise * 2 - 1
    return noise


def jigsaw_piece_mask(
    center_coord: tuple[int, int],
    shape: tuple[int, int],
    tile_width: int,
    grid_spacing: float = 0.5,
    grid_offset: tuple[int, int] = (0, 0),
    seed: int | None = None,
    frequency: float = 0.1,
    amplitude: int = 10,
) -> np.ndarray:
    # TODO Test and fix offset stuff
    y, x = center_coord
    oy, ox = grid_offset
    t_w = tile_width
    h, w = shape
    h_shift = int(t_w * grid_spacing)
    w_shift = int(t_w * grid_spacing)
    full_mask = np.zeros(shape, dtype=bool)
    t_y = (y - t_w // 2 + (t_w - h_shift) // 2 + oy) % h
    b_y = (t_y + h_shift) % h
    l_x = (x - t_w // 2 + (t_w - w_shift) // 2 + ox) % w
    r_x = (l_x + w_shift) % w
    full_mask = np.ones((t_w, t_w), dtype=bool)

    ys = (np.arange(t_w) + y - t_w // 2) % h
    xs = (np.arange(t_w) + x - t_w // 2) % w
    xs, ys = np.meshgrid(xs, ys)
    stacked_coords = np.dstack([xs, ys])
    noise_tile = (noise2coords(stacked_coords * frequency, seed) * amplitude).astype(
        int
    )
    top_y_valid_mask = ys == t_y
    bottom_y_valid_mask = ys == b_y
    left_x_valid_mask = xs == l_x
    right_x_valid_mask = xs == r_x

    tile_ys = np.arange(t_w)
    tile_xs = np.arange(t_w)
    tile_xs, tile_ys = np.meshgrid(tile_xs, tile_ys)

    top_y_coords = noise_tile[top_y_valid_mask] + tile_ys[top_y_valid_mask]
    bottom_y_coords = noise_tile[bottom_y_valid_mask] + tile_ys[bottom_y_valid_mask]
    left_x_coords = noise_tile[left_x_valid_mask] + tile_xs[left_x_valid_mask]
    right_x_coords = noise_tile[right_x_valid_mask] + tile_xs[right_x_valid_mask]

    # TODO optimize this
    for i in range(t_w):
        full_mask[0: top_y_coords[i], i] = False
        full_mask[bottom_y_coords[i]:, i] = False
    for i in range(t_w):
        full_mask[i, 0: left_x_coords[i]] = False
        full_mask[i, right_x_coords[i]:] = False

    return full_mask


def jigsaw_grid_mask(
    shape: tuple[int, int],
    tile_width: int = 256,
    grid_spacing: float = 0.5,
    grid_offset: tuple[int, int] = (0, 0),
    seed: int | None = None,
    frequency: float = 0.1,
    amplitude: int = 10,
) -> np.ndarray:
    grid = np.zeros((*shape, 3), dtype=np.uint8)
    oy, ox = grid_offset
    t_w = tile_width
    h, w = shape

    # TODO Extpand the size of the last tile to prevent double overlap
    h_shift = int(t_w * grid_spacing)
    w_shift = int(t_w * grid_spacing)
    tile_num = 0
    for x in range(t_w // 2 + ox, h + t_w // 2 + ox, h_shift):
        for y in range(t_w // 2 + oy, w + t_w // 2 + oy, w_shift):
            tile_num += 1
            xs = np.arange(t_w) + x - t_w // 2
            ys = np.arange(t_w) + y - t_w // 2
            xs, ys = np.meshgrid(xs, ys)
            xs = xs % w
            ys = ys % h
            # 1. Draw outline of shift_size tile centered at (i, j)
            shift_grid_mask = np.zeros((t_w, t_w, 3), dtype=np.uint8)
            cv2.rectangle(
                shift_grid_mask,
                ((t_w - w_shift) // 2 + amplitude, (t_w - h_shift) // 2 + amplitude),
                ((t_w + w_shift) // 2 - amplitude, (t_w + h_shift) // 2 - amplitude),
                (255, 255, 255),
                1,
            )
            grid[ys, xs] |= shift_grid_mask

            # 2. Draw outline of tile_shape tile centered at (i, j)
            # Only draw this if we are at the center
            if (
                x > w // 2 - h_shift // 2
                and y > h // 2 - w_shift // 2
                and x < w // 2 + h_shift // 2
                and y < h // 2 + w_shift // 2
            ):
                tile_mask = np.zeros((t_w, t_w, 3), dtype=np.uint8)
                red_tile = (y // h_shift + x // w_shift) % 2
                colour = (255, 255, 255) if red_tile else (255, 255, 255)
                cv2.rectangle(
                    tile_mask,
                    (red_tile, red_tile),
                    (t_w - red_tile - 1, t_w - red_tile - 1),
                    colour,
                    1,
                )
                grid[ys, xs] |= tile_mask

            piece_mask = jigsaw_piece_mask(
                (y, x),
                (h, w),
                t_w,
                grid_spacing,
                grid_offset,
                seed,
                frequency,
                amplitude,
            )
            tile_mask = np.zeros((t_w, t_w, 3), dtype=np.uint8)
            np.random.seed(tile_num)
            colour = np.random.randint(1, 256, 3)
            for i in range(3):
                # Assign random colour to each piece
                tile_mask[:, :, i] = piece_mask * colour[i]
            grid[ys, xs] |= tile_mask
    assert np.all(grid > 0)
    return grid
