import numpy as np
from PIL import Image
import cv2

from .utils import timing

RIVER_LEN_BOUNDS = (2000, 6600)
RIVER_WIDTH_BOUNDS = (2000, 6400)
RIVER_ORDER_BOUNDS = (0, 30)


def normalise_river_data(val, bounds):
    return (val - bounds[0]) / (bounds[1] - bounds[0]) * 255


class RiverFilter:
    def __init__(self, min_river_length, min_river_width):
        self.min_river_length = min_river_length
        self.min_river_width = min_river_width
    
    def get_normalised_values(self):
        return normalise_river_data(self.min_river_length, RIVER_LEN_BOUNDS), normalise_river_data(self.min_river_width, RIVER_WIDTH_BOUNDS)

    def set_river_length(self, val: int):
        self.min_river_length = val

    def set_river_width(self, val: int):
        self.min_river_width = val


RIVER_FILTERS = [
    RiverFilter(6532, 2000),
    RiverFilter(3421, 2194),
    RiverFilter(2271, 3100)
]


def apply_filters(
    river_length: np.ndarray,
    river_width: np.ndarray,
    river_order: np.ndarray,
    filters: list[RiverFilter] = RIVER_FILTERS,
    min_river_order: int = 0,
    max_river_order: int = 10,
    min_river_dilate: int = 3,
    max_river_dilate: int = 18,
) -> np.ndarray:
    river_mask = np.zeros(river_length.shape, dtype=np.uint8)
    for filter in filters:
        min_rl_val, min_rw_val = filter.get_normalised_values()
        river_mask |= (river_length > min_rl_val) & (river_width > min_rw_val)
    river_mask &= (river_order >= min_river_order) & (river_order <= max_river_order)
    dilate_iters = np.ceil(river_width / 255 * (max_river_dilate-min_river_dilate)).astype(np.uint8)
    dilate_iters[dilate_iters > 0] += min_river_dilate
    dilate_iters[river_mask == 0] = 0
    for i in range(1, max_river_dilate):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (i, i))
        width_mask = (dilate_iters == i).astype(np.uint8) * 255
        width_mask = cv2.dilate(width_mask, kernel, iterations=1)
        river_mask |= width_mask > i

    return river_mask


def filter_components(river_mask: np.ndarray, min_component_size: int) -> np.ndarray:
    if min_component_size > 0:
        num_components, labeled = cv2.connectedComponents(river_mask, connectivity=8)
        values, counts = np.unique(labeled, return_counts=True)
        for value, count in zip(values, counts):
            if count < min_component_size:
                river_mask[labeled == value] = 0

    return river_mask


def get_stacked_rivers(data_dir: str, height: int, width: int) -> np.ndarray:
    shape = (width, height)
    river_length = np.array(Image.open(f'{data_dir}/River_Length_16384.tif').resize(shape))
    river_width = np.array(Image.open(f'{data_dir}/River_Width_16384.tif').resize(shape))
    river_orders = np.array(Image.open(f'{data_dir}/global_river_order_16384.tif').resize(shape))
    return np.dstack([river_length, river_width, river_orders])
