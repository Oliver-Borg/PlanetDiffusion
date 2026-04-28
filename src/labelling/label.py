
import random
import os

from tqdm import tqdm
import cv2
from pysheds.view import Raster, ViewFinder
from pysheds.grid import Grid
import pyproj
from affine import Affine
import numpy as np

from ..core.constants import LODS, MAX_ELEVATION_LEVEL
from ..core.utils import load_image
from ..core.derivative import elevation_to_gradient, gradient_to_SGF
from ..core.utils import roll_up, roll_down, roll_left, roll_right

# Specify directional mapping
D8_FLOW_DIRECTIONS = (64, 128, 1, 2, 4, 8, 16, 32)
IDENTITY_AFFINE_TRANSFORM = Affine(1, 0, 0, 0, 1, 0)
CRS = pyproj.Proj('epsg:32614', preserve_units=False)


def _extract_network(elevation, threshold, ridge=False):
    vf = ViewFinder(
        affine=IDENTITY_AFFINE_TRANSFORM,
        shape=elevation.shape,
        crs=CRS
    )

    raster = Raster(
        elevation * -1 if ridge else elevation,
        viewfinder=vf
    )

    grid = Grid(vf)
    grid.from_raster(raster)

    # Fill pits in DEM
    pit_filled_dem = grid.fill_pits(raster)

    # Fill depressions in a DEM. Raises depressions to same elevation as lowest neighbor.
    flooded_dem = grid.fill_depressions(pit_filled_dem)

    # Resolve flats in DEM
    inflated_dem = grid.resolve_flats(flooded_dem)

    # Compute flow directions
    fdir = grid.flowdir(inflated_dem)

    # TODO try use gradient map?

    # Calculate flow accumulation
    # i.e., the number of cells upstream of each cell
    acc = grid.accumulation(fdir)

    max_val = elevation.shape[0] * elevation.shape[1]
    min_val = 3  # Ignore 0, 1 and 2

    log_max_val = np.log2(max_val)
    log_min_val = np.log2(min_val)

    mask_threshold = 2 ** ((log_max_val - log_min_val)*threshold + log_min_val)
    order = grid.stream_order(fdir, acc > mask_threshold)

    # # Extract river network
    # branches = grid.extract_river_network(fdir, acc > x)

    return order.astype(np.uint8)


def extract_valley_network(elevation, threshold=0):
    return _extract_network(elevation, threshold, ridge=False)


def extract_ridge_network(elevation, threshold=0):
    return _extract_network(elevation, threshold, ridge=True)


KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
LARGE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))


def enlarge(arr, max_strahler_orders, enlarge_first_strahler_orders):
    max_val = np.max(arr)

    orders = np.arange(max_val, max(max_val - max_strahler_orders, 0), -1)

    # Create mask of values
    to_return = np.zeros_like(arr, dtype=bool)

    for i, order in enumerate(orders):
        # Get all points of certain order
        vals = (arr == order).astype(np.uint8)

        # Enlarge
        if i < enlarge_first_strahler_orders:
            vals = cv2.dilate(vals, KERNEL, iterations=1)
        to_return[vals > 0] = True

    return to_return


def lerp(a, b, t):
    return (1-t) * a + t * b


def quantise(image, k, iterations=200):
    # https://www.analyticsvidhya.com/blog/2021/07/colour-quantization-using-k-means-clustering-and-opencv/

    flat = image.flatten()
    condition = (cv2.TERM_CRITERIA_EPS +
                 cv2.TERM_CRITERIA_MAX_ITER, iterations, 1.0)
    _, label, center = cv2.kmeans(
        flat, k, None, condition, 10, cv2.KMEANS_RANDOM_CENTERS)

    final_img = center[label.flatten()]
    labels = np.sort(np.unique(final_img))
    final_img = final_img.reshape(image.shape)
    return final_img, labels


def filter_connected_components(arr, min_area, value=0):
    # Find connected components
    num_components, connected_components = cv2.connectedComponents(arr)

    # Iterate over the connected components and remove those that are too small
    for component in range(num_components):
        comp_mask = connected_components == component
        count = np.sum(comp_mask)
        if count < min_area:
            arr[comp_mask] = value
    return arr


# NOTE: Following entries optimised for 256x256 tiles.
# TODO adjust based on size
#
# Remove entities that are too small
MIN_STEEP_AREA = 128
MIN_FLATNESS_AREA = 2048

# Used to detect flat regions
MIN_FLATNESS_THRESHOLD = 0.015

# Used to detect cliffs/mesas
LARGE_CLIFF_DIFFERENCE = 2.5  # metres
MIN_CLIFF_DIFFERENCE = 2  # metres
NUM_QUANTS = 4


def generate_contours(heightmap, num_contours):
    # Represents whether each element should be marked as on the level curve
    lines = np.zeros_like(heightmap, dtype=np.uint16)

    min_val = np.min(heightmap)
    max_val = np.max(heightmap)
    height_range = max_val - min_val
    if height_range == 0:
        return lines  # Return empty image

    # Normalise between 0 and 1
    heightmap = (heightmap - min_val) / height_range

    levels = np.linspace(0, 1, num_contours+2)[1:-1]

    # min_normalised_val = levels[0]  # e.g., 0.2
    # max_normalised_val = levels[-1]  # e.g., 0.8
    # normalised_range = max_normalised_val - min_normalised_val

    shifted_left = roll_left(heightmap)
    shifted_down = roll_down(heightmap)
    shifted_up = roll_up(heightmap)
    shifted_right = roll_right(heightmap)

    for i, level in enumerate(levels):
        d = np.abs(heightmap - level)

        h1 = heightmap >= level
        h2 = ~h1

        diff_l = shifted_left - level
        diff_d = shifted_down - level
        diff_u = shifted_up - level
        diff_r = shifted_right - level

        # val_to_set = min_contour_mapping + \
        #     (1 - min_contour_mapping) * \
        #     (level - min_normalised_val) / normalised_range

        for dim in (diff_l, diff_d, diff_u, diff_r):
            q = dim < 0
            lines[
                (h1 & q | h2 & ~q)    # Check if level is between heightmap and shift
                & (d <= np.abs(dim))  # Ensure closest neighbour is chosen
            ] = i  # val_to_set

    # lines = filter_connected_components(lines, min_area)
    return lines


def generate_conditioning(elev,
                          resolution,  # meters per pixel
                          # seems to work well
                          base_threshold_range=(0.4, 0.8),
                          min_component_size=0.1,
                          max_strahler_orders=4,  # Get x largest strahler orders
                          # Increase brush size for x largest strahler orders
                          enlarge_first_strahler_orders=2,
                          to_image=True
                          ):
    """
    Returns (sketch, embedding)

    - Sketch: Creates n-channel image containing:
        1. Ridges
        2. Valleys
        3. Steep cliff/mesa lines
        4. Flat sections
        5. (optional) height constraints

    """

    # Calculate useful items
    gradient = elevation_to_gradient(elev, xres=resolution, yres=resolution)
    sgf = gradient_to_SGF(*gradient)

    # Generate two different versions of blurred elevation
    slight_blurred_elev = cv2.GaussianBlur(elev, (7, 7), 1)
    # blurred_elev = cv2.GaussianBlur(elev, (7, 7), 5)

    # Determine minimum component size (for ridges and valleys)
    #   1024 x 1024 --> 256
    #   512 x 512 --> 128
    #   256 x 256 --> 64
    min_component_size = round(np.sqrt(elev.size) / 4)

    # Calculate magnitude of slope at each point
    slope_magnitude = np.sqrt(np.sum(np.dstack(gradient) ** 2, axis=2))
    slope_magnitude = cv2.medianBlur(slope_magnitude, ksize=3)

    steep_edges = np.zeros(slope_magnitude.shape, dtype=np.uint8)

    bilat_elev = cv2.bilateralFilter(elev, 9, 50, 50)
    bilat_elev = cv2.normalize(
        bilat_elev, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    edges = cv2.Canny(bilat_elev, 60, 90)

    # Dilate the image to expand the size of the objects
    dilated = cv2.dilate(edges, KERNEL, iterations=3) > 0

    bilat_slope_magnitude = cv2.bilateralFilter(slope_magnitude, 9, 50, 50)

    # Calculate flat areas based on smoothed slope magnitude
    flat_areas = (bilat_slope_magnitude <
                  MIN_FLATNESS_THRESHOLD).astype(np.uint8)
    flat_areas = filter_connected_components(flat_areas, MIN_FLATNESS_AREA)

    # Fill in gaps
    flat_areas = cv2.dilate(flat_areas, KERNEL, iterations=1)
    flat_areas = cv2.morphologyEx(flat_areas, cv2.MORPH_CLOSE, LARGE_KERNEL)
    flat_areas = cv2.medianBlur(flat_areas, ksize=9)

    # Set non-steep parts to 0
    bilat_slope_magnitude[slope_magnitude < 0.5] = 0
    bilat_slope_magnitude = cv2.dilate(
        bilat_slope_magnitude, KERNEL, iterations=1) > 0

    # Calculate steep cliff/mesa lines
    quant, labels = quantise(elev, NUM_QUANTS)

    s_w, s_h = slope_magnitude.shape
    dilated = dilated[:s_w, :s_h]

    for ind in labels:
        mask = (quant <= ind).astype(np.uint8)
        mask = cv2.medianBlur(mask, ksize=7)
        mask = cv2.Canny(mask, 0, 1) > 0
        mask = mask[:s_w, :s_h]

        mask_sum = np.sum(mask)
        if mask_sum == 0:
            continue

        overlap_percentage = np.sum(
            mask & dilated & bilat_slope_magnitude) / mask_sum

        if overlap_percentage > 0.5:
            mask = cv2.dilate(mask.astype(np.uint8), KERNEL, iterations=1)

            if overlap_percentage <= 0.8:
                mask = cv2.erode(mask, KERNEL, iterations=1)

            steep_edges[mask > 0] = 1

    steep_edges = filter_connected_components(steep_edges, 256)
    steep_edges = steep_edges > 0

    # First calculate flat regions by computing scaled gradient field of blurred elevation map
    flat_percentage = np.sqrt(np.sum(flat_areas) / flat_areas.size)

    # Adjust minimum threshold based on how flat the terrain is
    min_threshold = lerp(*base_threshold_range, flat_percentage)
    max_threshold = base_threshold_range[1]

    # Threshold controls how detailed the image is
    threshold = random.uniform(min_threshold, max_threshold)

    # Extract networks
    ridges = extract_ridge_network(slight_blurred_elev, threshold)
    valleys = extract_valley_network(slight_blurred_elev, threshold)

    # Filter connected components (removes small artefacts)
    ridges = filter_connected_components(ridges, min_component_size)
    valleys = filter_connected_components(valleys, min_component_size)

    # Enlarge lines based on counts
    # https://stackoverflow.com/questions/46895772/thicken-a-one-pixel-line
    ridges = enlarge(ridges, max_strahler_orders,
                     enlarge_first_strahler_orders)
    valleys = enlarge(valleys, max_strahler_orders,
                      enlarge_first_strahler_orders)

    # Ensure elements are stackable
    ridges = ridges[:s_w, :s_h]
    valleys = valleys[:s_w, :s_h]
    steep_edges = steep_edges[:s_w, :s_h]

    # Stack
    final_sketch = np.dstack(
        (ridges, valleys, steep_edges, flat_areas)).astype(bool)

    if to_image:
        # Minor post-processing for visualisation
        #  - Convert to 8-bit image for display
        #  - Flip alpha channel (easier to see non-flat regions)
        final_sketch = (255 * final_sketch).clip(0,
                                                 255).astype(np.uint8)  # Convert to RGBA
        final_sketch[:, :, -1] = 255 - final_sketch[:, :, -1]


    return final_sketch


def main():
    import matplotlib.pyplot as plt
    import json
    from ..core.terrain_dataset import METADATA_FILE

    example_dir = './data/processed/to_divide/'
    example_files = os.listdir(example_dir)[:10]

    elev_file = 'elevation-256x256.png'

    fig, plots = plt.subplots(2, len(example_files))

    for i, tile in enumerate(tqdm(example_files)):
        metadata_path = os.path.join(example_dir, tile, METADATA_FILE)
        if not os.path.exists(metadata_path):
            continue

        with open(metadata_path) as fp:
            metadata = json.load(fp)
        elev_range = metadata['range']
        factor = metadata['factor']

        resolution, scale = LODS[MAX_ELEVATION_LEVEL - factor]

        elev_path = os.path.join(example_dir, tile, elev_file)
        elev = load_image(elev_path) * elev_range
        sketch = generate_conditioning(elev, resolution=resolution)

        plots[0][i].imshow(elev)
        plots[1][i].imshow(sketch)

    plt.show()


if __name__ == '__main__':
    main()
