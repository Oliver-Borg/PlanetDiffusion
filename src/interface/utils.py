import datetime
import os
import itertools

import numpy as np
import cv2
from skimage.morphology import skeletonize
from PIL import Image

from planetAI.src.data.mean_precip import PrecipSketch
from planetAI.src.data.uncertainty_sketch import UncertaintySketcher
from planetAI.src.data.utils import get_brush_kernel, PlanetConfig, get_bounds, profile, timing
from planetAI.src.data.sphere_mapping import SphereMapping, QuadSphere, normal_to_quad
from planetAI.src.data.data_creator import cv2_rotate
from planetAI.src.data.sketch_gen import accumulation, temperature_paint
from planetAI.src.data.noise_settings import NoiseSettings
from planetAI.src.data.noise_funcs import stacked_noise, stacked_multi_noise
from planetAI.src.data.landcover_utils import LandcoverClasses, gray_to_land


from .interface_types import TempPresetEventMetadata, LandCoverEventMetadata, DEMEventMetadata


def get_temp_folder() -> str:
    '''
    Get the path to the temp folder
    '''
    return os.path.join(os.path.dirname(__file__), '.temp')


def get_line(p1: tuple[int, int], p2: tuple[int, int], wrap_around_width: int | None = None) -> list[tuple[int, int]]:
    """Bresenham's Line Algorithm"""
    if None in p1:
        return [p2, p2]
    if None in p2:
        return [p1, p1]
    x1, y1 = p1
    x2, y2 = p2
    points = [(x1, y1)]
    dx = x2 - x1
    dy = y2 - y1
    waw = wrap_around_width
    if waw is not None and abs(dx) > waw//2:
        if dx > 0:
            x1 += waw
        else:
            x2 += waw
    dx = x2 - x1

    is_steep = abs(dy) > abs(dx)
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
    dx = x2 - x1
    dy = y2 - y1
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1
    y = y1
    for x in range(x1, x2 + 1):
        coord = (y, x % waw if waw else x) if is_steep else (x, y % waw if waw else y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx
    if swapped:
        points.reverse()
    return points


@profile
def get_paint_points(
    points: list[tuple[int, int]],
    h: int,
    w: int,
    brush_size: int = 1,
    wrap_around_width: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    '''
    Get a paint mask from a list of points
    Args:
        points (list[tuple[int, int]]): The points to use
        shape (tuple[int, int]): The shape of the mask
        brush_size (int): The size of the brush to use
    Returns:
        tuple[np.ndarray, np.ndarray]: The y and x coordinates of the paint mask
    '''
    if len(points) == 0:
        return (np.array([]), np.array([]))

    canvas = np.zeros((h, w), dtype=np.uint8)

    if len(points) == 1:
        x, y = points[0]
        canvas[y, x] = 255
    else:
        for p1, p2 in zip(points[:-1], points[1:]):
            for x, y in get_line(p1, p2, wrap_around_width):
                canvas[y, x] = 255

    kernel = get_brush_kernel(brush_size)

    base_ys, base_xs = np.where(kernel > 0)
    base_ys -= brush_size
    base_xs -= brush_size
    if len(points) == 1:
        x, y = points[0]
        canvas[base_ys + y, base_xs + x] = 255
    for i in range(1, len(points)):
        p1 = points[i-1]
        p2 = points[i]
        for x, y in get_line(p1, p2, wrap_around_width):
            canvas[base_ys + y, base_xs + x] = 255
    return np.where(canvas > 0)


def get_lasso_points(
    points: list[tuple[int, int]],
    shape: tuple[int, int],
    wrap_around_width: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    '''
    Get a lasso filled points from a list of points
    Args:
        points (list[tuple[int, int]]): The points to use
        shape (tuple[int, int]): The shape of the mask
    Returns:
        tuple[np.ndarray, np.ndarray]: The y and x coordinates of the lasso mask
    '''
    first_point = points[0]
    last_point = points[-1]
    line = get_line(first_point, last_point, wrap_around_width)
    points.extend(line)
    xs, ys = np.array(points).T

    # 1. Create a bounding box around the lasso points
    min_x = np.min(xs)
    max_x = np.max(xs)
    min_y = np.min(ys)
    max_y = np.max(ys)

    # 2. Create a mask for the bounding box
    mask = np.zeros((max_y-min_y+3, max_x-min_x+3), dtype=np.uint8)
    mask[ys-min_y+1, xs-min_x+1] = 1

    # 3. Flood fill the outside of the mask
    cv2.floodFill(mask, None, (0, 0), 2)
    mask[mask != 2] = 1
    mask[mask == 2] = 0

    # 5. Create full mask
    ys, xs = np.where(mask[1:-1, 1:-1] == 1)
    ys += min_y
    xs += min_x
    return ys, xs


def derive_landcover(dem: np.ndarray, temperature: np.ndarray, land_noise: np.ndarray | None = None) -> np.ndarray:
    """
    Derive landcover based on elevation, temperature and latitude
    """

    uncertainty_sketcher = UncertaintySketcher(PlanetConfig())

    output = np.zeros_like(temperature)

    counts = uncertainty_sketcher.counts

    dem_vals = np.unique(dem)
    temp_vals = np.unique(temperature)

    for dem_sketch_val in dem_vals:
        if dem_sketch_val == 0:
            continue
        for temp_sketch_val in temp_vals:
            land_vals = []
            land_counts = []
            for key, count in counts.items():
                land_value, temp_value, dem_value = key
                if land_value == LandcoverClasses.MARS.gray_colour:
                    continue
                if land_value == LandcoverClasses.SNOW_AND_ICE.gray_colour:
                    continue
                if temp_value == temp_sketch_val and dem_value == dem_sketch_val:
                    land_vals.append(land_value)
                    land_counts.append(count)
                    continue
            sorter = np.argsort(land_counts)
            land_counts = np.array(land_counts)[sorter]
            land_vals = np.array(land_vals)[sorter]
            if len(land_vals) > 0 and land_noise is None:
                target_val = land_vals[np.argmax(land_counts)]
                output[(dem == dem_sketch_val) & (temperature == temp_sketch_val)] = target_val
            elif land_noise is None:
                target_val = LandcoverClasses.TREE_COVER.gray_colour
                output[(dem == dem_sketch_val) & (temperature == temp_sketch_val)] = target_val
            else:
                cumulative_counts = np.cumsum(land_counts)
                cumulative_counts = cumulative_counts / cumulative_counts.max()
                for target_val, cumulative_count in zip(land_vals, cumulative_counts):
                    output[
                        (dem == dem_sketch_val) &
                        (temperature == temp_sketch_val) &
                        (land_noise <= cumulative_count) &
                        (output == 0)
                    ] = target_val

    ys, xs = np.where(dem < 10000000)
    h, w = dem.shape
    lats = 90 - (ys / h * 180)
    longs = (xs / w * 360) - 180
    lats = lats.reshape(dem.shape)
    longs = longs.reshape(dem.shape)

    output[
        (dem > 0) &
        (temperature * (land_noise + 1.0 if land_noise is not None else 1.0) / 255 * 82 - 45 < 0) &
        ((lats > 70) | (lats < -70))
    ] = LandcoverClasses.SNOW_AND_ICE.gray_colour

    return output

    ys, xs = np.where(dem < 10000000)
    h, w = dem.shape
    lats = 90 - (ys / h * 180)
    longs = (xs / w * 360) - 180
    lats = lats.reshape(dem.shape)
    longs = longs.reshape(dem.shape)

    # Convert temperature to °C (-45 to 37 degrees)
    temperature = temperature / 255 * 82 - 45

    # Convert elevation to meters (0 to 8271m)
    dem = dem / 255 * 8271

    # Initialize with water
    landcover = np.zeros_like(dem, dtype=np.uint8)
    landcover[:, :] = LandcoverClasses.OPEN_WATER.gray_colour

    # Water bodies
    landcover[(dem == 0)] = LandcoverClasses.OPEN_WATER.gray_colour

    # Snow and ice (polar regions and high mountains)
    landcover[
        (dem > 0) &
        (temperature < 0) &
        ((lats > 70) | (lats < -70))
    ] = LandcoverClasses.SNOW_AND_ICE.gray_colour

    # Shrubland (semi-arid regions)
    landcover[
        (dem > 0) & (landcover == 0) &
        (temperature > 20) & (temperature < 37) &
        ((lats > -60) & (lats < 35))              # Mid latitudes
    ] = LandcoverClasses.SHRUBLAND.gray_colour

    # Bare ground (very hot deserts and very high mountains)
    landcover[
        (dem > 0) &
        ((temperature > 25) |                     # Hot deserts
         (dem > 4000))                           # Very high mountains
    ] = LandcoverClasses.BARE.gray_colour
    # Rainforests (equatorial regions with adequate temperature)
    landcover[
        (dem > 0) & (dem < 2000) &              # Lower elevations
        (temperature > 20) &                     # Warm temperatures
        (lats > -23.5) & (lats < 23.5)          # Between tropics
    ] = LandcoverClasses.TREE_COVER.gray_colour

    # Grasslands (temperate regions and higher elevations)
    landcover[
        (dem > 0) & (landcover == 0) &
        ((dem > 2000) & (dem < 3500) |           # Mountain grasslands
         (temperature > 5) & (temperature < 25))  # Temperate grasslands
    ] = LandcoverClasses.GRASSLAND.gray_colour

    # Temperate forests (fill remaining valid land in suitable climate)
    landcover[
        (dem > 0) & (landcover == 0)
    ] = LandcoverClasses.TREE_COVER.gray_colour

    return landcover


@profile
def create_temp_preset(
    metadata: TempPresetEventMetadata, shape: tuple[int, int], as_sketch: bool = False
) -> np.ndarray:
    '''
    Create a temperature preset sketch using the latitude temperature list
    '''
    temp_preset = np.zeros(shape, dtype=np.uint8)
    lat_temp_list = metadata.lat_temp_list
    min_temp = metadata.min_temp
    max_temp = metadata.max_temp
    noise = metadata.noise
    h, w = shape
    for i in range(len(lat_temp_list)-1):
        (lat1, temp1), (lat2, temp2) = lat_temp_list[i], lat_temp_list[i+1]
        y1 = int(round((lat1 + 90)/180 * h))
        y2 = int(round((lat2 + 90)/180 * h))
        dist = y2 - y1
        temps = np.linspace(temp1, temp2, dist)
        temps = (np.clip(temps, min_temp, max_temp) - min_temp) / (max_temp - min_temp) * 255
        temps = temps.astype(np.uint8)
        # Reshape from 1D to 2D
        temps = np.repeat(temps[:, np.newaxis], w, axis=1)
        temp_preset[y1:y2] = temps

    # Pad the temp_preset before rotating
    temp_preset = np.pad(temp_preset, ((h//2, h//2), (w//2, w//2)), mode='edge')
    angle = metadata.pivot
    center = (w, h)
    temp_preset = cv2_rotate(temp_preset, angle, center)
    temp_preset = temp_preset[h//2:h//2+h, w//2:w//2+w]

    if w < h*2:
        noise = noise[:, h-w//2:h+w-w//2]
    if noise.shape == temp_preset.shape:
        temp_preset = (
            temp_preset.astype(np.float32) * (1 - metadata.noise_weight) + noise * metadata.noise_weight
        )
    temp_preset = np.clip(temp_preset, 0, 255).astype(np.uint8)
    if as_sketch:
        temp_preset[temp_preset == 0] = 1
        temp_preset = temperature_paint(temp_preset, PlanetConfig())

    return temp_preset


@timing
def planet_noise(
    shape: tuple[int, int],
    noise_settings: NoiseSettings | list[NoiseSettings],
    coord_offset: tuple[float, float] | None = None
) -> np.ndarray:
    '''
    Generate planet noise
    '''
    sphere_mapping = SphereMapping(shape=shape)
    noise = np.ones(shape)
    if not isinstance(noise_settings, list):
        noise_settings = [noise_settings]
    for noise_setting in noise_settings:
        noise *= sphere_mapping.generate_noise(noise_setting, coord_offset=coord_offset)
    return noise


@timing
def unsketch(sketch: np.ndarray) -> np.ndarray:
    '''
    Convert a sketch back to a DEM using distances from edges.
    Args:
        sketch (np.ndarray): The sketch to convert
    Returns:
        np.ndarray: The DEM
    '''
    h, w = sketch.shape
    # First _unsketch the normal sketch and then spin it by w // 2 and _unsketch again then take max
    unsketched1 = _unsketch(sketch)
    return unsketched1
    sketch2 = np.roll(sketch, w // 2, axis=1)
    unsketched2 = np.roll(_unsketch(sketch2), -w // 2, axis=1)
    unsketched = np.maximum(unsketched1, unsketched2)
    return unsketched


def _unsketch(sketch: np.ndarray) -> np.ndarray:
    colours = np.unique(sketch)
    colours = colours[colours > 0]
    if len(colours) == 0:
        return np.zeros_like(sketch, dtype=np.float32)
    result = np.zeros_like(sketch, dtype=np.float32)
    for colour in colours:
        mask = sketch == colour
        num_components, component_mask = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        for i in range(1, num_components):
            ys, xs = np.where(component_mask == i)
            current_mask = np.zeros_like(sketch)
            current_mask[ys, xs] = 1
            edges = (
                cv2.dilate(current_mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1)
                - current_mask.astype(np.uint8)
            )
            if np.sum(edges) == 0:
                continue

            # central_line = skeletonize(component_mask == i)
            # cent_ys, cent_xs = np.where(central_line)
            # cent_values = sketch[cent_ys, cent_xs]

            edge_ys, edge_xs = np.where(edges > 0)
            edge_values = sketch[edge_ys, edge_xs]
            for y, x in zip(ys, xs):
                # Interpolate each pixel based on distance to edges and edge values
                # And average distance to central line and its values
                edge_distances = np.sqrt((edge_ys - y) ** 2 + (edge_xs - x) ** 2)
                edge_weights = 1 / (edge_distances + 1e-5)
                # edge_weights /= np.sum(edge_weights)

                # cent_distances = np.sqrt((cent_ys - y) ** 2 + (cent_xs - x) ** 2)
                # cent_weights = 1 / (cent_distances + 1e-5)
                # cent_weights /= np.sum(cent_weights)

                values = edge_values  # np.concatenate([edge_values, cent_values])
                weights = edge_weights  # np.concatenate([edge_weights, cent_weights])
                weights /= np.sum(weights)

                # edge_value = np.sum(edge_values * edge_weights)
                # cent_value = np.sum(cent_values * cent_weights)
                # value = (edge_value + cent_value) / 2

                value = (values * weights).sum()
                result[y, x] = value
    result = (result + sketch) / 2
    return result


class SketchArgs:
    def __init__(
        self,
        downsketch: np.ndarray,
        downland_sketch: np.ndarray,
        downtemp_sketch: np.ndarray,
    ):
        self.downsketch = downsketch
        self.downland_sketch = downland_sketch
        self.downtemp_sketch = downtemp_sketch
        self._quad_downsketch = None
        self._quad_downland_sketch = None
        self._quad_downtemp_sketch = None

    @property
    def array(self) -> list[np.ndarray | None]:
        return [self.downsketch, self.downland_sketch, self.downtemp_sketch]

    @property
    def quad_downsketch(self) -> np.ndarray:
        if self._quad_downsketch is None:
            if self.downsketch.shape[1] == 2 * self.downsketch.shape[0]:
                self._quad_downsketch = normal_to_quad(self.downsketch, discrete=True)
            else:
                self._quad_downsketch = self.downsketch
        return self._quad_downsketch

    @property
    def quad_downland_sketch(self) -> np.ndarray:
        if self._quad_downland_sketch is None:
            if self.downland_sketch.shape[1] == 2 * self.downland_sketch.shape[0]:
                self._quad_downland_sketch = normal_to_quad(self.downland_sketch, discrete=True)
            else:
                self._quad_downland_sketch = self.downland_sketch
        return self._quad_downland_sketch

    @property
    def quad_downtemp_sketch(self) -> np.ndarray:
        if self._quad_downtemp_sketch is None:
            if self.downtemp_sketch.shape[1] == 2 * self.downtemp_sketch.shape[0]:
                self._quad_downtemp_sketch = normal_to_quad(self.downtemp_sketch, discrete=True)
            else:
                self._quad_downtemp_sketch = self.downtemp_sketch
        return self._quad_downtemp_sketch


def derive_normal_rivers(
    sketches: SketchArgs,
    planet_cfg: PlanetConfig,
    dem: np.ndarray,
    use_precip: bool = True,
    weights: np.ndarray | None = None,
    efficiency: np.ndarray | None = None,
    normalize: bool = True,
):
    if use_precip and weights is None:
        precip_sketcher = PrecipSketch(planet_cfg)
        full_mapping = precip_sketcher.get_full_mapping()
        river_weights = precip_sketcher.get_sketch(
            sketches.downsketch,
            sketches.downland_sketch,
            sketches.downtemp_sketch,
            full_mapping,
        )
        river_weights = cv2.resize(river_weights, dem.shape[::-1], interpolation=cv2.INTER_CUBIC)
    elif weights is not None:
        river_weights = weights
    else:
        river_weights = np.ones_like(dem)

    rivers = accumulation(dem, river_weights, efficiency, planet_cfg)

    if not normalize:
        return rivers, river_weights
    return rivers / (rivers.max() or 1) * 255, river_weights


def derive_sketch_rivers(
    sketches: SketchArgs,
    planet_cfg: PlanetConfig,
    weights: np.ndarray | None = None,
    efficiency: np.ndarray | None = None,
    noise: np.ndarray | None = None,
    noise_factor: float = 0.5,
    normalize: bool = True,
    return_both: bool = False,
    resolution_scale: float = 1.0,
):
    resolution_scale = min(resolution_scale, 1.0)
    dilated_downsketch = cv2.dilate(
        sketches.downsketch, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    unsketched = unsketch(dilated_downsketch)
    smoothed_sketch = cv2.blur(unsketched, (7, 7)) * 1.2
    h, w = sketches.downsketch.shape
    H = int(h * planet_cfg.delta * resolution_scale)
    W = int(w * planet_cfg.delta * resolution_scale)
    smoothed_sketch = cv2.resize(smoothed_sketch, (W, H), interpolation=cv2.INTER_CUBIC)
    if weights is not None:
        weights = cv2.resize(weights, (W, H), interpolation=cv2.INTER_CUBIC)
    if efficiency is not None:
        efficiency = cv2.resize(efficiency, (W, H), interpolation=cv2.INTER_CUBIC)
    if noise is not None and noise.shape != (H, W):
        noise = cv2.resize(noise, (W, H), interpolation=cv2.INTER_CUBIC)
    if noise is None or noise.shape != (H, W):
        noise = planet_noise(
            (H, W),
            NoiseSettings(
                frequency=0.5
            )
        )
    noise[noise < 0.05] = 0
    smoothed_sketch = smoothed_sketch * (1 + noise * noise_factor)
    normal_sketch = cv2.resize(
        dilated_downsketch, (W, H), interpolation=cv2.INTER_NEAREST
    )
    landcover_sketch = cv2.resize(
        sketches.downland_sketch, (W, H), interpolation=cv2.INTER_NEAREST
    )

    smoothed_sketch[normal_sketch == 0] = 0
    smoothed_sketch[landcover_sketch == LandcoverClasses.SNOW_AND_ICE.gray_colour] = 0
    smoothed_sketch[landcover_sketch == LandcoverClasses.ANTARCTICA.gray_colour] = 0

    normal_river_sketch, weights = derive_normal_rivers(
        sketches,
        planet_cfg,
        smoothed_sketch,
        True,
        weights,
        efficiency,
        normalize,
    )
    normal_river_sketch[landcover_sketch == 0] = 0
    # run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # For debugging
    # os.makedirs("derivation_sketch", exist_ok=True)
    # Image.fromarray(weights).save(f"derivation_sketch/{run_timestamp}_river_weights.tif")
    actual_H = h * planet_cfg.delta
    actual_W = w * planet_cfg.delta

    normal_river_sketch = cv2.resize(normal_river_sketch, (actual_W, actual_H), interpolation=cv2.INTER_NEAREST)
    quad_river_sketch = cv2.dilate(normal_river_sketch, np.ones((3, 3), np.uint8), iterations=2)
    quad_river_sketch = normal_to_quad(quad_river_sketch, discrete=True)
    river_mask = skeletonize(quad_river_sketch ** 2 / (quad_river_sketch.max() ** 2 or 1) * 255 > 2) > 0
    quad_river_sketch[~river_mask] = 0

    # normal_sphere = SphereMapping(shape=(actual_H, actual_W), discrete=True, method="surface-straight")
    # dem_sketch = cv2.resize(
    #     sketches.downsketch, (actual_W, actual_H), interpolation=cv2.INTER_NEAREST
    # )
    # dem_face = normal_sphere.get_tile((-45, 90), zoom=32, atlas=dem_sketch, tile_size=8192)
    # smoothed_sketch = cv2.resize(
    #     smoothed_sketch, (actual_W, actual_H), interpolation=cv2.INTER_NEAREST
    # )
    # smoothed_sketch_face = normal_sphere.get_tile((-45, 90), zoom=32, atlas=smoothed_sketch, tile_size=8192)
    # river_upa_face = normal_sphere.get_tile(
    #     (-45, 90),
    #     zoom=32,
    #     atlas=cv2.dilate(
    #         normal_river_sketch,
    #         np.ones((5, 5), dtype=np.uint8)
    #     ),
    #     tile_size=8192
    # )
    # weights_face = normal_sphere.get_tile(
    #     (-45, 90),
    #     zoom=32,
    #     atlas=cv2.resize(
    #         weights,
    #         (actual_W, actual_H),
    #         cv2.INTER_NEAREST
    #     ),
    #     tile_size=8192
    # )

    # river_upa_face = river_upa_face / (river_upa_face.max() or 1) * 200
    # weights_face = weights_face / (weights_face.max() or 1) * 255

    # smoothed_sketch_face = np.dstack([smoothed_sketch_face] * 3)
    # smoothed_sketch_face[river_upa_face > 1] = np_rgb(river_upa_face, cmap="viridis")[river_upa_face > 1]
    # weights_face = np_rgb(weights_face, cmap="viridis")

    # os.makedirs("derivation_sketch", exist_ok=True)
    # Image.fromarray(dem_face).save("derivation_sketch/sketch_face.jpeg", format="jpeg", quality=95)
    # Image.fromarray(
    #     smoothed_sketch_face.clip(0, 255).astype(np.uint8)
    # ).save("derivation_sketch/smoothed_sketch_face.jpeg", format="jpeg", quality=95)
    # Image.fromarray(weights_face).save("derivation_sketch/weights_face.jpeg", format="jpeg", quality=95)
    # smoothed_sketch = normal_to_quad(smoothed_sketch)
    # normal_rgb_rivers = np_rgb(normal_river_sketch.clip(0, 255), cmap="viridis")
    # rgb_sketch = np.dstack([smoothed_sketch] * 3)
    # rgb_sketch[normal_river_sketch > 5] = normal_rgb_rivers[normal_river_sketch > 5]
    # Image.fromarray(normal_rgb_rivers).save(
    #     os.path.join(self.planet_cfg.test_dir, f"{name}_normal_rivers.png")
    # )
    # Image.fromarray(rgb_sketch.clip(0, 255).astype(np.uint8)).save(
    #     os.path.join(self.planet_cfg.test_dir, f"{name}_sketch_normal_rivers.png")
    # )
    if return_both:
        return quad_river_sketch, normal_river_sketch

    return quad_river_sketch


def rotate_surface_coords(
    surface_coords: tuple[np.ndarray, np.ndarray, np.ndarray],
    dest_coord: tuple[float, float],
    src_coord: tuple[float, float] = (0, 0)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rotate the surface coordinates from one coordinate to another
    """
    x, y, z = surface_coords
    lat, long = src_coord
    assert -90 <= lat <= 90, f"Latitude must be between -90 and 90: {lat:.2f}"
    assert -180 <= long <= 180, f"Longitude must be between -180 and 180: {long:.2f}"
    lat = np.deg2rad(-lat)
    long = np.deg2rad(-long)

    # Rotate the grid about the z axis
    r_z_x = x * np.cos(long) - y * np.sin(long)
    r_z_y = x * np.sin(long) + y * np.cos(long)
    r_z_z = z

    # Rotate the grid about the y axis
    r_y_x = r_z_x * np.cos(lat) + r_z_z * np.sin(lat)
    r_y_z = -r_z_x * np.sin(lat) + r_z_z * np.cos(lat)
    x, y, z = r_y_x, r_z_y, r_y_z

    lat, long = dest_coord
    assert -90 <= lat <= 90, f"Latitude must be between -90 and 90: {lat:.2f}"
    assert -180 <= long <= 180, f"Longitude must be between -180 and 180: {long:.2f}"
    lat = np.deg2rad(lat)
    long = np.deg2rad(long)

    # Rotate the grid about the y axis
    r_y_x = x * np.cos(lat) + z * np.sin(lat)
    r_y_z = -x * np.sin(lat) + z * np.cos(lat)

    # Rotate the grid about the z axis
    r_z_x = r_y_x * np.cos(long) - y * np.sin(long)
    r_z_y = r_y_x * np.sin(long) + y * np.cos(long)
    x, y, z = r_z_x, r_z_y, r_y_z

    return x, y, z


test_x = np.random.rand(256, 256)
test_y = np.random.rand(256, 256)
test_z = np.random.rand(256, 256)

first_rot = np.random.rand(2)
first_rot[0] = first_rot[0] * 180 - 90
first_rot[1] = first_rot[1] * 360 - 180
second_rot = np.random.rand(2)
second_rot[0] = second_rot[0] * 180 - 90
second_rot[1] = second_rot[1] * 360 - 180


assert np.allclose(rotate_surface_coords((test_x, test_y, test_z), (0, 0), (0, 0)), (test_x, test_y, test_z))

first_rot_coords = rotate_surface_coords((test_x, test_y, test_z), first_rot)
second_rot_coords = rotate_surface_coords((test_x, test_y, test_z), second_rot)

first_to_second_rot_coords = rotate_surface_coords(
    first_rot_coords,
    second_rot,
    first_rot
)

assert np.allclose(
    first_to_second_rot_coords,
    second_rot_coords
)


def surface_noise(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    noise_settings: NoiseSettings | list[NoiseSettings],
    coord_offset: tuple[float, float] | None = None
) -> np.ndarray:
    if coord_offset is not None:
        x, y, z = rotate_surface_coords(x, y, z, coord_offset)

    noise = stacked_noise(x, y, z, noise_settings)

    return noise


@profile
def noise_atlas(
    shape: tuple[int, int],
    noise_settings: list[NoiseSettings],
    preview_coords: tuple[float, float] = (0, 0),
    view_coords: tuple[float, float] = (0, 0),
    mask: np.ndarray | None = None
):
    h, w = shape
    noise_ys = np.arange(h).astype(np.float32)
    noise_xs = np.arange(w).astype(np.float32)
    noise_xs, noise_ys = np.meshgrid(noise_xs, noise_ys)
    if mask is not None:
        noise_xs[~mask] = np.nan
        noise_ys[~mask] = np.nan
    surface_coords = QuadSphere(shape=shape).atlas_coords_to_surface_coords(noise_ys, noise_xs)

    prev_lat, prev_long = preview_coords
    view_lat, view_long = view_coords
    assert -90 <= prev_lat <= 90, f"Latitude must be between -90 and 90: {prev_lat:.2f}"
    assert 0 <= prev_long <= 360, f"Longitude must be between 0 and 360: {prev_long:.2f}"

    assert -90 <= view_lat <= 90, f"Latitude must be between -90 and 90: {view_lat:.2f}"
    assert 0 <= view_long <= 360, f"Longitude must be between 0 and 360: {view_long:.2f}"

    surface_coords = rotate_surface_coords(
        surface_coords,
        dest_coord=(prev_lat, prev_long - 180),
        src_coord=(view_lat, view_long - 180)
    )
    noise = (stacked_multi_noise(surface_coords, noise_settings) * 255).clip(0, 255).astype(np.uint8)
    return noise


def format_landcover_noise(noise: np.ndarray, metadata: LandCoverEventMetadata) -> np.ndarray:
    # 2. Min-max normalize the noise
    if noise.min() != noise.max():
        noise = (noise - noise.min()) / (noise.max() - noise.min())

    # 3. Use primary_ratio to blend between primary and secondary classes
    primary_ratio = metadata.primary_ratio
    primary_colour = metadata.primary_class.colour
    if metadata.secondary_class is not None:
        secondary_colour = metadata.secondary_class.colour
    else:
        secondary_colour = 0
    preview = np.zeros(noise.shape, dtype=np.uint8)
    preview_noise_tile = noise.copy()
    preview[preview_noise_tile < primary_ratio] = primary_colour
    preview[preview_noise_tile >= primary_ratio] = secondary_colour

    return preview


def create_dem_preset(
        metadata: DEMEventMetadata,
        planet_shape: tuple[int, int],
        preview_shape: tuple[int, int] = None,
        noise: np.ndarray = None) -> np.ndarray:
    '''
    Create a DEM preview
    '''
    if preview_shape is None:
        preview_shape = planet_shape
    H, W = planet_shape
    h, w = preview_shape

    assert H >= h and W >= w, "Planet shape must be larger than preview shape"

    # 1. Sample planet noise
    if noise is None:
        noise = planet_noise(planet_shape, metadata.noise_settings)

    noise = noise * 255

    preview_noise_tile = noise[(H-h)//2:(H-h)//2+h, (W-w)//2:(W-w)//2+w]
    preview = np.clip(preview_noise_tile, 0, 255).astype(np.uint8)

    return preview


def get_noise_threshold_mask(
    mask: np.ndarray,
    coord: tuple[int, int],
    noise: np.ndarray,
    threshold: float = 50.0
) -> np.ndarray:
    """
    TODO Improve this
    Take some noise and use object detection from the noise,
    plus the threshold to create a mask of objects.
    """
    y, x = coord
    coord_val = noise[y, x]
    noise_objects = np.ones_like(noise, dtype=np.uint8) * 255
    noise_objects[noise > coord_val + threshold / 2] = 0
    noise_objects[noise < coord_val - threshold / 2] = 0
    noise_objects[noise == 0] = 0
    # Floodfill from the coord and get the object
    cv2.floodFill(noise_objects, None, (x, y), 128)
    object_mask = noise_objects == 128
    noise_objects[~object_mask] = 0
    noise_objects[object_mask] = 128
    ymin, xmin, ymax, xmax = get_bounds(object_mask)
    for fy, fx in itertools.product([ymin - 1, ymax + 1], [xmin - 1, xmax + 1]):
        if fy < 0 or fy >= noise_objects.shape[0] or fx < 0 or fx >= noise_objects.shape[1]:
            continue
        if noise_objects[fy, fx] == 0:
            cv2.floodFill(noise_objects, None, (fx, fy), 255)
            noise_objects[noise_objects == 0] = 128  # Fill the gaps
            break

    return (noise_objects == 128)  # & mask
