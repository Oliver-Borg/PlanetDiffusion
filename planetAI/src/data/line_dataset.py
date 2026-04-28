from math import sqrt
import numpy as np
import itertools
import cv2
from skimage.morphology import skeletonize
from .utils import PlanetConfig, timing
from .sphere_mapping import SphereMapping
import numpy.typing as npt
from tqdm import tqdm
from .bathy import ClassColours
from dataclasses import dataclass
import os
from PIL import Image


@dataclass
class Point:
    lat: float
    long: float


@dataclass
class CartLine:
    start: Point
    end: Point


@dataclass
class BoundaryLine(CartLine):
    colour: ClassColours
    left_plate: str
    right_plate: str


@dataclass
class Plate:
    lines: list[CartLine]

    @timing
    def filter_coordinates(
        self,
        sphere: SphereMapping
    ) -> tuple[
        np.ndarray,  # (m, 3)
        np.ndarray,  # (m, 3)
        np.ndarray,  # (m, 3)
    ]:
        # ys, xs = np.where(sphere.atlas >= 0)
        # x, y, z = sphere.atlas_coords_to_surface_coords(ys, xs)
        # all_coords = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)  # (n, 3)

        points = np.array(
            [tuple(reversed(convert_to_atlas_coords(line.start, sphere))) for line in self.lines],
            dtype=np.int32
        )
        atlas = sphere.atlas.copy()

        # polygon_vertices = points.reshape((-1, 1, 2))
        cv2.fillPoly(atlas, [points], 255)

        ys, xs = np.where(atlas == 255)
        x, y, z = sphere.atlas_coords_to_surface_coords(ys, xs)
        all_coords = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)  # (n, 3)
        return all_coords, ys, xs

        # Minimum Enclosing Sphere solution
        # points = np.array([convert_to_surface_coords(line.start, sphere) for line in self.lines])

        # centre, r2 = miniball.get_bounding_ball(points)

        # dist2 = np.sum((all_coords - centre) ** 2, axis=1)

        # approx_valid = dist2 <= r2

        # all_coords = all_coords[approx_valid]
        # ys = ys[approx_valid]
        # xs = xs[approx_valid]
        # return all_coords, ys, xs

        # Look at PNPOLY
        # for line in self.lines:
        #     atlas = sphere.atlas.copy()
        #     a, b, c, d = convert_line_to_plane(line, sphere)
        #     val = a * all_coords[:, 0] + b * all_coords[:, 1] + c * all_coords[:, 2] + d
        #     valid = val >= 0
        #     all_coords = all_coords[valid]
        #     ys = ys[valid]
        #     xs = xs[valid]
        #     atlas[ys, xs] = 255
        #     continue
        # return all_coords, ys, xs


def convert_line_to_plane(line: CartLine, sphere: SphereMapping) -> tuple[float, float, float, float]:
    src = convert_to_surface_coords(line.start, sphere)
    dest = convert_to_surface_coords(line.end, sphere)
    origin = np.array([0, 0, 0])
    x = src - origin
    y = dest - origin
    normal = np.cross(x, y)
    d = -np.dot(normal, x)
    a, b, c = normal
    return a, b, c, d


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


def rotate_point(point: tuple[float, float], ang: float, center: tuple[float, float] = (0.5, 0.5)):
    y, x = point
    cy, cx = center
    y -= cy
    x -= cx
    ang = np.deg2rad(ang)
    x_rot = x * np.cos(ang) - y * np.sin(ang)
    y_rot = x * np.sin(ang) + y * np.cos(ang)
    return y_rot + cy, x_rot + cx


class LineDataset:
    def __init__(self, atlas: np.ndarray, thickness: int = 3, min_val: float = 0):
        self.min_val = min_val
        self.lines = self._extract_lines(atlas)
        self.thickness = thickness

    @timing
    def _extract_lines(self, atlas: np.ndarray) -> np.ndarray:
        # lines is an n x 6 array with src u, src v, dest u, dest v, src colour, dest colour
        lines = []
        h, w = atlas.shape
        ys, xs = np.where(atlas > self.min_val)
        for y, x in tqdm(list(zip(ys, xs))):
            for dy, dx in itertools.product((-1, 0, 1), (-1, 0, 1)):
                if dy == 0 and dx == 0:
                    continue
                if 0 > y or y + dy >= atlas.shape[0]:
                    continue
                if 0 > x or x + dx >= atlas.shape[1]:
                    continue
                next_pixel = atlas[y + dy, x + dx]
                if next_pixel > self.min_val:
                    lines.append([
                        y / h,
                        x / w,
                        (y + dy) / h,
                        (x + dx) / w,
                        atlas[y, x],
                        next_pixel
                    ])
        return np.array(lines)

    def get_sketch(self, shape: tuple[int, int], rotation: float = 0.0, use_max: bool = True) -> np.ndarray:
        """
        Returns a float sketch from the lines
        """
        # Create a blank sketch
        sketch = np.zeros(shape, dtype=np.float32)
        num_placements = np.zeros(shape, dtype=np.int32)
        h, w = shape
        # Iterate over the lines and draw them on the sketch
        for line in tqdm(self.lines):
            src_y, src_x = rotate_point((line[0], line[1]), rotation)
            dest_y, dest_x = rotate_point((line[2], line[3]), rotation)

            src_y = int(np.array(src_y * shape[0]).round().clip(0, h - 1))
            src_x = int(np.array(src_x * shape[1]).round().clip(0, w - 1))
            dest_y = int(np.array(dest_y * shape[0]).round().clip(0, h - 1))
            dest_x = int(np.array(dest_x * shape[1]).round().clip(0, w - 1))

            line_length = max(sqrt((src_y - dest_y) ** 2 + (src_x - dest_x) ** 2), 1.0)
            src_colour = line[4]
            dest_colour = line[5]
            points = get_line((src_y, src_x), (dest_y, dest_x))
            for y, x in points:
                start_dist = sqrt((src_y - y) ** 2 + (src_x - x) ** 2)
                w1 = min(start_dist / line_length, 1.0)
                w2 = 1.0 - w1
                num_placements[y, x] += 1
                if use_max:
                    sketch[y, x] = np.maximum(w1 * src_colour + w2 * dest_colour, sketch[y, x])
                else:
                    sketch[y, x] += w1 * src_colour + w2 * dest_colour
        num_placements = np.maximum(num_placements, 1)
        if not use_max:
            sketch /= num_placements
        return sketch


@timing
def line_rotater(
    line_data: np.ndarray,
    rotation: float,
    do_skeletonize: bool = False,
    min_val: float = 0,
    use_max: bool = True,
) -> np.ndarray:
    if not line_data.any():
        return line_data
    if do_skeletonize:
        line_data_mask = skeletonize(line_data) > 0
        line_data[~line_data_mask] = 0
    line_dataset = LineDataset(line_data, min_val=min_val)
    rotated = line_dataset.get_sketch(line_data.shape, rotation=rotation, use_max=use_max)
    return rotated


@timing
def line_resizer(
    line_data: np.ndarray,
    dest_shape: tuple[int, int],
    do_skeletonize: bool = False,
    thickness: int = 3,
    min_val: float = 0,
    use_max: bool = True,
) -> np.ndarray:
    # TODO Investigate doing something where we use the index of the line in the resized atlas
    # Resize the line to the new shape
    if do_skeletonize:
        line_data_mask = skeletonize(line_data) > 0
        line_data[~line_data_mask] = 0
    line_dataset = LineDataset(line_data, thickness=thickness, min_val=min_val)
    return line_dataset.get_sketch(dest_shape, use_max=use_max)


def get_line_segments(
    src: tuple[int, int], dest: tuple[int, int], width: int
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """If the source or destination points are outside the width,
    find the intersection with the width and wrap around."""
    src_x, src_y = src
    dest_x, dest_y = dest
    src_x = src_x % width
    dest_x = dest_x % width

    if src_x > dest_x:
        # Ensure src_x is always less than dest_x for easier calculations
        src_x, dest_x = dest_x, src_x
        src_y, dest_y = dest_y, src_y
    dest = (dest_x, dest_y)
    src = (src_x, src_y)

    if dest_x - src_x > width // 2:
        # If the distance is more than half the width, we need to wrap around
        src_x += width
        dx = dest_x - src_x
        dy = dest_y - src_y
        m = dy / dx
        c = src_y - m * src_x
        # Calculate the intersection points with the edges of the width
        y_int = m * (width - 1) + c
        return [
            (dest, (width - 1, int(round(y_int)))),
            ((0, int(round(y_int))), src)
        ]
    else:
        return [(src, dest)]


def length(src: tuple[int, int], dest: tuple[int, int]) -> float:
    """Calculate the length of the line segment from src to dest."""
    return ((src[0] - dest[0]) ** 2 + (src[1] - dest[1]) ** 2) ** 0.5


def get_distance_matrix(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    # edge_coords = np.array(np.where(edge_mask > 0)).transpose((1, 0))
    # mask_coords = np.array(np.where(mask > 0)).transpose((1, 0))
    # edge_coords = edge_coords.reshape(edge_coords.shape[0], 1, 2)
    # mask_coords = mask_coords.reshape(1, mask_coords.shape[0], 2)
    # distances: np.ndarray = np.sum((edge_coords - mask_coords)**2, axis=2)
    num_channels = first.shape[-1]
    first = first.reshape(first.shape[0], 1, num_channels)
    second = second.reshape(1, second.shape[0], num_channels)
    distances = np.sqrt(np.sum((first - second) ** 2, axis=2))
    return distances


def array_dot(
    xs: npt.NDArray[np.float32],  # shape (n, 3)
    ys: npt.NDArray[np.float32],  # shape (n, 3)
) -> npt.NDArray[np.float32]:      # shape (n,)
    return np.einsum('ij,ij->i', xs, ys)


def get_distances_to_line(
    all_coords: np.ndarray,
    start_coords: np.ndarray,
    end_coords: np.ndarray,
    shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    distance_matrix = np.zeros(shape) + np.Infinity
    closest_line_indices = np.zeros(shape, dtype=np.uint16)
    for i, (start_coord, end_coord) in tqdm(list(enumerate(zip(start_coords, end_coords)))):
        # TODO Add some filtering to make this faster
        x = start_coord - all_coords
        y = end_coord - all_coords
        xx = np.sum(x ** 2, axis=-1)
        xy = array_dot(x, y)
        yy = np.sum(y ** 2, axis=-1)
        t = (xx - xy) / (yy - 2 * xy + xx)
        t = np.repeat(t[:, np.newaxis], 3, axis=1)
        t = np.clip(t, 0, 1)
        z = x + t * (y - x)
        dist: np.ndarray = np.linalg.norm(z, axis=1).reshape(shape)
        closest_line_indices[dist < distance_matrix] = i
        distance_matrix = np.minimum(distance_matrix, dist)

    return distance_matrix, closest_line_indices


def reconstruct_from_indices_with_interp(
    all_coords: np.ndarray,  # (n, 3)
    start_coords: np.ndarray,  # (l, 3)
    end_coords: np.ndarray,  # (l, 3)
    small_indices: np.ndarray,  # (h, w)
    small_shape: tuple[int, int],
    shape: tuple[int, int]
) -> np.ndarray:
    # WIP
    h, w = small_shape
    next_shape = (2 * h, 2 * w)
    distance_matrix = np.zeros(next_shape) + np.Infinity
    refined_indices = np.zeros(next_shape, dtype=np.uint16)
    for rh in range(-1, 2):
        for rw in range(-1, 2):
            indices = cv2.resize(
                np.roll(small_indices, (rh, rw)), tuple(reversed(next_shape)), interpolation=cv2.INTER_NEAREST_EXACT
            )
            dist = reconstruct_from_indices(
                all_coords,
                start_coords,
                end_coords,
                indices,
                next_shape
            )
            refined_indices[dist < distance_matrix] = indices[dist < distance_matrix]
            distance_matrix = np.minimum(distance_matrix, dist)
    if next_shape == shape:
        return distance_matrix
    else:
        return reconstruct_from_indices_with_interp(
            all_coords,
            start_coords,
            end_coords,
            refined_indices,
            next_shape,
            shape
        )


def reconstruct_from_indices(
    all_coords: np.ndarray,  # (n, 3)
    start_coords: np.ndarray,  # (l, 3)
    end_coords: np.ndarray,  # (l, 3)
    indices: np.ndarray,  # (n,)
    shape: tuple[int, int]
) -> np.ndarray:
    indices = indices.flatten()
    nearest_start_coords = start_coords[indices]  # (n, 3)
    nearest_end_coords = end_coords[indices]  # (n, 3)
    x = nearest_start_coords - all_coords  # (n, 3)
    y = nearest_end_coords - all_coords  # (n, 3)
    xx = np.sum(x ** 2, axis=-1)  # (n,)
    xy = array_dot(x, y)  # (n,)
    yy = np.sum(y ** 2, axis=-1)  # (n,)
    t = (xx - xy) / (yy - 2 * xy + xx)  # (n,)
    t = np.repeat(t[:, np.newaxis], 3, axis=1)  # (n, 3)
    t = np.clip(t, 0, 1)  # (n, 3)
    z = x + t * (y - x)  # (n, 3)
    dist: np.ndarray = np.linalg.norm(z, axis=1).reshape(shape)
    return dist


def extract_bathy_segments(planet_cfg: PlanetConfig) -> list[BoundaryLine]:
    """Extract bathymetry segments from the atlas file.
    Returns a list of tuples containing the start and end coordinates (long, lat) and the class colour."""
    lines_file = os.path.join(planet_cfg.data_dir, "PB2002_boundaries.dig.txt")
    with open(lines_file, "r") as f:
        lines = f.readlines()
    prev_point = None
    current_class = None

    segments: list[BoundaryLine] = []

    max_lines = 100
    left_plate = ""
    right_plate = ""

    for line in tqdm(lines):
        if current_class is None:
            if line[2] == "-":
                current_class = ClassColours.TRANSFORM
            elif line[2] == "/" or line[2] == "\\":
                current_class = ClassColours.CONVERGENT
            else:
                print(f"Unknown symbol {line[2]} from line {line}")
                current_class = ClassColours.DIVERENT
            left_plate = line[:2]
            right_plate = line[3:].strip()
            continue
        if "*** end of line segment ***" in line:
            # break
            current_class = None
            prev_point = None
            continue
        long, lat = line.split(",")
        long = float(long)
        lat = float(lat)
        if prev_point is None:
            prev_point = Point(lat=lat, long=long)
        else:
            segments.append(BoundaryLine(prev_point, Point(lat=lat, long=long), current_class, left_plate, right_plate))
            max_lines -= 1
            if max_lines == 0:
                break
            prev_point = Point(lat=lat, long=long)
    return segments


def extract_bathy_plates(planet_cfg: PlanetConfig) -> dict[str, Plate]:
    lines_file = os.path.join(planet_cfg.data_dir, "PB2002_plates.dig.txt")
    with open(lines_file, "r") as f:
        lines = f.readlines()
    current_name = None
    prev_point = None
    current_lines = []
    plates = {}
    for line in lines:
        if current_name is None:
            current_name = line.strip()
        elif line.strip() == "*** end of line segment ***":
            plates[current_name] = Plate(
                current_lines
            )
            current_name = None
            current_lines = []
        else:
            long, lat = line.split(",")
            long = float(long)
            lat = float(lat)
            if prev_point is None:
                prev_point = Point(lat=lat, long=long)
            else:
                current_lines.append(
                    CartLine(
                        prev_point,
                        Point(lat=lat, long=long)
                    )
                )
                prev_point = Point(lat=lat, long=long)
    return plates


def convert_to_pixel_coords(point: tuple[float, float], atlas_shape: tuple[int, int]) -> tuple[int, int]:
    """Convert a point in (long, lat) to pixel coordinates in the atlas."""
    long, lat = point
    h, w = atlas_shape
    y = int(round((90 - float(lat)) / 180 * h))
    x = int(round((float(long) + 180) / 360 * w))
    return x, y


def convert_to_surface_coords(point: Point, sphere_mapping: SphereMapping) -> tuple[float, float, float]:
    """Convert a point in (long, lat) to (x, y, z) cartesian coordinates."""
    x, y, z = sphere_mapping.get_surface_coords((point.lat, point.long), tile_size=1)
    return float(x), float(y), float(z)


def convert_to_atlas_coords(point: Point, sphere_mapping: SphereMapping) -> tuple[float, float]:
    y, x = sphere_mapping.get_tile_mapping((point.lat, point.long), tile_size=1, round_result=True)
    return float(y), float(x)


def get_bathy_atlas(planet_cfg: PlanetConfig) -> np.ndarray:
    segments = extract_bathy_segments(planet_cfg)
    h, w = planet_cfg.H, planet_cfg.W

    atlas = np.zeros((h, w), dtype=np.uint8)

    start_xs = []
    start_ys = []
    end_xs = []
    end_ys = []

    for seg in segments:
        src = seg.start
        dest = seg.end
        col = seg.colour.value
        src = convert_to_pixel_coords(src, atlas.shape)
        dest = convert_to_pixel_coords(dest, atlas.shape)
        for small_src, small_dest in get_line_segments(src, dest, atlas.shape[1]):
            if length(small_src, small_dest) > 1000:
                continue
            start_xs.append(small_src[0])
            start_ys.append(small_src[1])
            end_xs.append(small_dest[0])
            end_ys.append(small_dest[1])
            atlas = cv2.line(atlas, small_src, small_dest, col, thickness=5)

    return atlas


def get_required_coords(
    shape: tuple[int, int], segments: list[BoundaryLine]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = shape
    atlas = np.zeros((h, w), dtype=np.uint8)
    sphere = SphereMapping(atlas)
    ys, xs = np.where(atlas >= 0)
    x, y, z = sphere.atlas_coords_to_surface_coords(ys, xs)
    all_atlas_coords = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)

    start_coords = []
    end_coords = []

    for seg in segments:
        src = seg.start
        dest = seg.end
        start_coords.append(convert_to_surface_coords(src, sphere))
        end_coords.append(convert_to_surface_coords(dest, sphere))

    start_coords = np.array(start_coords)
    end_coords = np.array(end_coords)
    return all_atlas_coords, start_coords, end_coords


def extract_bathy_distances(planet_cfg: PlanetConfig) -> np.ndarray:
    segments = extract_bathy_segments(planet_cfg)
    h, w = planet_cfg.h, planet_cfg.w
    all_atlas_coords, start_coords, end_coords = get_required_coords((h, w), segments)

    small_distance_matrix, indices = get_distances_to_line(
        all_coords=all_atlas_coords,
        start_coords=start_coords,
        end_coords=end_coords,
        shape=(h, w)
    )

    assert np.array_equal(
        small_distance_matrix,
        reconstruct_from_indices(
            all_coords=all_atlas_coords,
            start_coords=start_coords,
            end_coords=end_coords,
            indices=indices.flatten(),
            shape=(h, w)
        )
    )

    H, W = planet_cfg.H, planet_cfg.W

    all_atlas_coords, start_coords, end_coords = get_required_coords((H, W), segments)
    # distance_matrix = reconstruct_from_indices_with_interp(
    #     all_coords=all_atlas_coords,
    #     start_coords=start_coords,
    #     end_coords=end_coords,
    #     small_indices=indices,
    #     small_shape=(h, w),
    #     shape=(H, W)
    # )

    # assert np.array_equal(distance_matrix, reconstructed_matrix)

    # Normalize the distance matrix to 0-255 range
    distance_matrix = cv2.normalize(
        small_distance_matrix, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    indices = cv2.normalize(
        indices, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    return distance_matrix


# TODO Use rift names to construct plates and limit number of lines per set of coords

if __name__ == "__main__":
    planet_cfg = PlanetConfig(size=0, downscale_offset=3)
    atlas = np.zeros((planet_cfg.H, planet_cfg.W), dtype=np.uint8)
    Plate(
        [
            CartLine(Point(lat=0, long=0), Point(lat=10, long=0)),
            CartLine(Point(lat=10, long=0), Point(lat=10, long=-10)),
            CartLine(Point(lat=10, long=-10), Point(lat=0, long=-10)),
            CartLine(Point(lat=0, long=-10), Point(lat=0, long=0)),
        ]
    ).filter_coordinates(SphereMapping(atlas))
    plates = extract_bathy_plates(planet_cfg)
    np.random.seed(0)
    random_colors = (np.random.rand(256, 3) * 255).astype(np.uint8)
    bathy_plates = np.zeros((planet_cfg.H, planet_cfg.W, 3), dtype=np.uint8)
    for i, plate_name in enumerate(plates):
        all_coords, ys, xs = plates[plate_name].filter_coordinates(SphereMapping(atlas))
        bathy_plates[ys, xs] = random_colors[i]

    Image.fromarray(bathy_plates).save(os.path.join(planet_cfg.test_dir, "bathy_plate.png"))

    exit(0)
    atlas = extract_bathy_distances(planet_cfg)

    discretised = (atlas // 20 + 1) * 15
    discretised[atlas == 0] = 0

    Image.fromarray(atlas).save(os.path.join(planet_cfg.test_dir, "bathy_distance.png"))
    Image.fromarray(discretised).save(os.path.join(planet_cfg.test_dir, "bathy_distance_sketch.png"))
    # atlas = get_bathy_atlas(planet_cfg)

    # Image.fromarray(atlas).save(os.path.join(planet_cfg.test_dir, "manual_bathy_lines.png"))
