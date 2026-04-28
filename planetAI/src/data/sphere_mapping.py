import logging
import numpy as np
import math
from PIL import Image as img
from tqdm import tqdm
import cv2

from scipy.spatial.transform import Rotation as R
from typing import Annotated, Callable, Literal, NamedTuple
import numpy.typing as npt

from planetAI.src.data.utils import timing
from skimage.morphology import skeletonize

from .noise_settings import NoiseSettings
from .noise_funcs import stacked_noise

try:
    from line_profiler import profile
except Exception:
    from .utils import profile

img.MAX_IMAGE_PIXELS = 933120000


def show_snapshot(box: np.ndarray, axis: int, front: bool = True):
    """
    This function takes a 3D grid and shows a snapshot of it along the specified axis.
    """
    d = box.shape[axis]
    box = np.swapaxes(box, 0, axis)
    snapshot = np.max(box[: d // 2], axis=0) if front else np.max(box[d // 2:], axis=0)
    snapshot = snapshot.astype(np.uint8)
    img.fromarray(snapshot).show()


SampleMode = Literal["mean", "max", "min"]


@profile
def bilinear_sample(
    image: np.ndarray, y: np.ndarray, x: np.ndarray,
    sampling_mode: SampleMode = "mean"
) -> np.ndarray:
    """
    Sample an image using bilinear interpolation.
   :param image: A 2D numpy array with the image.
   :param y: A numpy array with the y coordinates.
   :param x: A numpy array with the x coordinates.

   :return: A numpy array with the sampled values.
    """

    # TODO Fix noise on edges and black pixels on Antarctica

    h, w = image.shape[:2]
    y = np.clip(y, 0, h - 1)
    x = x % w  # Assumes wrap-around

    y_f = np.floor(y).astype(np.int16)
    y_c = y_f + 1
    x_f = np.floor(x).astype(np.int16)
    x_c = x_f + 1

    dy_f = y - y_f
    dy_c = y_c - y
    dx_f = x - x_f
    dx_c = x_c - x

    x_c %= w
    y_c = np.clip(y_c, 0, h - 1)

    if len(image.shape) == 3:
        c = image.shape[2]
        # Expand from h, w to h, w, c
        dy_f = np.broadcast_to(dy_f[..., np.newaxis], dy_f.shape + (c,))
        dy_c = np.broadcast_to(dy_c[..., np.newaxis], dy_c.shape + (c,))
        dx_f = np.broadcast_to(dx_f[..., np.newaxis], dx_f.shape + (c,))
        dx_c = np.broadcast_to(dx_c[..., np.newaxis], dx_c.shape + (c,))

    if sampling_mode == "mean":
        sampled = (image[y_f, x_f] * dx_c + image[y_f, x_c] * dx_f) * dy_c + (
            image[y_c, x_f] * dx_c + image[y_c, x_c] * dx_f
        ) * dy_f
    elif sampling_mode == "max":
        sampled = np.maximum(
            np.maximum(image[y_f, x_f], image[y_f, x_c]),
            np.maximum(image[y_c, x_f], image[y_c, x_c])
        )
    elif sampling_mode == "min":
        sampled = np.minimum(
            np.minimum(image[y_f, x_f], image[y_f, x_c]),
            np.minimum(image[y_c, x_f], image[y_c, x_c])
        )
    else:
        raise ValueError(
            f"Invalid bilinear sampling mean: {sampling_mode}. "
            "Must be one of 'mean', 'max', or 'min'."
        )

    return sampled.astype(image.dtype)


def rotation_matrix(axis: tuple[float, float, float], theta: float) -> np.ndarray:
    """
    Get the rotation matrix for a rotation about an axis.
   :param axis: A tuple with the x, y and z components of the axis.
   :param theta: The angle of rotation in radians.

   :return: A 3x3 numpy array with the rotation matrix.
    """
    r = R.from_rotvec(np.array(axis) * theta)
    return r.as_matrix()


def apply_rotation(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, rotation: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply the rotation matrix to the 3D coordinates.
    """
    x_r = rotation[0, 0] * x + rotation[0, 1] * y + rotation[0, 2] * z
    y_r = rotation[1, 0] * x + rotation[1, 1] * y + rotation[1, 2] * z
    z_r = rotation[2, 0] * x + rotation[2, 1] * y + rotation[2, 2] * z

    return x_r, y_r, z_r


def rotate_atlas_coords(
    atlas_y: np.ndarray, atlas_x: np.ndarray, angle: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotate the atlas coordinates by a given angle in degrees.
    :param atlas_y: A numpy array with the y coordinates in the atlas.
    :param atlas_x: A numpy array with the x coordinates in the atlas.
    :param angle: The angle of rotation in degrees.
    :return: A tuple with the rotated y and x coordinates in the atlas.
    """
    angle_rad = np.deg2rad(angle)
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    coords = np.vstack((atlas_x.flatten(), atlas_y.flatten()))
    rotated_coords = rotation_matrix @ coords
    rotated_y = rotated_coords[1, :].reshape(atlas_y.shape)
    rotated_x = rotated_coords[0, :].reshape(atlas_x.shape)
    return rotated_y, rotated_x


def sample_resize(
    image: np.ndarray,
    factor: int,
    sample_mode: SampleMode = "mean",
) -> np.ndarray:
    h, w = image.shape[:2]
    if factor > 0:
        # Upsize the image using nearest
        return cv2.resize(
            image,
            (w * 2 ** factor, h * 2 ** factor),
            interpolation=cv2.INTER_NEAREST
        )
    elif factor == 0:
        return image

    stride0 = image[:-1:2, :-1:2]
    stride1 = image[1::2, :-1:2]
    stride2 = image[:-1:2, 1::2]
    stride3 = image[1::2, 1::2]

    if sample_mode == "mean":
        sampled = np.stack((stride0, stride1, stride2, stride3), axis=-1)
        sampled = np.mean(sampled, axis=-1)
    elif sample_mode == "max":
        sampled = np.stack((stride0, stride1, stride2, stride3), axis=-1)
        sampled = np.max(sampled, axis=-1)
    elif sample_mode == "min":
        sampled = np.stack((stride0, stride1, stride2, stride3), axis=-1)
        sampled = np.min(sampled, axis=-1)
    else:
        raise ValueError(
            f"Invalid sample mode: {sample_mode}. "
            "Must be one of 'mean', 'max', or 'min'."
        )

    return sample_resize(sampled, factor + 1, sample_mode=sample_mode)


class SurfaceCoords(NamedTuple):
    """
    A class to represent the surface coordinates of a tile on the sphere.
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray


class AtlasCoords(NamedTuple):
    """
    A class to represent the atlas coordinates of a tile on the sphere.
    """

    y: np.ndarray
    x: np.ndarray


class QuadCoords(NamedTuple):
    """
    A class to represent the quad coordinates of a tile on the sphere.
    """

    u: np.ndarray
    v: np.ndarray


class LatLongCoords(NamedTuple):
    """
    A class to represent the WGS84 map coordinates of a tile on the sphere.
    """

    lat: np.ndarray
    long: np.ndarray


UP = 1
RIGHT = 2
DOWN = 3
LEFT = 4


# face 1, face 2 -> face 1 edge, face 2 edge
connectivity: dict[tuple[int, int], tuple[int, int]] = {
    (0, 1): (DOWN, UP),
    (0, 2): (RIGHT, UP),
    (0, 3): (UP, UP),
    (0, 4): (LEFT, UP),
    (1, 0): (UP, DOWN),
    (1, 2): (RIGHT, LEFT),
    (1, 4): (LEFT, RIGHT),
    (1, 5): (DOWN, UP),
    (2, 0): (UP, RIGHT),
    (2, 1): (LEFT, RIGHT),
    (2, 3): (RIGHT, LEFT),
    (2, 5): (DOWN, RIGHT),
    (3, 0): (UP, UP),
    (3, 2): (LEFT, RIGHT),
    (3, 4): (RIGHT, LEFT),
    (3, 5): (DOWN, DOWN),
    (4, 0): (UP, LEFT),
    (4, 3): (LEFT, RIGHT),
    (4, 1): (RIGHT, LEFT),
    (4, 5): (DOWN, LEFT),
    (5, 1): (UP, DOWN),
    (5, 2): (RIGHT, DOWN),
    (5, 3): (DOWN, DOWN),
    (5, 4): (LEFT, DOWN),
}


class SphereMapping:
    """
    This class takes a 2D atlas with the WGS84 coordinates and maps it to a virtual 3D sphere.
    We can then sample grids given a coordinate, rotation (about the coordinate)
    and desired pixel width and height to get a 2D tile.

    This is done so that we can sample grids that have the same area on the sphere.
    """

    def __init__(
        self,
        atlas: np.ndarray | None = None,
        shape: tuple[int, int] | None = None,
        method: Literal[
            "surface-to-center", "pinhole", "surface-straight"
        ] = "surface-to-center",
        discrete: bool = False,
        sampling_mode: SampleMode = "mean",
    ):
        """
       :param atlas: A 2D numpy array with the WGS84 coordinates.
       :param shape: A tuple with the shape of the atlas.
       :param method: The method used to project the tile to the surface.
        It is recommended to use "surface-to-center" for sampling and inference and surface-straight for viewing.
       :param discrete: Whether to use discrete sampling for rotations and sampling.
        """
        assert (
            atlas is not None or shape is not None
        ), "Either atlas or shape must be provided."
        self.atlas = None
        if atlas is None:
            self.set_shape(shape)
        else:
            self.set_atlas(atlas)

        self.sampling_mode = sampling_mode

        self.method = method
        self.discrete = discrete

    def set_atlas(self, atlas: np.ndarray) -> None:
        """
        Set the atlas.
        """
        self.set_shape(atlas.shape)
        self.atlas = atlas

    def set_shape(self, shape: tuple[int, int] | tuple[int, int, int]) -> None:
        """
        Set the shape of the atlas.
        """
        self.height = shape[0]
        self.width = shape[1]
        self.channels = 1 if len(shape) == 2 else shape[2]
        assert (
            self.width == 2 * self.height
        ), f"The atlas must have a 2:1 aspect ratio and shape should be (h, w). Got {shape}"
        self.pixel_resolution = self.width / 360.0  # Pixels/degree
        self.degree_resolution = 1 / self.pixel_resolution  # Degrees/pixel
        self.radius = self.width / (2 * np.pi)  # TODO Maybe ceil this since the radius and diameter now don't agree
        self.int_diameter = math.ceil(2 * self.radius)

    def get_surface_coords(
        self, coord: tuple[float, float], tile_size: int = 256, zoom: float = 1.0
    ) -> SurfaceCoords:
        """
        This function takes a WGS84 coordinate and returns the surface coordinates of the tile.
       :param coord: A tuple with the latitude and longitude in degrees.
       :param tile_size: The width and height of the tile.
       :param zoom: The zoom level of the tile.
       :param method: The method used to project the tile to the surface.
            "surface-to-center": The tile is projected from the surface to the center of the sphere.
            "pinhole": The tile is projected using a virtual pinhole camera.
            "surface-straight": The tile is projected straight from the surface to the sphere.

        """

        h_s = tile_size // 2
        # zoom /= (self.radius / 32 / (2 * np.pi))

        if self.method == "surface-to-center":
            x = np.ones((tile_size, tile_size)) * self.radius
            y_range = np.linspace(-h_s, h_s, tile_size) / zoom
            y = np.outer(np.ones(tile_size), y_range)
            z_range = np.linspace(h_s, -h_s, tile_size) / zoom
            z = np.outer(z_range, np.ones(tile_size))
            # We want to find the sphere intercept of a line from the origin to a point on the grid
            length = np.sqrt(x**2 + y**2 + z**2)
            surface_x = x * self.radius / length
            surface_y = y * self.radius / length
            surface_z = z * self.radius / length

        elif self.method == "pinhole":
            focal_distance = (1 + 1 / zoom) * self.radius

            fx = focal_distance
            fy = 0
            fz = 0

            x = np.zeros((tile_size, tile_size)) + focal_distance + tile_size / 2
            y_range = np.linspace(-h_s, h_s - 1, tile_size)
            y = np.outer(np.ones(tile_size), y_range)
            z_range = np.linspace(h_s - 1, -h_s, tile_size)
            z = np.outer(z_range, np.ones(tile_size))

            # We want to find the sphere intercept of lines from a reference frame through a focal point

            # O: origins of the lines
            # u: Direction of the lines

            # 1. Find direction vectors for the lines from the reference frame to the focal point
            u_x = fx - x
            u_y = fy - y
            u_z = fz - z

            # 2. Normalize the direction vectors
            u_norm = np.sqrt(u_x**2 + u_y**2 + u_z**2)
            u_x = u_x / u_norm
            u_y = u_y / u_norm
            u_z = u_z / u_norm

            # 3. Find the discriminant of the quadratic equation: disc = (u dot O)^2 - |O|^2 + r^2
            # where r is the radius of the sphere
            disc = (u_x * fx + u_y * fy + u_z * fz) ** 2 - (
                fx**2 + fy**2 + fz**2 - self.radius**2
            )
            sol_mask = disc >= 0
            # We can use disc < 0 later to mask this properly

            # 4. Find the distance to the sphere. We use the negative root to get the point closer to the sphere
            disc[~sol_mask] = 0
            t = -(u_x * fx + u_y * fy + u_z * fz) - np.sqrt(disc)
            t[~sol_mask] = 0

            # 5. Find the surface coordinates
            surface_x = fx + t * u_x
            surface_y = fy + t * u_y
            surface_z = fz + t * u_z

            # 6. Reflect the z and y coordinates about the x axis
            # since the projection through the focal point is inverted
            surface_z = -surface_z
            surface_y = -surface_y

            # 7. Set invalid surface coords to NaN
            surface_x[~sol_mask] = np.nan
            surface_y[~sol_mask] = np.nan
            surface_z[~sol_mask] = np.nan

        elif self.method == "surface-straight":
            zoom /= self.radius / 32 / (2 * np.pi)
            x = np.ones((tile_size, tile_size)) * self.view_distance(zoom)
            y_range = (
                np.linspace(-h_s, h_s - 1, tile_size) / zoom
            )  # Maybe should be 0.5 on either side
            y = np.outer(np.ones(tile_size), y_range)
            z_range = np.linspace(h_s - 1, -h_s, tile_size) / zoom
            z = np.outer(z_range, np.ones(tile_size))

            fx = np.ones((tile_size, tile_size)) * self.radius
            fy = y
            fz = z

            # We want to find the sphere intercept of lines from a reference frame through many focal points

            # O: origins of the lines
            # u: Direction of the lines

            # 1. Find direction vectors for the lines from the reference frame to the focal points
            u_x = fx - x
            u_y = fy - y
            u_z = fz - z

            # 2. Normalize the direction vectors (TODO optimise by just dividing by x distance)
            u_norm = np.sqrt(u_x**2 + u_y**2 + u_z**2)
            u_x = u_x / u_norm
            u_y = u_y / u_norm
            u_z = u_z / u_norm

            # 3. Find the discriminant of the quadratic equation: disc = (u dot O)^2 - |O|^2 + r^2

            disc = (u_x * fx + u_y * fy + u_z * fz) ** 2 - (
                fx**2 + fy**2 + fz**2 - self.radius**2
            )
            sol_mask = disc >= 0
            # We can use disc < 0 later to mask this properly

            # 4. Find the distance to the sphere. We use the negative root to get the intercept closest to the frame
            disc[~sol_mask] = 0
            t = -(u_x * fx + u_y * fy + u_z * fz) - np.sqrt(disc)
            t[~sol_mask] = 0

            # 5. Find the surface coordinates
            surface_x = fx + t * u_x
            surface_y = fy + t * u_y
            surface_z = fz + t * u_z

            # 6. Set invalid surface coords to NaN
            surface_x[~sol_mask] = np.nan
            surface_y[~sol_mask] = np.nan
            surface_z[~sol_mask] = np.nan

        # +lat is north, -lat is south, +long is east, -long is west
        # +z is north, -z is south, +y is east, -y is west
        (r_lat, r_long) = coord
        r_lat = math.radians(-r_lat)
        r_long = math.radians(r_long)

        # Rotate the grid about the y axis
        r_y_surface_x = surface_x * np.cos(r_lat) + surface_z * np.sin(r_lat)
        r_y_surface_z = -surface_x * np.sin(r_lat) + surface_z * np.cos(r_lat)
        # r_y_surface_y = surface_y

        # Rotate the grid about the z axis
        r_z_surface_x = r_y_surface_x * np.cos(r_long) - surface_y * np.sin(r_long)
        r_z_surface_y = r_y_surface_x * np.sin(r_long) + surface_y * np.cos(r_long)
        r_z_surface_z = r_y_surface_z

        return r_z_surface_x, r_z_surface_y, r_z_surface_z

    def surface_coords_to_coords(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> LatLongCoords:
        """
        Convert 3D surface coordinates to WGS84 coordinates.

       :param x: A numpy array with the x coordinates on the surface.
       :param y: A numpy array with the y coordinates on the surface.
       :param z: A numpy array with the z coordinates on the surface.

       :return: A tuple with the latitudes and longitudes.
        """
        # Normalize the 3D cartesian coordinates
        mag = np.sqrt(x**2 + y**2 + z**2)
        x = x / mag
        y = y / mag
        z = z / mag
        del mag

        # Convert the 3D cartesian coordinates to latitudes and longitudes
        lat = np.arcsin(z)
        long = np.arctan2(y, x)
        # Then convert the radians to degrees
        lat = np.degrees(lat)
        long = np.degrees(long)
        return lat, long

    def atlas_coords_to_surface_coords(
        self, atlas_y: np.ndarray, atlas_x: np.ndarray
    ) -> SurfaceCoords:
        """
        Convert WGS84 atlas coordinates to 3D surface coordinates.

       :param atlas_y: A numpy array with the y coordinates in the atlas.
       :param atlas_x: A numpy array with the x coordinates in the atlas.

       :return: A tuple with the x, y and z coordinates on the surface.
        """

        # Convert the surface coordinates to lat and long
        lat = (atlas_y - 0.5) / self.pixel_resolution
        long = (atlas_x + 0.5) / self.pixel_resolution
        lat = 90 - lat
        long = long - 180
        lat = np.deg2rad(lat)
        long = np.deg2rad(long)

        # Then convert the radians to 3D cartesian coordinates
        x = self.radius * np.cos(lat) * np.cos(long)
        y = self.radius * np.cos(lat) * np.sin(long)
        z = self.radius * np.sin(lat)

        return x, y, z

    def view_distance(self, zoom: float) -> float:
        """
        This function returns the distance from the surface to the view point.
        """
        return self.radius / (min(zoom, 0.9999999))

    def click_coords(
        self,
        click_coord: tuple[int, int],
        center_coord: tuple[float, float],
        tile_size: int = 256,
        zoom: float = 1.0,
        surface_coords: SurfaceCoords | None = None,
    ) -> LatLongCoords:
        """
        This function takes a click on the screen and returns the WGS84 coordinates.
        """
        click_x, click_y = click_coord
        if surface_coords is None:
            surface_x, surface_y, surface_z = self.get_surface_coords(
                center_coord, tile_size, zoom
            )
        else:
            surface_x, surface_y, surface_z = surface_coords
        h, w = surface_x.shape
        if click_x < 0 or click_x >= w or click_y < 0 or click_y >= h:
            return np.array([np.nan]), np.array([np.nan])
        click_s_x = surface_x[click_y, click_x]
        click_s_y = surface_y[click_y, click_x]
        click_s_z = surface_z[click_y, click_x]

        click_lat, click_long = self.surface_coords_to_coords(
            click_s_x, click_s_y, click_s_z
        )
        return click_lat, click_long

    @profile
    def get_tile_mapping(
        self,
        coord: tuple[float, float],
        tile_size: int = 256,
        zoom: float = 1.0,
        round_result: bool = True,
        surface_coords: SurfaceCoords | None = None,
    ) -> AtlasCoords:
        """
        This function takes a WGS84 coordinate and returns the tile to atlas mapping (one to one).
       :param coord: A tuple with the latitude and longitude in degrees.
       :param tile_size: The width and height of the tile.
       :param zoom: The zoom level of the tile.
       :param round_result: Whether to round the mapping to the nearest integer.

       :return: A tuple with the y and x coordinates of the tile in the atlas.
        """

        # In our coordinate system, z is up, y is right and x is forward
        # (towards the screen if (0,0) is center)
        if surface_coords is None:
            surface_x, surface_y, surface_z = self.get_surface_coords(
                coord, tile_size, zoom
            )
        else:
            surface_x, surface_y, surface_z = surface_coords
        # TODO Fix rectangular tiles

        # Convert the surface coordinates to lat and long
        lat = np.arcsin(surface_z / self.radius)
        long = np.arctan2(surface_y, surface_x)

        # Then convert the radians to degrees
        lat = np.degrees(lat)
        long = np.degrees(long)
        lat = 90 - lat
        long = long + 180

        # Then convert the degrees to pixel coordinates
        atlas_y = lat * self.pixel_resolution
        atlas_x = long * self.pixel_resolution

        if round_result:
            atlas_y = np.round(atlas_y).astype(np.int32).clip(0, self.height - 1)
            atlas_x = np.round(atlas_x).astype(np.int32).clip(0, self.width - 1)
        return atlas_y, atlas_x

    def tile_mapping_to_coords(
        self, y: np.ndarray, x: np.ndarray
    ) -> LatLongCoords:
        """
        Convert the tile mapping to WGS84 coordinates.
        """
        lat = 90 - y / self.pixel_resolution
        long = x / self.pixel_resolution - 180
        return lat, long

    def get_full_tile_mapping(
        self, coord: tuple[float, float], tile_size: int = 256
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        This function takes a WGS84 coordinate and returns the full tile to atlas mapping (one to many).

       :param coord: A tuple with the latitude and longitude in degrees.

       :param width: The width of the tile.

       :param height: The height of the tile.

       :param round_mapping: Whether to round the mapping to the nearest integer.

       :return: A tuple with the y and x coordinates of the tile in
       the atlas and the corresponding y and x coordinates in the tile.
        """
        atlas_y, atlas_x = self.get_tile_mapping(coord, tile_size)

        # Naive interpolation

        # 1. Get the bounding box of the affected region
        max_y = atlas_y.max()
        min_y = atlas_y.min()
        max_x = atlas_x.max()
        min_x = atlas_x.min()

        # 2. In this range, translate the pixel coordinates to latitudes and longitudes

        y_range = np.arange(min_y, max_y + 1)
        x_range = np.arange(min_x, max_x + 1)
        y_range, x_range = np.meshgrid(y_range, x_range)
        # TODO Check if this offset is correct
        lat = (y_range - 0.5) / self.pixel_resolution
        long = (x_range + 0.5) / self.pixel_resolution
        lat = 90 - lat
        long = long - 180

        lat = np.deg2rad(lat)
        long = np.deg2rad(long)

        # 3. Translate these latitudes and longitudes to 3D cartesian coordinates
        # z is up, y is right and x is forward
        x = self.radius * np.cos(lat) * np.cos(long)
        y = self.radius * np.cos(lat) * np.sin(long)
        z = self.radius * np.sin(lat)

        # 4. Rotate these 3D cartesian coordinates by the reverse of the coord
        r_lat, r_long = coord
        r_lat = -math.radians(-r_lat)
        r_long = -math.radians(r_long)

        # Rotate about z axis by r_long
        r_z_x = x * np.cos(r_long) - y * np.sin(r_long)
        r_z_y = x * np.sin(r_long) + y * np.cos(r_long)
        r_z_z = z

        # Rotate about y axis by r_lat
        r_y_x = r_z_x * np.cos(r_lat) + r_z_z * np.sin(r_lat)
        r_y_y = r_z_y
        r_y_z = r_z_x * np.sin(r_lat) - r_z_z * np.cos(r_lat)

        # 5. Project these 3D cartesian coordinates to the pixel coordinates at x = r

        if self.method == "surface-to-center":
            # Find the intercept of the vectors with the plane x = r
            r = self.radius
            t = r / r_y_x
            x[:] = r
            y = r_y_y * t
            z = r_y_z * t
        elif self.method == "pinhole":
            raise NotImplementedError
        elif self.method == "surface-straight":
            # Keep y and z the same, scale x to intercept the plane
            r = self.radius
            x[:] = r
            y = r_y_y
            z = r_y_z

        # 6. Round these pixel coordinates to the nearest integer to get the coordinates in the tile

        image_x = np.round(y).astype(np.int32) + tile_size // 2
        image_y = np.round(z).astype(np.int32) + tile_size // 2
        valid = (
            (image_x >= 0)
            & (image_x < tile_size)
            & (image_y >= 0)
            & (image_y < tile_size)
        )

        # 7. Set the pixels in the bounding box to the corresponding pixels in the tile
        atlas_y = y_range[valid]
        atlas_x = x_range[valid]
        image_y = image_y[valid]
        image_x = image_x[valid]

        return atlas_y, atlas_x, image_y, image_x

    @profile
    def get_tile(
        self,
        coord: tuple,
        rotation: float = 0,
        tile_size: int = 256,
        atlas: np.ndarray = None,
        atlas_indices: AtlasCoords = None,
        discrete: bool = False,
        zoom: float = 1.0,
        nan_value: int | float = 0,
        custom_rotate_func: Callable[[np.ndarray, float], np.ndarray] | None = None,
    ) -> np.ndarray:
        """
        This function takes a WGS84 coordinate and rotation about the coordinate and returns a 2D tile.
        """
        nan_mask = np.zeros((tile_size, tile_size), dtype=bool)
        if atlas_indices is not None and (
            not np.issubdtype(atlas_indices[0].dtype, np.integer)
            or not np.issubdtype(atlas_indices[1].dtype, np.integer)
        ):
            nan_mask = np.isnan(atlas_indices[0]) | np.isnan(atlas_indices[1])
            atlas_y, atlas_x = atlas_indices
            atlas_y[nan_mask] = 0
            atlas_x[nan_mask] = 0
            if discrete:
                atlas_y = atlas_y.astype(np.int32)
                atlas_x = atlas_x.astype(np.int32)
            atlas_indices = (atlas_y, atlas_x)
        discrete = discrete or self.discrete
        atlas = self.atlas if atlas is None else atlas
        if rotation == 0:
            if atlas_indices is None:
                atlas_y, atlas_x = self.get_tile_mapping(
                    coord, tile_size, zoom, round_result=discrete
                )
            else:
                atlas_y, atlas_x = atlas_indices
            h, w = atlas.shape[:2]
            tile = np.zeros((tile_size, tile_size, self.channels), dtype=atlas.dtype)
            if discrete or np.issubdtype(atlas_y.dtype, np.integer):
                tile = atlas[atlas_y.round().astype(np.int32).clip(0, h - 1), atlas_x.round().astype(np.int32) % w]
            else:
                tile = bilinear_sample(atlas, atlas_y, atlas_x, self.sampling_mode)
        else:
            # TODO Optimise this (using max width for 45 degree rotation is ~2x slower)
            max_size = math.ceil(tile_size * math.sqrt(2))
            if atlas_indices is None:
                atlas_y, atlas_x = self.get_tile_mapping(
                    coord, max_size, zoom, round_result=discrete
                )
            else:
                atlas_y, atlas_x = atlas_indices
            tile = np.zeros((max_size, max_size, self.channels), dtype=atlas.dtype)
            atlas = self.atlas if atlas is None else atlas
            h, w = atlas.shape[:2]
            if discrete or np.issubdtype(atlas_y.dtype, np.integer):
                tile = atlas[atlas_y.round().astype(np.int32).clip(0, h - 1), atlas_x.round().astype(np.int32) % w]
            else:
                tile = bilinear_sample(atlas, atlas_y, atlas_x)
            # Rotate the tile
            if custom_rotate_func is None:
                center = (max_size / 2, max_size / 2)
                rot_mat = cv2.getRotationMatrix2D(center, rotation, 1.0)
                extra_kwargs = {"flags": cv2.INTER_NEAREST} if discrete else {"flags": cv2.INTER_CUBIC}
                tile = cv2.warpAffine(tile, rot_mat, (max_size, max_size), **extra_kwargs)
            else:
                tile = custom_rotate_func(tile, rotation)
            # Crop the tile
            top = max_size // 2 - tile_size // 2
            left = max_size // 2 - tile_size // 2
            tile = tile[top: top + tile_size, left: left + tile_size]

        if nan_mask.any():
            # There is a bug here when passing in a rotation > 0
            # and atlas indices that already account for the rotation max size
            # and you have nan values
            try:
                tile[nan_mask] = nan_value
            except Exception as e:
                logging.warning(f"Couldn't replace nan values due to: {e}")
        return tile

    def set_tile(
        self, coord: tuple, tile: np.ndarray, atlas: np.ndarray | None = None
    ) -> np.ndarray:
        """
        This function takes a WGS84 coordinate and rotation about the coordinate and sets a 2D tile.
       :param coord: A tuple with the latitude and longitude in degrees.
       :param tile: A 2D numpy array with the tile.
       :param atlas: A 2D numpy array with the atlas.

       :return: A 2D numpy array with the updated atlas.
        """
        # TODO Optimize somehow
        if atlas is None:
            atlas = self.atlas
        atlas_y, atlas_x, image_y, image_x = self.get_full_tile_mapping(
            coord, tile.shape[0]
        )
        atlas[atlas_y, atlas_x] = tile[image_y, image_x]
        return atlas

    def get_distributed_points(
        self,
        mask: np.ndarray,
        n: int | None = None,
        offset: tuple[float, float] = (0, 0),
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        This function returns a list of n points that are distributed on the sphere.
       :param mask: A 2D numpy array with the mask of the sphere.
       :param n: The number of points to distribute on the sphere.
       :param offset: A tuple with the latitude and longitude in degrees to offset the points.

       :return: A tuple with the latitude and longitude of the points.
        """

        if n == 0:
            return np.array([]), np.array([])

        if n is None:
            # Use surface area of the sphere to determine the number of points
            mask_radius = mask.shape[1] / (2 * np.pi)
            n = math.ceil(4 * np.pi * mask_radius**2)

        # Use fibonacci lattice points
        # https://arxiv.org/pdf/0912.4540
        # https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
        phi = math.pi * (math.sqrt(5) - 1.0)
        indices = np.arange(0, n, dtype=int)
        y = 1 - (indices / float(n - 1)) * 2
        radius = np.sqrt(1 - y * y)
        theta = phi * indices
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        # Convert the surface coordinates to lat and long
        lat = np.arcsin(z)
        long = np.arctan2(y, x)
        # Then convert the radians to degrees
        lat = np.degrees(lat)
        long = np.degrees(long)
        lat += offset[0]
        long += offset[1]
        lat = 90 - lat
        long = (long + 180) % 360
        # TODO Potentially deal with lat > 90 and lat < -90
        # Then convert the degrees to pixel coordinates
        mask_res = mask.shape[1] / 360
        mask_h = mask.shape[0]
        mask_w = mask.shape[1]
        atlas_y = np.round(lat * mask_res).astype(np.int32).clip(0, mask_h - 1)
        atlas_x = np.round(long * mask_res).astype(np.int32).clip(0, mask_w - 1)
        # Remove the points from lat/long that are not on the mask
        mask_points = mask[atlas_y, atlas_x]
        lat = lat[mask_points > 0]
        long = long[mask_points > 0]
        return 90 - lat, long - 180

    def generate_noise(
        self,
        settings: NoiseSettings,
        amplitude_mask: np.ndarray | None = None,
        coord_offset: tuple[float, float] | None = None,
    ) -> Annotated[npt.NDArray[np.float64], Literal["h", "w"]]:
        """
        Use opensimplex to generate noise on the sphere.
       :param settings: A NoiseSettings object with the settings for the noise.
       :param amplitude_mask: A 2D numpy array with the amplitude mask for the noise.
       This should already have mean filters applied.
        """
        total_noise = np.zeros((self.height, self.width))

        # 1. Convert latitude and longitude to 3D cartesian coordinates
        # z is up, y is right and x is forward
        xs = np.arange(self.width)
        ys = np.arange(self.height)
        xs, ys = np.meshgrid(xs, ys)
        lats = (ys - 0.5) / self.pixel_resolution
        longs = (xs + 0.5) / self.pixel_resolution
        lats = 90 - lats
        longs = longs - 180
        lats = np.deg2rad(lats)
        longs = np.deg2rad(longs)
        if coord_offset is not None:
            lat, long = coord_offset
            lat = math.radians(lat)
            long = math.radians(long)
            lats = lats + lat
            longs = longs + long
        x = self.radius * np.cos(lats) * np.cos(longs)
        y = self.radius * np.cos(lats) * np.sin(longs)
        z = self.radius * np.sin(lats)

        total_noise = stacked_noise(x, y, z, settings, amplitude_mask)
        return total_noise

        # if threshold_mask is not None:
        #     gradient_map = rank.mean((threshold_mask*255).astype(np.uint8), disk(settings.erode_iters))/255
        #     threshold = gradient_map * threshold
        #     persistence = (gradient_map+1) * persistence

        # Use the rough mask as the first layer
        # if amplitude_mask is not None:
        #     stacked = np.dstack((x, y, z)) * freq
        #     noise = noise3coords(stacked, settings.seed)
        #     total_noise += (noise+1)*0.5*amplitude_mask
        #     amplitude *= persistence
        #     freq *= freq_mult
        #     if octaves > 1:
        #         octaves -= 1
        # if amplitude_mask is not None:
        #     amplitude = amplitude_mask

        # Potentially add weight

    def add_line(
        self,
        src_cart_coords: tuple[float, float],
        dest_cart_coords: tuple[float, float],
        thickness: int = 1,
        color: tuple[int, int, int] | int = (255, 255, 255),
        atlas: np.ndarray = None,
    ) -> np.ndarray:
        """
        This function adds a line to the atlas.
        """
        update_atlas = atlas is None
        if atlas is None:
            atlas = self.atlas
        h, w = atlas.shape[:2]
        channels = atlas.shape[2] if len(atlas.shape) > 2 else 1
        if channels == 1 and isinstance(color, tuple):
            color = int(color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114)
        elif channels == 1:
            color = int(color)
        src_lat, src_long = src_cart_coords
        dest_lat, dest_long = dest_cart_coords
        src_y, src_x = self.get_tile_mapping((src_lat, src_long), tile_size=1)
        dest_y, dest_x = self.get_tile_mapping((dest_lat, dest_long), tile_size=1)
        src_x = int(src_x)
        src_y = int(src_y)
        dest_x = int(dest_x)
        dest_y = int(dest_y)

        if src_x > dest_x:
            src_x, dest_x = dest_x, src_x
            src_y, dest_y = dest_y, src_y

        # Wraps around axis so make sure to draw lines to edges first
        if dest_x - src_x > w // 2:
            mid_x_0 = 0
            mid_x_1 = w - 1
            mid_y_0 = int(
                (dest_y - src_y) / (dest_x - src_x) * (mid_x_0 - src_x) + src_y
            )
            mid_y_1 = int(
                (dest_y - src_y) / (dest_x - src_x) * (mid_x_1 - src_x) + src_y
            )
            mid_y_0 = mid_y_0 % h
            mid_y_1 = mid_y_1 % h

            atlas = cv2.line(
                atlas, (src_x, src_y), (mid_x_0, mid_y_0), color, thickness
            )

            atlas = cv2.line(
                atlas, (dest_x, dest_y), (mid_x_1, mid_y_1), color, thickness
            )
        else:
            atlas = cv2.line(atlas, (src_x, src_y), (dest_x, dest_y), color, thickness)
        if update_atlas:
            self.atlas = atlas
        return atlas


class QuadSphere(
    SphereMapping
):  # TODO This should not actually be a subclass of SphereMapping
    def __init__(
        self,
        atlas: np.ndarray | None = None,
        shape: tuple[int, int] | None = None,
        discrete: bool = False,
        sampling_mode: SampleMode = "mean",
    ):
        super().__init__(atlas, shape, method="surface-to-center", discrete=discrete)
        self.level = int(math.ceil(math.log2(self.width / 4)))
        self.face_width = self.int_diameter // 2 * 2
        self.quad_sphere_atlas = None
        self.face_centers = [(90, 0), (0, 0), (0, 90), (0, 180), (0, -90), (-90, 0)]
        self.sampling_mode = sampling_mode
        # Test coords

        cartesian_coords = (
            np.array(
                [[0, 0, 1], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, -1]]
            )
            * self.radius
        )
        for face, coord in enumerate(self.face_centers):
            x, y, z = self.get_surface_coords(coord, 1)
            carts = np.array([x, y, z])[:, 0, 0]
            assert np.allclose(cartesian_coords[face], carts)
            u, v = self.surface_coords_to_quad_coords(x, y, z)
            expected_u = self.face_width // 2
            expected_v = face * self.face_width + self.face_width // 2
            assert u.round() == expected_u
            assert v.round() == expected_v

            new_x, new_y, new_z = self.quad_coords_to_surface_coords(u, v)
            assert np.allclose(
                carts / self.radius, np.array([new_x, new_y, new_z])[:, 0, 0]
            )

            lat, long = self.surface_coords_to_coords(x, y, z)
            assert np.allclose(np.array([lat, long])[:, 0, 0], np.array(coord))

        self.quad_sphere_atlas: np.ndarray | None = None
        if self.atlas is not None:
            self.quad_sphere_atlas = self.get_quad_sphere_atlas()

    def set_quad_atlas(self, quad_atlas: np.ndarray) -> None:
        self.quad_sphere_atlas = quad_atlas
        h, w = quad_atlas.shape[:2]
        self.face_width = h
        assert w == 6 * h, "The width of the quad atlas must be 6 times the height."
        # Formulae:
        # self.radius = self.width / (2 * np.pi)
        # self.face_width = int(math.ceil(2*self.radius)) // 2 * 2
        self.radius = h / 2
        width = int(2 ** round(math.log2(self.radius * 2 * np.pi)))
        height = width // 2
        self.set_shape((height, width))

    @property
    def quad_shape(self) -> tuple[int, int]:
        return self.face_width, 6 * self.face_width

    @profile
    def get_quad_sphere_atlas(self, atlas: np.ndarray | None = None) -> np.ndarray:
        """
        This function uses a WGS84 projected 2D atlas and maps it to a quad-sphere atlas.

       :return: A 6 face 2D numpy array with the quad-sphere atlas.
        """
        atlas = atlas if atlas is not None else self.atlas
        channels = atlas.shape[2] if len(atlas.shape) > 2 else 1
        assert atlas is not None, "Atlas must be provided."
        if self.quad_sphere_atlas is not None:
            return self.quad_sphere_atlas
        shape = (
            (self.face_width, 6 * self.face_width, channels)
            if channels > 1
            else (self.face_width, 6 * self.face_width)
        )
        quad_sphere_atlas = np.zeros(shape, dtype=atlas.dtype)
        for i, center in enumerate(self.face_centers):
            face = self.get_tile(center, tile_size=self.face_width, atlas=atlas)
            quad_sphere_atlas[:, i * self.face_width: (i + 1) * self.face_width] = face
        # for i, center in enumerate(self.face_centers):
        #     new_face = self.get_quad_tile(center, tile_size=self.face_width, atlas=quad_sphere_atlas)
        #     quad_face = quad_sphere_atlas[:, i*self.face_width:(i+1)*self.face_width]
        #     side_by_side = np.hstack((quad_face, new_face))
        #     pass

        return quad_sphere_atlas

    def get_normal_atlas(self, quad_atlas: np.ndarray | None = None) -> np.ndarray:
        """
        This function takes a quad-sphere atlas and returns a normal atlas.
        """
        if quad_atlas is None:
            normal_atlas = np.zeros(
                (self.height, self.width, self.channels), dtype=np.uint8
            )
            quad_atlas = self.quad_sphere_atlas
        else:
            h, w = self.height, self.width
            channels = quad_atlas.shape[2] if len(quad_atlas.shape) > 2 else 1
            shape = (h, w, channels) if channels > 1 else (h, w)
            normal_atlas = np.zeros(shape, dtype=np.uint8)
        # for i, center in enumerate(self.face_centers):
        #     face = self.quad_sphere_atlas[:, i*self.face_width:(i+1)*self.face_width]
        #     atlas_y, atlas_x = self.get_tile_mapping(center, self.face_width)
        #     normal_atlas[atlas_y, atlas_x] = face
        #     # normal_atlas = self.set_tile(center, face, normal_atlas)

        atlas_y = np.arange(self.height)
        atlas_x = np.arange(self.width)
        atlas_x, atlas_y = np.meshgrid(atlas_x, atlas_y)

        x, y, z = self.atlas_coords_to_surface_coords(atlas_y, atlas_x)
        u, v = self.surface_coords_to_quad_coords(x, y, z)

        u = u.round().astype(np.int32)
        v = v.round().astype(np.int32)

        # TODO: Add bilinear interpolation

        normal_atlas = quad_atlas[u, v]

        return normal_atlas

    def quad_coords_to_surface_coords(
        self, u: np.ndarray, v: np.ndarray
    ) -> SurfaceCoords:
        """
        Convert quad UV coordinates to 3D surface coordinates.

       :param u: A numpy array with the u coordinates on the quad_sphere_atlas.
       :param v: A numpy array with the v coordinates on the quad_sphere_atlas.

       :return: A tuple with the x, y and z coordinates on the surface.
        """
        # 1. Find the face number and select plane for projection

        face_num = v // self.face_width
        v = v % self.face_width
        u = u / self.face_width * 2 - 1
        v = v / self.face_width * 2 - 1
        assert np.all(u >= -1) and np.all(u <= 1)
        assert np.all(v >= -1) and np.all(v <= 1)

        f_0 = face_num == 0
        f_5 = face_num == 5
        f_1 = face_num == 1
        f_3 = face_num == 3
        f_2 = face_num == 2
        f_4 = face_num == 4

        # 2. Find the cube coordinates
        x = np.zeros_like(u, dtype=np.float32)
        y = np.zeros_like(u, dtype=np.float32)
        z = np.zeros_like(u, dtype=np.float32)

        x[f_0] = u[f_0]
        y[f_0] = v[f_0]
        z[f_0] = 1

        x[f_5] = -u[f_5]
        y[f_5] = v[f_5]
        z[f_5] = -1

        x[f_1] = 1
        y[f_1] = v[f_1]
        z[f_1] = -u[f_1]

        x[f_3] = -1
        y[f_3] = -v[f_3]
        z[f_3] = -u[f_3]

        x[f_2] = -v[f_2]
        y[f_2] = 1
        z[f_2] = -u[f_2]

        x[f_4] = v[f_4]
        y[f_4] = -1
        z[f_4] = -u[f_4]

        # 3. Project onto sphere
        mag = np.sqrt(x**2 + y**2 + z**2)
        x = x / mag
        y = y / mag
        z = z / mag
        del mag

        return x, y, z

    def surface_coords_to_quad_coords(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        preserve_nan: bool = False,
        round: bool = False,
    ) -> QuadCoords:
        """
        Convert 3D surface coordinates to the quad UV coordinates.

       :param x: A numpy array with the x coordinates on the surface.
       :param y: A numpy array with the y coordinates on the surface.
       :param z: A numpy array with the z coordinates on the surface.
       :param preserve_nan: Whether to preserve NaN values in the output.

       :return: A tuple with the u and v coordinates on the quad_sphere_atlas.
        """
        # Note this mutates the input arrays

        # 1. Find the face number and select plane for projection

        f_0_5 = (np.abs(z) >= np.abs(x)) & (np.abs(z) >= np.abs(y))
        f_0 = (z >= 0) & f_0_5
        f_5 = (z < 0) & f_0_5
        f_1_3 = (np.abs(x) >= np.abs(y)) & ~f_0_5
        f_1 = (x >= 0) & f_1_3
        f_3 = (x < 0) & f_1_3
        f_2_4 = ~f_0_5 & ~f_1_3
        f_2 = (y >= 0) & f_2_4
        f_4 = (y < 0) & f_2_4

        face_num = np.zeros_like(x, dtype=np.uint8)
        face_num[f_0] = 0
        face_num[f_1] = 1
        face_num[f_2] = 2
        face_num[f_3] = 3
        face_num[f_4] = 4
        face_num[f_5] = 5

        # 2. Find parametric equation for each surface_coord
        mag = np.sqrt(x**2 + y**2 + z**2)
        x = x / mag
        y = y / mag
        z = z / mag
        del mag

        # 3. Find projection distance

        face_width = self.face_width
        h_f_w = face_width / 2
        t = np.zeros_like(x, dtype=np.float32)
        t[f_0] = h_f_w / z[f_0]
        t[f_5] = h_f_w / -z[f_5]
        t[f_1] = h_f_w / x[f_1]
        t[f_3] = h_f_w / -x[f_3]
        t[f_2] = h_f_w / y[f_2]
        t[f_4] = h_f_w / -y[f_4]

        # 4. Project onto plane

        x = x * t
        y = y * t
        z = z * t
        del t

        # 5. Find u and v coordinates
        # u is height, v is width

        u = np.zeros_like(x, dtype=np.float32)
        v = np.zeros_like(x, dtype=np.float32)
        if preserve_nan:
            u[u == 0] = np.nan
            v[v == 0] = np.nan

        u[f_0] = x[f_0] + h_f_w
        v[f_0] = y[f_0] + h_f_w
        u[f_5] = -x[f_5] + h_f_w
        v[f_5] = y[f_5] + h_f_w

        u[f_1] = -z[f_1] + h_f_w
        v[f_1] = y[f_1] + h_f_w
        u[f_3] = -z[f_3] + h_f_w
        v[f_3] = -y[f_3] + h_f_w

        u[f_2] = -z[f_2] + h_f_w
        v[f_2] = -x[f_2] + h_f_w
        u[f_4] = -z[f_4] + h_f_w
        v[f_4] = x[f_4] + h_f_w

        # 6. Clip the u and v coordinates
        u = np.clip(u, 0, face_width - 1)
        v = np.clip(v, 0, face_width - 1)
        # 7. Add offset to v coordinates
        v += face_num * float(face_width)
        if round:
            u = np.round(u).astype(np.int32)
            v = np.round(v).astype(np.int32)
            v = v % (6 * face_width)
        return u, v

    def get_naive_quad_tile_mapping(
        self, coords: tuple[float, float], tile_size: int = 256, round: bool = True
    ) -> QuadCoords:
        """
        This function takes a WGS84 coordinate and returns the tile to quad-sphere atlas mapping (one to one).
       :param coord: A tuple with the latitude and longitude in degrees.
       :param tile_size: The width and height of the tile.
       :param round: Whether to round the mapping to the nearest integer.

       :return: A tuple with the y and x coordinates of the tile in the quad-sphere atlas.
        """

        # This is the naive approach which may not have contiguous u, v coordinates
        x, y, z = self.get_surface_coords(coords, tile_size)
        u, v = self.surface_coords_to_quad_coords(x, y, z)
        if round:
            u = np.round(u).astype(np.int32).clip(0, self.face_width - 1)
            v = np.round(v).astype(np.int32).clip(0, 6 * self.face_width - 1)
        return u, v

    def get_nearest_valid_coord(
        self, coord: tuple[float, float], tile_size: int = 256
    ) -> tuple[float, float]:
        """
        This function takes a WGS84 coordinate and returns the nearest valid coordinate.
        This is to prevent encountering errors when calling get_quad_tile_mapping.
        """
        x, y, z = self.get_surface_coords(coord, 1)
        u, v = self.surface_coords_to_quad_coords(x, y, z)
        h_t_s = tile_size // 2

        # 1. Find source face
        u_i = int(u.round()[0])
        v_i = int(v.round()[0])
        source_face = v_i // self.face_width
        v_i %= self.face_width
        side = source_face not in [0, 5]

        # Subtract half tile size to make u_i and v_i the top left corner of the tile.
        # This is easier to work with than the center of the tile.
        u_i -= h_t_s
        v_i -= h_t_s

        u_min = u_i
        u_max = u_i + tile_size
        v_min = v_i
        v_max = v_i + tile_size

        u_out = u_min < 0 or u_max >= self.face_width
        v_out = v_min < 0 or v_max >= self.face_width

        # These cases are valid so we can return the original coordinates
        if side and not u_out:
            return coord
        elif side and u_out:
            return coord
        elif not side and not u_out and not v_out:
            return coord

        if u_min < 0:
            u_i = 0
        elif u_max >= self.face_width:
            u_i = self.face_width - tile_size
        if v_min < 0:
            v_i = 0
        elif v_max >= self.face_width:
            v_i = self.face_width - tile_size

        u_i += h_t_s
        v_i += h_t_s

        x, y, z = self.quad_coords_to_surface_coords(np.array([u_i]), np.array([v_i]))
        lat, long = self.surface_coords_to_coords(x, y, z)
        return float(lat), float(long)

    def get_3x4_quad_atlas(
        self, atlas: np.ndarray | None = None, corner_crop: bool = False
    ) -> np.ndarray:
        """
        This function takes a quad-sphere atlas and returns a 3x4 atlas.
        """
        if atlas is None:
            atlas = self.quad_sphere_atlas
        channels = atlas.shape[2] if len(atlas.shape) > 2 else 1
        face_width = atlas.shape[1] // 6
        shape = (
            (face_width * 3, face_width * 4, channels)
            if channels > 1
            else (face_width * 3, face_width * 4)
        )
        atlas_3x4 = np.zeros(shape, dtype=np.uint8)
        top_face = atlas[:, :face_width]
        bottom_face = atlas[:, -face_width:]
        for i in range(4):
            current_face = atlas[:, (i + 1) * face_width: (i + 2) * face_width]
            stacked = (
                np.vstack((top_face, current_face, bottom_face))
                if i == 1 else
                np.vstack((np.zeros_like(current_face), current_face, np.zeros_like(current_face)))
            )
            top_face = cv2.rotate(top_face, cv2.ROTATE_90_CLOCKWISE)
            bottom_face = cv2.rotate(bottom_face, cv2.ROTATE_90_COUNTERCLOCKWISE)
            atlas_3x4[:, i * face_width: (i + 1) * face_width] = stacked
        if corner_crop:
            for i in range(4):
                ys = np.arange(face_width)
                xs = np.arange(face_width)
                xs, ys = np.meshgrid(xs, ys)
                top_inv_tri = (ys <= face_width - xs) | (ys <= xs)
                bottom_inv_tri = (ys >= xs) | (ys >= face_width - xs)
                atlas_3x4[:face_width, i * face_width: (i + 1) * face_width][
                    top_inv_tri
                ] = 0
                atlas_3x4[-face_width:, i * face_width: (i + 1) * face_width][
                    bottom_inv_tri
                ] = 0

        return atlas_3x4

    @profile
    def get_quad_tile_mapping(
        self,
        coord: tuple[int, int] | None = None,
        tile_size: int = 256,
        round: bool = True,
        uv: tuple[int, int] | None = None,
        test_mode: bool = False,
    ) -> QuadCoords:
        """
        This function takes a WGS84 coordinate and returns the tile to quad-sphere atlas mapping (one to one).
       :param coord: A tuple with the latitude and longitude in degrees.
       :param tile_size: The width and height of the tile.
       :param round: Whether to round the mapping to the nearest integer.
       :param uv: A tuple with the u and v coordinates in the tile. Overrides the coord parameter.

       :return: A tuple with the y and x coordinates of the tile in the quad-sphere atlas.
        """

        if tile_size > self.face_width:
            raise NotImplementedError(
                "Tile size must be less than or equal to the face width."
            )

        if uv is None:
            assert coord is not None, "Either coord or uv must be provided."
            x, y, z = self.get_surface_coords(coord, 1)
            u, v = self.surface_coords_to_quad_coords(x, y, z)

            u_i = int(u.round()[0])
            v_i = int(v.round()[0])
        else:
            u_i, v_i = uv

        h_t_s = tile_size // 2

        # 1. Find source face
        source_face = v_i // self.face_width
        v_i %= self.face_width
        side = source_face not in [0, 5]

        # Subtract half tile size to make u_i and v_i the top left corner of the tile.
        # This is easier to work with than the center of the tile.
        u_i -= h_t_s
        v_i -= h_t_s

        u = np.arange(u_i, u_i + tile_size)
        v = np.arange(v_i, v_i + tile_size)
        # 2. Check if tile is on vertex (raise exception)
        u_out = u.min() < 0 or u.max() >= self.face_width
        v_out = v.min() < 0 or v.max() >= self.face_width

        if u_out and v_out:
            # TODO Do automatic offset since corner tiles are very tricky to generate
            # raise NotImplementedError("Tile is on vertex.")
            diffs = [
                u.min(),
                u.max() - self.face_width,
                v.min(),
                v.max() - self.face_width,
            ]
            min_diff = np.argmin(np.abs(diffs))
            if min_diff == 0:
                u_i = 0
            elif min_diff == 1:
                u_i = self.face_width - tile_size
            elif min_diff == 2:
                v_i = 0
            elif min_diff == 3:
                v_i = self.face_width - tile_size
            u = np.arange(u_i, u_i + tile_size)
            v = np.arange(v_i, v_i + tile_size)
            u_out = u.min() < 0 or u.max() >= self.face_width
            v_out = v.min() < 0 or v.max() >= self.face_width
            assert u_out or v_out, "Tile should be on edge after shift."

        v, u = np.meshgrid(v, u)

        if side and not u_out:
            # This is for side faces that don't overlap with the top or bottom faces
            v += self.face_width * (source_face - 1)
            v %= self.face_width * 4
            v += self.face_width
        elif side and u_out:
            # This is for side faces that do overlap with the top or bottom faces
            if u_i < 0:
                # Overlap with top face
                mask = u < 0
                out_u = u[mask]
                out_v = v[mask]
                v += self.face_width * (source_face - 1)
                v %= self.face_width * 4
                v += self.face_width
                out_u += self.face_width
                for _ in range(source_face - 1):
                    out_u, out_v = self.face_width - 1 - out_v, out_u
                u[mask] = out_u
                v[mask] = out_v
            elif u_i + tile_size >= self.face_width:
                # Overlap with bottom face
                mask = u >= self.face_width
                out_u = u[mask]
                out_v = v[mask]
                v += self.face_width * (source_face - 1)
                v %= self.face_width * 4
                v += self.face_width
                out_u -= self.face_width
                for _ in range(source_face - 1):
                    out_u, out_v = out_v, self.face_width - 1 - out_u
                out_v += 5 * self.face_width
                u[mask] = out_u
                v[mask] = out_v
        elif not side:
            # This is for top and bottom faces that do overlap with the side faces
            top_u = abs(u_i + h_t_s - self.face_width)
            bot_u = u_i + h_t_s
            top_v = abs(v_i + h_t_s - self.face_width)
            bot_v = v_i + h_t_s

            # 1. Get direction down, right, up, left
            direction = np.argmin([top_u, top_v, bot_u, bot_v])
            if source_face == 0:
                dest_face = [1, 2, 3, 4][direction]
                # rotations = 0, 1, 2, 3
                rotations = direction
            elif source_face == 5:
                dest_face = [3, 2, 1, 4][direction]
                # rotations = 2, 3, 0, 1
                rotations = (direction + 2) % 4

            mask = (u < 0) | (u >= self.face_width) | (v < 0) | (v >= self.face_width)

            in_f = ~mask
            out_f = mask

            # 2. Rotate u[out_f] and v[out_f] by 90 degrees * direction

            for _ in range(rotations):
                u[out_f], v[out_f] = v[out_f], self.face_width - 1 - u[out_f]

            u %= self.face_width
            v %= self.face_width
            v[in_f] += source_face * self.face_width
            v[out_f] += dest_face * self.face_width

            # 3. Rotate axes of u, v by 90 degrees * rotations

            for _ in range(4 - rotations):
                u = np.rot90(u)
                v = np.rot90(v)
            # Avoid tensor negative stride
            u = u.copy()
            v = v.copy()

        else:
            raise NotImplementedError

        if test_mode and self.quad_sphere_atlas is not None:
            atlas = self.quad_sphere_atlas.copy()
            atlas[:, :] = 0
            if len(atlas.shape) > 2:
                atlas = atlas[:, :, 0]
            atlas[u, v] = 255
            assert (atlas == 255).sum() == tile_size**2, "Tile must be one to one."
        return u, v

    @profile
    def get_quad_tile(
        self,
        coord: tuple,
        rotation: float = 0,
        tile_size: int = 256,
        atlas: np.ndarray = None,
        atlas_indices: QuadCoords = None,
        discrete: bool = False,
        zoom: float = 1.0,
        custom_rotate_func: Callable[[np.ndarray, float], np.ndarray] | None = None,
    ) -> np.ndarray:
        """
        This function takes a WGS84 coordinate and rotation about the coordinate and returns a 2D tile.
        """
        # TODO Add bilinear interpolation
        discrete = discrete or self.discrete
        if rotation == 0:
            if atlas_indices is None:
                atlas_y, atlas_x = self.get_quad_tile_mapping(coord, tile_size)
            else:
                atlas_y, atlas_x = atlas_indices
            tile = np.zeros((tile_size, tile_size, self.channels), dtype=np.uint8)
            atlas = self.quad_sphere_atlas if atlas is None else atlas
            tile = atlas[atlas_y, atlas_x]
        else:
            # TODO Optimise this (using max width for 45 degree rotation is ~2x slower)
            max_size = math.ceil(tile_size * math.sqrt(2))
            if atlas_indices is None:
                atlas_y, atlas_x = self.get_quad_tile_mapping(coord, max_size)
            else:
                atlas_y, atlas_x = atlas_indices
            tile = np.zeros((max_size, max_size, self.channels), dtype=np.uint8)
            atlas = self.quad_sphere_atlas if atlas is None else atlas
            # tile = bilinear_sample(atlas, atlas_y, atlas_x)
            tile = atlas[atlas_y, atlas_x]
            # Rotate the tile
            if custom_rotate_func is None:
                center = (max_size / 2, max_size / 2)
                rot_mat = cv2.getRotationMatrix2D(center, rotation, 1.0)
                extra_kwargs = {"flags": cv2.INTER_NEAREST} if discrete else {}
                tile = cv2.warpAffine(tile, rot_mat, (max_size, max_size), **extra_kwargs)
            else:
                tile = custom_rotate_func(tile, rotation)
            # Crop the tile
            top = max_size // 2 - tile_size // 2
            left = max_size // 2 - tile_size // 2
            tile = tile[top: top + tile_size, left: left + tile_size]

        return tile

    def set_quad_tile(
        self, coord: tuple, tile: np.ndarray, atlas: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Set a tile in the quad-sphere atlas.

       :param coord: A tuple with the latitude and longitude in degrees.
       :param tile: A 2D numpy array with the tile.
       :param atlas: A 2D numpy array with the quad-sphere atlas.

       :return: A 2D numpy array with the updated quad-sphere atlas.
        """

        u, v = self.get_quad_tile_mapping(coord, tile.shape[0])
        atlas = self.quad_sphere_atlas if atlas is None else atlas
        atlas[u, v] = tile
        return atlas

    def get_mapping(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> QuadCoords:
        """
        This function takes 3D unit-sphere cartesian coordinates and returns
        the pixel coordinates in the quad-sphere atlas.
       :param x: A numpy array with the x coordinates.
       :param y: A numpy array with the y coordinates.
       :param z: A numpy array with the z coordinates.

       :return: A tuple with the y and x coordinates of the tile in the quad-sphere atlas.
        TODO Fix this function: unimportant since actually just using the
        get_tile method works the same as in the original paper
        """

        # https://oceancolor.gsfc.nasa.gov/resources/browse_help/quadsphere/
        # 1. Determine face number and select coordinates for projection

        face_num = np.zeros_like(x, dtype=np.uint8)
        q = np.zeros_like(x, dtype=np.float32)
        r = np.zeros_like(x, dtype=np.float32)
        s = np.zeros_like(x, dtype=np.float32)
        f_0_5 = (np.abs(z) >= np.abs(x)) & (np.abs(z) >= np.abs(y))
        f_0 = (z >= 0) & f_0_5
        f_5 = (z < 0) & f_0_5
        f_1_3 = np.abs(x) >= np.abs(y)
        f_1 = (x >= 0) & f_1_3
        f_3 = (x < 0) & f_1_3
        f_2_4 = ~f_0_5 & ~f_1_3
        f_2 = (y >= 0) & f_2_4
        f_4 = (y < 0) & f_2_4

        assert np.array_equal(
            f_0 + f_1 + f_2 + f_3 + f_4 + f_5, np.ones_like(x, dtype=np.uint8)
        ), "Faces must be mutually exclusive."

        face_num[f_0] = 0
        face_num[f_1] = 1
        face_num[f_2] = 2
        face_num[f_3] = 3
        face_num[f_4] = 4
        face_num[f_5] = 5

        q[f_0] = z[f_0]
        r[f_0] = y[f_0]
        s[f_0] = -x[f_0]
        q[f_5] = -z[f_5]
        r[f_5] = y[f_5]
        s[f_5] = x[f_5]

        q[f_1] = x[f_1]
        r[f_1] = y[f_1]
        s[f_1] = z[f_1]
        q[f_3] = -x[f_3]
        r[f_3] = -y[f_3]
        s[f_3] = z[f_3]

        q[f_2] = y[f_2]
        r[f_2] = -x[f_2]
        s[f_2] = z[f_2]
        q[f_4] = -y[f_4]
        r[f_4] = x[f_4]
        s[f_4] = z[f_4]

        # 2. Compute face coordinates (u,v) from cartesian coordinates
        u = np.zeros_like(q, dtype=np.float64)
        v = np.zeros_like(q, dtype=np.float64)
        f = np.abs(r) >= np.abs(s)
        u[f] = np.sqrt((1 - q[f]) / (1 - 1 / np.sqrt(2 + (s[f] / r[f]) ** 2)))
        v[f] = (
            u[f]
            * (12 / np.pi)
            * (
                np.arctan(s[f] / np.abs(r[f]))
                - np.arcsin(s[f] / np.sqrt(2 * (r[f] ** 2 + s[f] ** 2)))
            )
        )
        u[f] = u[f] * np.sign(r[f])
        u[~f] = np.sqrt((1 - q[~f]) / (1 - 1 / np.sqrt(2 + (r[~f] / s[~f]) ** 2)))
        v[~f] = (
            u[~f]
            * (12 / np.pi)
            * (
                np.arctan(r[~f] / np.abs(s[~f]))
                - np.arcsin(r[~f] / np.sqrt(2 * (r[~f] ** 2 + s[~f] ** 2)))
            )
        )
        v[~f] = v[~f] * np.sign(s[~f])

        # 3. Compute pixel coordinates from face coordinates using level number

        lev = self.level
        u = (2**lev) * (u + 1) / 2
        v = (2**lev) * (v + 1) / 2

        u = np.clip(u, 0, 2**lev - 1)
        v = np.clip(v, 0, 2**lev - 1)

        u = u.astype(np.int32)
        v = v.astype(np.int32)

        # The faces are numbered 0-5 with 0 being the North face, 1 through 4 being equatorial with 1
        # corresponding to Greenwich, and 5 being South.

        u += 2**lev * face_num

        return v, u

    def add_line(
        self,
        src_cart_coords: tuple[float, float],
        dest_cart_coords: tuple[float, float],
        thickness: int = 1,
        color: tuple[int, int, int] | int = (255, 255, 255),
        atlas: np.ndarray = None,
    ) -> np.ndarray:
        """
        This function adds a line to the atlas.
        """
        update_atlas = atlas is None
        if atlas is None:
            atlas = self.quad_sphere_atlas
        h, w = atlas.shape[:2]
        channels = atlas.shape[2] if len(atlas.shape) > 2 else 1
        if channels == 1 and isinstance(color, tuple):
            color = int(color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114)
        elif channels == 1:
            color = int(color)
        src_lat, src_long = src_cart_coords
        dest_lat, dest_long = dest_cart_coords
        src_y, src_x = self.get_quad_tile_mapping((src_lat, src_long), tile_size=1)
        dest_y, dest_x = self.get_quad_tile_mapping((dest_lat, dest_long), tile_size=1)
        src_x = int(src_x)
        src_y = int(src_y)
        dest_x = int(dest_x)
        dest_y = int(dest_y)
        src_face = src_x // self.face_width
        dest_face = dest_x // self.face_width
        if (0 < src_face < 5 and 0 < dest_face < 5) or src_face == dest_face:
            # TODO Proper wrap around and support top and bottom faces
            atlas = cv2.line(atlas, (src_x, src_y), (dest_x, dest_y), color, thickness)
        if update_atlas:
            self.quad_sphere_atlas = atlas
        return atlas


@timing
def normal_to_quad(atlas: np.ndarray, discrete: bool = False) -> np.ndarray:
    return QuadSphere(atlas, discrete=discrete).quad_sphere_atlas


@timing
def quad_to_normal(atlas: np.ndarray, discrete: bool = False) -> np.ndarray:
    normal_shape = get_normal_shape(atlas.shape)

    quad_shape = get_quad_shape(normal_shape)
    if atlas.shape[:2] != quad_shape:
        atlas = cv2.resize(
            atlas,
            quad_shape[::-1],
            interpolation=cv2.INTER_NEAREST if discrete else cv2.INTER_CUBIC
        )
    return QuadSphere(shape=normal_shape, discrete=discrete).get_normal_atlas(atlas)


def quad_rivers_to_normal(rivers: np.ndarray) -> np.ndarray:
    normal_river_sketch = cv2.dilate(rivers, np.ones((3, 3), np.uint8), iterations=2)
    normal_river_sketch = quad_to_normal(normal_river_sketch, discrete=True)
    river_mask = skeletonize(normal_river_sketch > 1) > 0
    normal_river_sketch[~river_mask] = 0
    return normal_river_sketch


def get_quad_shape(normal_shape: tuple[int, int]) -> tuple[int, int]:
    h, w = normal_shape
    assert w == 2 * h
    face_width = math.ceil(w / np.pi) // 2 * 2
    return (face_width, 6 * face_width)


def get_normal_shape(quad_shape: tuple[int, int]) -> tuple[int, int]:
    h, w = quad_shape[:2]
    assert w == 6 * h
    normal_w = int(2 ** round(np.log2(h * np.pi)))
    normal_h = normal_w // 2
    return normal_h, normal_w


if __name__ == "__main__":
    tile_size = 256
    size = 0
    W = 512 * 2**size
    H = 256 * 2**size
    atlas = img.open(f"./planetAI/data/world.satellite.{W}x{H}.png")
    atlas = np.array(atlas)

    quad_atlas = normal_to_quad(atlas)
    normal_atlas = quad_to_normal(quad_atlas)

    sphere = SphereMapping(atlas)
    quad_sphere = QuadSphere(atlas, discrete=True)
    quad_sphere_atlas = quad_sphere.quad_sphere_atlas
    img.fromarray(quad_sphere_atlas).save("./tests/quad_sphere_mapping.png")
    img.fromarray(quad_sphere.get_3x4_quad_atlas(corner_crop=True)).save(
        "./tests/quad_sphere_3x4_mapping.png"
    )
    # normal_atlas = quad_sphere.get_normal_atlas()
    # img.fromarray(normal_atlas).save("./tests/quad_sphere_normal_atlas.png")
    mask = img.open("./data/unused/world.oceanmask.512x256.png")
    mask = np.array(mask)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)

    # Test random uv coords
    cases = 1000
    for _ in range(cases):
        lat = np.random.uniform(-90, 90)
        long = np.random.uniform(-180, 180)
        x, y, z = quad_sphere.get_surface_coords((lat, long), 1)
        u, v = quad_sphere.surface_coords_to_quad_coords(x, y, z)
        # TODO Find out why this doesn't work
        # u, v = quad_sphere.get_quad_tile_mapping((lat, long), tile_size=1)
        face = v // quad_sphere.face_width
        x, y, z = quad_sphere.quad_coords_to_surface_coords(u, v)
        lat_, long_ = quad_sphere.surface_coords_to_coords(x, y, z)
        lat_diff = lat - float(lat_[0][0])
        long_diff = long - float(long_[0][0])
        th = 0.01
        # There are still some failures but it is good enough for now
        if abs(lat_diff) > th:
            print(
                f"Failed lat: abs({lat_diff}) > th for face {int(face[0][0])} with uv {u[0][0], v[0][0]}"
            )
        if abs(long_diff) > th:
            print(
                f"Failed long: abs({long_diff}) > th for face {int(face[0][0])} with uv {u[0][0], v[0][0]}"
            )

    # Test get_valid_coord
    cases = 1000
    for _ in range(cases):
        lat = np.random.uniform(-90, 90)
        long = np.random.uniform(-180, 180)
        lat_, long_ = quad_sphere.get_nearest_valid_coord((lat, long))
        try:
            u, v = quad_sphere.get_quad_tile_mapping((lat_, long_), tile_size=1)
        except NotImplementedError:
            print(f"Failed to get tile mapping for {lat_, long_} from {lat, long}")

    # Test get tiles and distributed points
    num_samples = 100
    lat, long = sphere.get_distributed_points(mask, num_samples)
    num_samples = len(lat)
    cols = 10
    rows = math.ceil(num_samples / cols)
    output_array = np.zeros((rows * tile_size, cols * tile_size, 3), dtype=np.uint8)
    for i, (lat, long) in enumerate(zip(lat, long)):
        tile = sphere.get_tile((lat, long), rotation=0, tile_size=tile_size)
        row = i // cols
        col = i % cols
        output_array[
            row * tile_size: (row + 1) * tile_size,
            col * tile_size: (col + 1) * tile_size,
        ] = tile
    img.fromarray(output_array).save("./tests/sphere_mapping.png")

    # Test set tile
    tile_size = 256
    lat, long = sphere.get_distributed_points(mask >= 0, 16000)
    empty_atlas = np.zeros((H, W, 3), dtype=np.uint8)
    empty_atlas[:, :, 0] = 255
    empty_sphere = SphereMapping(empty_atlas)
    empty_quad_sphere = QuadSphere(empty_atlas)
    tile = np.ones((tile_size, tile_size, 3), dtype=np.uint8) * 255

    lat = list(lat)
    long = list(long)
    lat = [-90] + lat
    long = [0] + long

    def clip(x: int, lower: int, upper: int):
        return min(max(x, lower), upper)

    with tqdm(total=len(lat)) as pbar:
        for i, (lat, long) in enumerate(zip(lat, long)):
            pbar.update(1)
            try:
                quad_tile = quad_sphere.get_quad_tile((lat, long), tile_size=tile_size)
                empty_quad_sphere.set_quad_tile((lat, long), quad_tile)
            except NotImplementedError:
                continue
            continue
            tile = sphere.get_tile((lat, long))
            empty_sphere.set_tile((lat, long), tile)
            y = (90 - lat) * empty_sphere.pixel_resolution
            y = int(clip(y, 0, H - 1))
            x = (long + 180) * empty_sphere.pixel_resolution
            x = int(clip(x, 0, W - 1))
            unproj_tile = sphere.atlas[y - 128: y + 128, x - 128: x + 128]
            new_tile = empty_sphere.get_tile((lat, long))
            empty_sphere.atlas[y, x] = [255, 0, 0]

    img.fromarray(empty_sphere.atlas).save("./tests/sphere_mapping_set_tile.png")
    img.fromarray(empty_quad_sphere.quad_sphere_atlas).save(
        "./tests/quad_sphere_mapping_set_tile.png"
    )
    normal_set = empty_quad_sphere.get_normal_atlas()
    img.fromarray(normal_set).save("./tests/quad_sphere_normal_set_tile.png")
