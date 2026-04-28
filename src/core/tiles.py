import re
from itertools import product
import os
import math
from PIL import Image

from .constants import (
    MAX_SATELLITE_LEVEL,
    MAX_ELEVATION_LEVEL,
    INVALID_ELEVATION_FILE_NAME
)

COVERAGE_TILE_WIDTH = 4096
TILE_SIZE_IN_PIXELS = 256


# zoom level so that 1 pixel = 1 tile at MAX_ELEVATION
PER_PIXEL_TILE_ZOOM = MAX_ELEVATION_LEVEL - \
    int(math.log2(TILE_SIZE_IN_PIXELS))  # 8

PER_PIXEL_COVERAGE_ZOOM = MAX_ELEVATION_LEVEL - \
    int(math.log2(COVERAGE_TILE_WIDTH))  # 4

ORIGIN_SHIFT = math.pi / 180.0 * 6378137


def lerp(f, t, p):
    return (1-p) * f + p*t


def gps_to_tiles(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    ytile = (lon_deg + 180.0) / 360.0 * n

    return (xtile, ytile)


def tiles_to_gps(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = ytile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * xtile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)


def generate_tiles(start, end, zoom, mask=None, mask_zoom=None, x_range=(0, 1), y_range=(0, 1)):

    if start[0] < end[0]:
        min_x = start[0]
        max_x = end[0]
    else:
        min_x = end[0]
        max_x = start[0]

    if start[1] < end[1]:
        min_y = start[1]
        max_y = end[1]
    else:
        min_y = end[1]
        max_y = start[1]

    d, b = gps_to_tiles(min_x, min_y, zoom)
    a, c = gps_to_tiles(max_x, max_y, zoom)

    min_t_x = lerp(b, c, x_range[0])
    max_t_x = lerp(b, c, x_range[1])
    min_t_y = lerp(a, d, y_range[0])
    max_t_y = lerp(a, d, y_range[1])

    x_range = range(math.floor(min_t_x), math.ceil(max_t_x) + 1)
    y_range = range(math.floor(min_t_y), math.ceil(max_t_y) + 1)

    offsets = (min_t_x, max_t_x, min_t_y, max_t_y)

    if mask is None or mask_zoom is None:  # Ignore mask
        def tile_gen():
            for t_y in y_range:
                for t_x in x_range:
                    yield (zoom, t_y, t_x, True)

    else:  # Consider mask
        def tile_gen():
            for j, t_y in enumerate(y_range):
                for i, t_x in enumerate(x_range):
                    yield (zoom, t_y, t_x, mask[j, i])

    return {
        'tiles': tile_gen(),
        'zoom': zoom,
        'mask_zoom': mask_zoom,
        'offsets': offsets,
        'width': len(x_range),
        'height': len(y_range),
        'total': len(x_range) * len(y_range)
    }


def latlon_to_meters(lat, lon):
    "Converts given lat/lon in WGS84 Datum to XY in Spherical Mercator EPSG:900913"

    mx = lon * ORIGIN_SHIFT
    my = math.log(math.tan((90 + lat) * math.pi / 360.0)) / (math.pi / 180.0)

    my = my * ORIGIN_SHIFT
    return mx, my


def meters_to_latlon(mx, my):
    """Converts XY point from Spherical Mercator EPSG:900913 to lat/lon in WGS84 Datum

    https://sampleserver1.arcgisonline.com/ArcGIS/rest/services/Geometry/GeometryServer/project?inSR=102100&outSR=4326&geometries=%7B%0D%0A++%22geometryType%22+%3A+%22esriGeometryPoint%22%2C%0D%0A++%22geometries%22+%3A+%5B%0D%0A+++++%7B%0D%0A+++++++%22x%22+%3A+-11696523.780400001%2C+%0D%0A+++++++%22y%22+%3A+4804891.0001000017%0D%0A+++++%7D%0D%0A++%5D%0D%0A%7D&f=HTML
    https://gis.stackexchange.com/questions/278165/getting-lat-lng-from-wkid-latestwkid-and-x-y-coordinates
    """
    lon = mx / ORIGIN_SHIFT
    lat = my / ORIGIN_SHIFT

    lat = 180 / math.pi * \
        (2 * math.atan(math.exp(lat * math.pi / 180.0)) - math.pi / 2.0)
    return lat, lon


def generate_tiles_from_zoom(zoom, tile_y, tile_x, factor):
    # Given a tile, zoom in and return all internal tiles
    assert factor >= 0

    zoom_range = 2 ** factor

    start_offset_y = tile_y * zoom_range
    start_offset_x = tile_x * zoom_range

    new_zoom = zoom + factor

    assert new_zoom <= MAX_SATELLITE_LEVEL

    tiles = ((
        new_zoom, start_offset_y + j, start_offset_x + i, True)
        for j in range(zoom_range)
        for i in range(zoom_range)
    )
    return {
        'tiles': tiles,
        'zoom': zoom,
        'width': zoom_range,
        'offsets': (0, 0, 1, 1),
        'height': zoom_range,
        'total': zoom_range * zoom_range
    }


def get_image_at_factor(paths, factor, tile_size_in_pixels, img_format='RGB', parent_dir=None):
    scale = 2 ** factor
    return concat_image_tiles(paths, scale, scale, tile_size_in_pixels, img_format=img_format, parent_dir=parent_dir)


def concat_image_tiles(paths, tiles_width, tiles_height, tile_size_in_pixels, img_format='RGB', parent_dir=None):
    assert len(paths) == tiles_width * tiles_height

    img_width = tiles_width * tile_size_in_pixels
    img_height = tiles_height * tile_size_in_pixels

    new_im = Image.new(img_format, (img_width, img_height))

    indices = product(range(tiles_height), range(tiles_width))
    for path, (y, x) in zip(paths, indices):
        if parent_dir is not None:
            path = os.path.join(parent_dir, path)

        if not os.path.exists(path):
            continue

        img = Image.open(path)

        offset_y = tile_size_in_pixels * y
        offset_x = tile_size_in_pixels * x

        new_im.paste(img, (offset_x, offset_y))

    return new_im


def get_image(tiles_info, loader_function, folder, tile_size_in_pixels, img_format, return_image=True, to_download=True, crop=True):
    paths = loader_function(tiles_info, folder, to_download)

    # must do it after since we dl images in parallel
    if return_image:
        img = concat_image_tiles(
            paths,
            tiles_info['width'],
            tiles_info['height'],
            tile_size_in_pixels,
            img_format
        )

        if crop:
            img = crop_img_by_tiles(
                img,
                tiles_info,
                tile_size_in_pixels
            )
        return img

# TODO make generic bounds calc


def crop_img_by_tiles(img, tiles, tile_size_in_pixels):
    img_width, img_height = img.size

    left, upper, right, lower = calculate_tiles_crop(
        tiles, tile_size_in_pixels)

    return img.crop((left, upper, right + img_width, lower + img_height))


def calculate_tiles_crop(tiles, tile_size_in_pixels):
    # Crop image based on range
    min_t_x, max_t_x, min_t_y, max_t_y = tiles['offsets']

    # Extract fractional parts
    a = min_t_x % 1
    b = 2 - max_t_x % 1
    c = min_t_y % 1
    d = 2 - max_t_y % 1

    left = math.floor(tile_size_in_pixels * a)
    right = - math.floor(tile_size_in_pixels * b) + 1

    upper = math.floor(tile_size_in_pixels * c)
    lower = - math.floor(tile_size_in_pixels * d) + 1

    return (left, upper, right, lower)


def tile_info_to_fname(tile_info, ext='png'):
    zoom, t_y, t_x = tile_info[:3]
    return f"{zoom}_{t_y}_{t_x}.{ext}"


def fname_to_tile_info(fname):
    return list(map(int, filter(None, re.split(r'\D+', fname.split('.')[0]))))


def get_invalid_list(elevation_dir):
    """Return list of invalid tiles"""
    with open(os.path.join(elevation_dir, INVALID_ELEVATION_FILE_NAME)) as fp:
        return set(fp.read().split())
