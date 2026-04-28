from itertools import product
import os

from tqdm import tqdm
import numpy as np
from PIL import Image

from .constants import (
    USA_FROM,
    USA_TO,
    MAX_ELEVATION_LEVEL,
    VALID_ELEVATION_DIR_NAME,
    NP_EXT,
    GRAD_DOF
)
from .utils import (
    FunctionWrapper,
    get_tiles_list,
    load_numpy,
    run_concurrent,
    array_to_image,
    normalise_array
)
from .tiles import (
    generate_tiles,
    generate_tiles_from_zoom,
    calculate_tiles_crop,
    PER_PIXEL_TILE_ZOOM,
    TILE_SIZE_IN_PIXELS,
)
from ..collection.downloader import (
    dl_heightmap,
    get_invalid_list
)


def elevation_to_image(elevation):
    """Convert elevation numpy array to PIL Image"""

    elevation = normalise_array(elevation)

    return array_to_image(elevation, bit_depth=16)


class MissingTileException(Exception):
    """Raised when an elevation tile is missing"""
    pass


def concat_elevation_tiles(paths, tiles_width, tiles_height, parent_dir=None, keep_overflow=False, raise_if_missing=True):
    """Concatenate multiple elevation tiles into a single, larger numpy array"""
    assert len(paths) == tiles_width * tiles_height

    overflow_amount = GRAD_DOF if keep_overflow else 0
    t_size = TILE_SIZE_IN_PIXELS
    img_width = tiles_width * t_size + overflow_amount
    img_height = tiles_height * t_size + overflow_amount
    next_t_size = t_size + overflow_amount

    to_return = np.zeros((img_width, img_height))

    indices = product(range(tiles_height), range(tiles_width))
    for path, (y, x) in zip(paths, indices):
        if parent_dir is not None:
            path = os.path.join(parent_dir, path)

        if not os.path.exists(path):
            if raise_if_missing:
                raise MissingTileException(f'Missing tile: {path}')
            continue

        elev_data = load_numpy(path)

        to_return[
            y * t_size: (y + 1) * t_size + overflow_amount,
            x * t_size: (x + 1) * t_size + overflow_amount
        ] = elev_data[:next_t_size, :next_t_size]

    return to_return


def get_elevation_data(tiles, folder, to_return=True, crop=True, **kwargs):
    valid_elevation_files = get_tiles_list(os.path.join(
        folder, VALID_ELEVATION_DIR_NAME), NP_EXT, recursive=False)
    invalid_elevation_files = get_invalid_list(folder)

    list_of_downloaded = valid_elevation_files | invalid_elevation_files

    paths = elevation_loader_function(
        tiles, folder, list_of_downloaded=list_of_downloaded, **kwargs)

    to_return = concat_elevation_tiles(paths, tiles['width'], tiles['height'])
    left, upper, right, lower = calculate_tiles_crop(
        tiles, TILE_SIZE_IN_PIXELS)

    if crop:
        to_return = to_return[
            upper:lower,
            left:right
        ]
    return to_return


def get_elevation_data_of_tile(zoom, tile_y, tile_x, folder, desired_zoom=MAX_ELEVATION_LEVEL):
    factor = desired_zoom - zoom
    tiles = generate_tiles_from_zoom(zoom, tile_y, tile_x, factor)
    return get_elevation_data(tiles, folder, crop=False)


def elevation_loader_function(tiles, folder, to_download=True, list_of_downloaded=None):

    functions_to_run = []

    paths = []
    for zoom, t_y, t_x, _ in tqdm(tiles['tiles'], total=tiles['total']):
        f_name = f'{zoom}_{t_y}_{t_x}.{NP_EXT}'
        output_path = os.path.join(folder, VALID_ELEVATION_DIR_NAME, f_name)
        paths.append(output_path)

        if list_of_downloaded is not None and f_name in list_of_downloaded:
            continue
        if os.path.exists(output_path):
            continue

        if not to_download:
            raise Exception('Missing tile located, and to_download is False')

        functions_to_run.append(FunctionWrapper(
            dl_heightmap,
            zoom, t_y, t_x,
            folder
        ))

    if to_download:
        run_concurrent(functions_to_run)

    return paths


def main():

    from .constants import LODS
    from .elevation import get_elevation_data
    from .derivative import elevation_to_SGF, SGF_to_image

    tiles = generate_tiles(USA_FROM, USA_TO, PER_PIXEL_TILE_ZOOM)

    # Generate heightmap of America
    elev_folder = 'data/elevation'
    elev_data = get_elevation_data(tiles, elev_folder)

    # Derivative data
    res = LODS[PER_PIXEL_TILE_ZOOM][0]
    deriv_data = elevation_to_SGF(elev_data, res, res)
    deriv_img = SGF_to_image(deriv_data)

    # Scale elevation data to generate image
    elev_data -= np.min(elev_data)
    elev_data = 255 * elev_data/np.max(elev_data)
    elev_img = Image.fromarray(elev_data).convert('L')

    # Save images
    elev_img.save('data/masks/elevation.jpg', quality=100)
    deriv_img.save('data/masks/derivative.png', quality=100)

    import matplotlib.pyplot as plt
    figure, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 9))
    ax1.imshow(elev_img)
    ax2.imshow(deriv_img)

    plt.show()


if __name__ == '__main__':
    main()
