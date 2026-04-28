
from tqdm import tqdm
import os

from .utils import (
    FunctionWrapper,
    run_concurrent
)
from .tiles import (
    generate_tiles_from_zoom,
    TILE_SIZE_IN_PIXELS,
    get_image
)
from ..collection.downloader import (
    dl_image,
    BASE_SATELLITE_URL
)


def satellite_loader_function(tiles, folder, to_download=True, list_of_downloaded=None):

    functions_to_run = []

    paths = []
    for zoom, t_y, t_x, _ in tqdm(tiles['tiles'], total=tiles['total']):

        f_name = f'{zoom}_{t_y}_{t_x}.jpg'

        output_path = os.path.join(folder, f_name)
        paths.append(output_path)

        if list_of_downloaded is not None and f_name in list_of_downloaded:
            continue

        if os.path.exists(output_path):
            continue

        if not to_download:
            raise Exception('Tile not found, and to_download is False')
        url = f'{BASE_SATELLITE_URL}/{zoom}/{t_y}/{t_x}'

        functions_to_run.append(FunctionWrapper(
            dl_image,
            url,
            output_path
        ))

    if to_download:
        run_concurrent(functions_to_run)

    return paths


def get_satellite_image(tiles, folder, **kwargs):
    return get_image(
        tiles,
        satellite_loader_function,
        folder,
        TILE_SIZE_IN_PIXELS,
        img_format='RGB',
        **kwargs
    )


def get_satellite_image_of_tile(zoom, tile_y, tile_x, factor, folder):
    tiles = generate_tiles_from_zoom(zoom, tile_y, tile_x, factor)
    return get_satellite_image(tiles, folder)


def main():
    pass  # TODO add options


if __name__ == '__main__':
    main()
