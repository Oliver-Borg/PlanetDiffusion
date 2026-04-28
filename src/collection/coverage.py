
import os
from tqdm import tqdm

from .downloader import dl_image, BASE_COVERAGE_URL
from ..core.tiles import (
    tiles_to_gps,
    latlon_to_meters,
    get_image,
    COVERAGE_TILE_WIDTH
)
from ..core.utils import FunctionWrapper, run_concurrent


# View Elevation Coverage Map:
# https://www.arcgis.com/apps/mapviewer/index.html?layers=3af669838f594b378f90c10f98e46a7f


COVERAGE_HEADERS = {
    'accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.54 Safari/537.36',
}
COVERAGE_PARAMS = {
    'bboxSR': '102100',
    'imageSR': '102100',
    'size': f'{COVERAGE_TILE_WIDTH},{COVERAGE_TILE_WIDTH}',
    'dpi': '96',
    'format': 'png32',
    'transparent': 'true',
    'layers': 'show:1',  # ,2,3,4,5,6,7,8,9,10,11
    'f': 'image',
}


def rgba_to_1(im):
    # Only consider alpha channel (either yes or no)
    return im.split()[-1].point(bool, mode='1')


def coverage_loader_function(tiles, folder, to_download=True, list_of_downloaded=None):

    functions_to_run = []

    paths = []

    params = COVERAGE_PARAMS.copy()
    for zoom, t_y, t_x, _ in tqdm(tiles['tiles'], total=tiles['total']):
        f_name = f'{zoom}_{t_y}_{t_x}_{COVERAGE_TILE_WIDTH}'
        cov_output_name = os.path.join(folder, f'{f_name}.png')

        # progress.set_description(f_name)

        lat, lng = tiles_to_gps(t_y, t_x, zoom)
        lat2, lng2 = tiles_to_gps(t_y + 1, t_x + 1, zoom)

        a, b = latlon_to_meters(lat, lng)
        c, d = latlon_to_meters(lat2, lng2)

        params['bbox'] = f'{a},{b},{c},{d}'

        paths.append(cov_output_name)

        if list_of_downloaded is not None and f_name in list_of_downloaded:
            continue

        if os.path.exists(cov_output_name):
            continue

        if not to_download:
            raise Exception('Missing tile located, and to_download is False')

        functions_to_run.append(FunctionWrapper(
            dl_image,
            BASE_COVERAGE_URL,
            cov_output_name,

            transform=rgba_to_1,
            params=params.copy(),
            headers=COVERAGE_HEADERS
        ))

    if to_download:
        run_concurrent(functions_to_run)

    return paths


def get_coverage_image(tiles, folder, **kwargs):
    return get_image(
        tiles,
        coverage_loader_function,
        folder,
        COVERAGE_TILE_WIDTH,
        img_format='1',
        **kwargs
    )
