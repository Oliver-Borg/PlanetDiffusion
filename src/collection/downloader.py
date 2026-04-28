
from threading import Lock

import os
import requests
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from dataclasses import dataclass, field

from .lerc import lerc
from ..core.dir_args import (
    ElevationArguments,
    SatelliteArguments,
    CoverageArguments
)
from ..core.dataclass_argparser import CustomArgumentParser
from ..core.constants import (
    NP_EXT,
    VALID_ELEVATION_DIR_NAME,
    INVALID_ELEVATION_FILE_NAME,
    USA_FROM,
    USA_TO
)
from ..core.tiles import (
    generate_tiles,
    get_invalid_list,
    PER_PIXEL_TILE_ZOOM,
)
from ..core.utils import (
    run_concurrent,
    FunctionWrapper,
    list_all_files,
    get_tiles_list,
    save_numpy
)

BASE_SATELLITE_URL = 'https://server.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile'
BASE_ELEVATION_URL = 'http://elevation3d.arcgis.com/arcgis/rest/services/WorldElevation3D/Terrain3D/ImageServer/tile'
BASE_COVERAGE_URL = 'https://elevation.arcgis.com/arcgis/rest/services/WorldElevation/DataExtents/MapServer/export'

HEIGHTMAP_DL_LOCK = Lock()


@dataclass
class DownloaderArguments(ElevationArguments, SatelliteArguments, CoverageArguments):

    satellite: bool = field(
        default=False,
        metadata={
            'help': 'Whether to download satellite data'
        }
    )

    coverage: bool = field(
        default=False,
        metadata={
            'help': 'Whether to download coverage data'
        }
    )


def dl_image(url, output_file, transform=None, **requests_kwargs):
    try:
        im = Image.open(requests.get(url, stream=True, **requests_kwargs).raw)
        if transform is not None:
            im = transform(im)

        im.save(output_file)
        return output_file
    except (requests.exceptions.ConnectionError, UnidentifiedImageError) as e:
        print(e)  # TODO add retry functionality



def dl_heightmap(zoom, tile_y, tile_x, folder):
    try:
        f_name = f'{zoom}_{tile_y}_{tile_x}'

        url = f'{BASE_ELEVATION_URL}/{zoom}/{tile_y}/{tile_x}'
        d = lerc.decode(requests.get(url).content)
        if isinstance(d, tuple):
            output_file = os.path.join(
                folder, VALID_ELEVATION_DIR_NAME, f_name)
            save_numpy(output_file, d[1])
            return f_name
        else:
            with HEIGHTMAP_DL_LOCK, open(os.path.join(folder, INVALID_ELEVATION_FILE_NAME), 'a') as fp:
                print(f_name, file=fp)
            return False

    except requests.exceptions.ConnectionError as e:
        print(e)  # TODO add retry functionality



def download_tiles(tiles, downloader_args: DownloaderArguments):

    valid_elevation_files = get_tiles_list(os.path.join(
        downloader_args.elevation_dir, downloader_args.valid_dir), NP_EXT)
    invalid_elevation_files = get_invalid_list(downloader_args.elevation_dir)
    downloaded_elev_files = valid_elevation_files | invalid_elevation_files

    downloaded_satt_files = set(list_all_files(
        downloader_args.satellite_dir, 'jpg'))

    # Note: satt subset of elev

    elev_functions_to_run = []
    satt_functions_to_run = []

    with tqdm(total=tiles['total']) as progress:
        for zoom, t_y, t_x, valid in tiles['tiles']:
            progress.update()

            if not valid:
                continue

            tile = f'{zoom}_{t_y}_{t_x}'
            progress.set_description(
                f'collected {len(elev_functions_to_run)}+{len(satt_functions_to_run)} ({tile})')

            # contains height data
            if tile not in downloaded_elev_files:
                elev_functions_to_run.append(FunctionWrapper(
                    dl_heightmap,
                    zoom, t_y, t_x,
                    downloader_args.elevation_dir,
                ))

            satt_file_name = f'{tile}.jpg'
            if satt_file_name not in downloaded_satt_files:
                satt_functions_to_run.append(FunctionWrapper(
                    dl_image,
                    f'{BASE_SATELLITE_URL}/{zoom}/{t_y}/{t_x}',
                    os.path.join(downloader_args.satellite_dir,
                                 'raw', satt_file_name)
                ))

    print('Download elevation data')
    run_concurrent(elev_functions_to_run)

    print('Download satellite data')
    run_concurrent(satt_functions_to_run)


def main():

    from .mask import MaskArguments, generate_mask

    parser = CustomArgumentParser(
        (DownloaderArguments, MaskArguments),
        description="Download data from ArcGIS online's REST API"
    )

    downloader_args, mask_args = parser.parse_args_into_dataclasses()

    mask = generate_mask(mask_args, downloader_args)

    tiles = generate_tiles(
        USA_FROM,
        USA_TO,
        downloader_args.elevation_zoom,
        mask,
        PER_PIXEL_TILE_ZOOM
    )
    download_tiles(
        tiles,
        downloader_args
    )


if __name__ == '__main__':
    main()
