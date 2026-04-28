import shutil
import os
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from tqdm import tqdm

from .coverage import get_coverage_image
from ..models.classifier.classes import TYPE_MAPPINGS
from ..core.dir_args import (
    ElevationArguments,
    SatelliteArguments,
    CoverageArguments
)
from ..core.satellite import get_satellite_image
from ..core.tiles import (
    PER_PIXEL_COVERAGE_ZOOM,
    PER_PIXEL_TILE_ZOOM,
    generate_tiles,
    fname_to_tile_info
)
from ..core.utils import list_all_files
from ..core.dataclass_argparser import CustomArgumentParser
from ..core.constants import (
    MAX_ELEVATION_LEVEL,
    USA_FROM,
    USA_TO,
    NP_EXT
)

MASK_QUALITY = 100


def open_mask(path, invert=False):
    mask = Image.open(path).convert('1').point(bool, mode='1')
    mask = np.array(mask).astype(bool)
    if invert:
        mask = np.logical_not(mask)

    return mask


def merge_masks(*masks):
    if not masks:
        return None

    mask = masks[0]
    for m in masks[1:]:
        mask = np.logical_and(mask, m)

    return mask


@dataclass
class MaskArguments:


    data_dir: str = field(
        default='./data',
        metadata={
            'help': 'Directory to store data'
        }
    )

    masks_dir: str = field(
        default='./data/masks',
        metadata={
            'help': 'Directory to store masks'
        }
    )

    coverage_mask: str = field(
        default='coverage_mask.jpg',
        metadata={
            'help': 'Name of coverage mask'
        },
    )
    overwrite_coverage: bool = field(
        default=False,
        metadata={
            'help': 'Whether to overwrite coverage data (will generate if does not exist)'
        }
    )

    user_mask: str = field(
        default='user_mask.jpg',
        metadata={
            'help': 'Name of user mask'
        },
    )

    ignore_mask: str = field(
        default='ignore_mask.jpg',
        metadata={
            'help': 'Name of ignore mask'
        },
    )
    generate_ignore_mask: bool = field(
        default=False,
        metadata={
            'help': 'Whether to generate ignore data'
        }
    )
    include_ml_moderated_in_ignore_mask: bool = field(
        default=False,
        metadata={
            'help': 'Whether to include ML-moderated invalid tiles in the ignore mask'
        }
    )

    satellite_image: str = field(
        default='satellite.jpg',
        metadata={
            'help': 'Name of satellite image'
        },
    )
    overwrite_satellite: bool = field(
        default=False,
        metadata={
            'help': 'Whether to overwrite satellite data (will generate if does not exist)'
        }
    )

    purge_elevation_files: bool = field(
        default=False,
        metadata={
            'help': 'Whether to delete elevation data that is not contained in the mask'
        }
    )
    display_mask: bool = field(
        default=False,
        metadata={
            'help': 'Whether to display the generated mask'
        }
    )


def generate_satellite(mask_args: MaskArguments, sat_args: SatelliteArguments):
    sat_img_path = os.path.join(mask_args.masks_dir, mask_args.satellite_image)
    if not os.path.exists(sat_img_path) or mask_args.overwrite_satellite:
        # Generate satellite image
        tiles = generate_tiles(USA_FROM, USA_TO, PER_PIXEL_TILE_ZOOM)
        sat_cache_folder = os.path.join(
            sat_args.satellite_dir, sat_args.cache_dir)
        sat_img = get_satellite_image(tiles, folder=sat_cache_folder)
        sat_img.save(sat_img_path, quality=MASK_QUALITY)
        return sat_img

    return Image.open(sat_img_path)


def generate_mask(mask_args: MaskArguments, coverage_args: CoverageArguments):
    masks = []
    masks.append(generate_coverage_mask(mask_args, coverage_args))

    user_mask = open_user_mask(mask_args)
    if user_mask is not None:
        masks.append(user_mask)

    ignore_mask = open_ignore_mask(mask_args)
    if ignore_mask is not None:
        masks.append(ignore_mask)

    return merge_masks(*masks)


def open_ignore_mask(mask_args: MaskArguments):
    ignore_mask_path = os.path.join(mask_args.masks_dir, mask_args.ignore_mask)

    
    # Generate ignore mask
    if mask_args.generate_ignore_mask:
        tiles = generate_tiles(USA_FROM, USA_TO, MAX_ELEVATION_LEVEL)
        min_t_x, _, min_t_y, _ = map(int, tiles['offsets'])
        mask = np.zeros((tiles['height'], tiles['width']), dtype=bool)

        file_generators = []

        for c_name in TYPE_MAPPINGS:
            c_type, _, ext = TYPE_MAPPINGS[c_name]
            invalid = c_type.invalid()
            for i in invalid:
                for t in ('train', 'valid', 'test', 'to_divide'):
                    invalid_dir = os.path.join(mask_args.data_dir, c_name, 'moderated', t, i.name)
                    file_generators.append(list_all_files(invalid_dir, ext))

                if mask_args.include_ml_moderated_in_ignore_mask:
                    invalid_dir = os.path.join(mask_args.data_dir, c_name, 'ml_moderated', i.name)
                    file_generators.append(list_all_files(invalid_dir, ext))
        
            # Add prefiltered images
            invalid_dir = os.path.join(mask_args.data_dir, c_name, 'moderated', 'ignore')
            file_generators.append(list_all_files(invalid_dir, ext))

            # Add other images
            invalid_dir = os.path.join(mask_args.data_dir, c_name, 'add_to_ignore_mask')
            file_generators.append(list_all_files(invalid_dir, ext))

        for invalid_files in file_generators:
            for f in invalid_files:
                zoom, tile_y, tile_x = fname_to_tile_info(f)
                scale = 2 ** (MAX_ELEVATION_LEVEL - zoom)
                t_y = tile_y * scale - min_t_y
                t_x = tile_x * scale - min_t_x

                t_y2 = (tile_y + 1) * scale - min_t_y
                t_x2 = (tile_x + 1) * scale - min_t_x

                mask[t_y:t_y2, t_x:t_x2] = True

        # Save image
        img_frombytes(mask).save(ignore_mask_path, quality=MASK_QUALITY)

        return np.logical_not(mask)

    if os.path.exists(ignore_mask_path):
        return open_mask(ignore_mask_path, True)

    return None


def open_user_mask(mask_args: MaskArguments):
    user_mask_path = os.path.join(mask_args.masks_dir, mask_args.user_mask)
    if os.path.exists(user_mask_path):
        return open_mask(user_mask_path)
    else:
        # Construct fully white image from coverage
        cov_mask_path = os.path.join(mask_args.masks_dir, mask_args.coverage_mask)
        if os.path.exists(cov_mask_path):
            print('User mask not found, generating blank template from coverage mask')
            img = np.ones_like(np.array(Image.open(cov_mask_path)), dtype=bool)
            Image.fromarray(img).save(user_mask_path)
            return img 

    return None


def generate_coverage_mask(mask_args: MaskArguments, coverage_args: CoverageArguments):

    cov_mask_path = os.path.join(mask_args.masks_dir, mask_args.coverage_mask)
    if not os.path.exists(cov_mask_path) or mask_args.overwrite_coverage:
        print('Create coverage image')
        cov_tiles = generate_tiles(
            USA_FROM,
            USA_TO,
            PER_PIXEL_COVERAGE_ZOOM
        )
        cov_img = get_coverage_image(cov_tiles, coverage_args.coverage_dir)
        cov_img.save(cov_mask_path, quality=MASK_QUALITY)

    return open_mask(cov_mask_path)


def img_frombytes(data):
    """
    https://stackoverflow.com/questions/50134468/convert-boolean-numpy-array-to-pillow-image
    """
    size = data.shape[::-1]
    databytes = np.packbits(data, axis=1)
    return Image.frombytes(mode='1', size=size, data=databytes)


def main():
    parser = CustomArgumentParser(
        (MaskArguments, SatelliteArguments, CoverageArguments, ElevationArguments),
        description='Generate coverage, user, ignore masks (and satellite image)'
    )
    mask_args, sat_args, cov_args, elev_args = parser.parse_args_into_dataclasses()

    # Generate satellite image
    sat_img = generate_satellite(mask_args, sat_args)

    # Generate mask
    mask = generate_mask(mask_args, cov_args)

    if mask_args.purge_elevation_files:
        tiles = generate_tiles(USA_FROM, USA_TO, MAX_ELEVATION_LEVEL)
        min_t_x, _, min_t_y, _ = map(int, tiles['offsets'])

        folder = os.path.join(elev_args.elevation_dir, elev_args.valid_dir)
        to_delete_folder = os.path.join(
            elev_args.elevation_dir, elev_args.to_delete)
        if not os.path.exists(to_delete_folder):
            os.makedirs(to_delete_folder, exist_ok=True)
        all_elev_files = list(list_all_files(folder, NP_EXT, recursive=False))

        to_delete_count = 0
        valid_count = 0

        progress = tqdm(all_elev_files)
        for f in progress:
            zoom, t_y, t_x = fname_to_tile_info(f)

            if zoom != MAX_ELEVATION_LEVEL:
                continue

            offset_y = t_y - min_t_y
            offset_x = t_x - min_t_x

            is_valid = mask[offset_y, offset_x]
            if is_valid:
                valid_count += 1
            else:
                to_delete_count += 1
                path = os.path.join(folder, f)
                shutil.move(path, to_delete_folder)

            progress.set_description(
                f'{f}={is_valid} ({valid_count}:{to_delete_count})')

    if mask_args.display_mask:
        import matplotlib.pyplot as plt
        figure, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 9))
        ax1.imshow(sat_img)
        ax2.imshow(mask)

        plt.show()


if __name__ == '__main__':
    main()
