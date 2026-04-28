import json
from dataclasses import dataclass, field
from itertools import product
import os
import random
import tarfile
import re

from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from PIL import Image

from .dataclass_argparser import CustomArgumentParser
from .constants import (
    MAX_ELEVATION_LEVEL,
    LODS,
    NP_EXT,
    PNG_EXT,
    JPG_EXT
)
from .tiles import (
    TILE_SIZE_IN_PIXELS,
    fname_to_tile_info,
    get_image_at_factor
)
from .utils import (
    list_all_files,
    load_image,
    save_image,
    normalise_image
)
from .terrain_transforms import (
    RandomTerrainTransform,
    NormaliseTransform
)
from ..models.classifier.classes import DerivativeFilterClass, SatelliteFilterClass


def generate_tile_components(tile_y, tile_x, scale):
    return list(product(
        range(tile_y, tile_y + scale),
        range(tile_x, tile_x + scale)
    ))


def scan_folders(folders, ext, zoom=None):
    if folders is None:
        return {}
    if not isinstance(folders, list):
        folders = [folders]

    tiles = {}

    for folder in folders:
        folder_files = list_all_files(folder, ext,
                                      full_path=True,
                                      recursive=True)

        for d, f in folder_files:
            z, *t = fname_to_tile_info(f)
            if zoom is not None and z != zoom:
                continue  # Only keep tiles of certain zoom level

            tiles[tuple(t)] = os.path.join(d, f)

    return tiles


@dataclass
class ProcessDatasetArguments:
    processed_folder: str = field(
        default='./data/processed',
        metadata={
            'help': 'The directory to store processed data'
        }
    )
    satellite_folder: str = field(
        default='./data/satellite',
        metadata={
            'help': 'The directory to store satellite data'
        }
    )
    derivative_folder: str = field(
        default='./data/derivative',
        metadata={
            'help': 'The directory to store derivative data'
        }
    )

    factor: int = field(
        default=3,
        metadata={
            'help': 'Factor used to concatenate tiles'
        }
    )
    zoom: int = field(
        default=MAX_ELEVATION_LEVEL,
        metadata={
            'help': 'What zoom level to use'
        }
    )
    generate_elevation: bool = field(
        default=False,
        metadata={
            'help': 'Whether to generate elevation data'
        }
    )
    generate_satellite: bool = field(
        default=False,
        metadata={
            'help': 'Whether to generate satellite data'
        }
    )
    generate_sketch: bool = field(
        default=False,
        metadata={
            'help': 'Whether to generate sketch data'
        }
    )
    generate_if_folder_exists: bool = field(
        default=False,
        metadata={
            'help': 'Only generate data if tile folder exists'
        }
    )
    output_sizes: int = field(
        default_factory=lambda: [256, 512],  # 1024, , 256, 128, 64, 32
        metadata={
            'help': 'Sizes of images to generate',
            'nargs': '+'
        }
    )

    archive_name: str = field(
        default=None,
        metadata={
            'help': 'Name of the tar.gz archive to create (None means do not create archive)'
        }
    )
    archive_files: str = field(
        default=None,
        metadata={
            'help': 'List of patterns to match files that will be added to the archive',
            'nargs': '+'
        }
    )

    archive_min_height: float = field(
        default=None,
        metadata={
            'help': 'If specified, only include tiles with elevation range >= archive_min_height',
        }
    )
    archive_max_height: float = field(
        default=None,
        metadata={
            'help': 'If specified, only include tiles with elevation range < archive_max_height',
        }
    )
    archive_min_file_size: float = field(
        default=None,
        metadata={
            'help': 'If specified, only include tiles with elevation file_size >= archive_min_file_size (in KB)',
        }
    )

    ignore_satellite: bool = field(
        default=False,
        metadata={
            'help': 'Whether to use moderated satellite data when filtering'
        }
    )


def any_path_exists(folders, file, default=False):
    # Helper function for checking if a path exists in many folders
    for folder in folders:
        path = os.path.join(folder, file)
        if os.path.exists(path):
            return path

    return default


def exists(x):
    return x is not None
        
class TerrainDataset(Dataset):
    def __init__(self,
                 folder,

                 index_mapping,
                 data_augmentation=True,
                 normalise=True,  # PIL Image -> (-1, 1)
                 seed=None,

                 remove_incomplete_tiles=False  # Safety, but is quite slow
                 ):

        self.folder = folder
        self.index_mapping = index_mapping
        self.data_augmentation = data_augmentation 

        self.tiles = os.listdir(folder)
        self.tiles = list(filter(lambda x: re.match(r"^\d+_\d+_\d+_\d+.*$", x), self.tiles))
        if remove_incomplete_tiles:

            # Remove incomplete
            non_null_mappings = list(filter(exists, index_mapping.values()))
            non_null_mappings.append(METADATA_FILE)

            def filter_function(tile_dir):
                return all(os.path.exists(os.path.join(folder, tile_dir, x)) for x in non_null_mappings)

            self.tiles = list(filter(filter_function, tqdm(
                self.tiles, desc='Building dataset')))

        assert len(self.tiles) > 0, 'dataset empty'

        self.seed = seed
        if seed is not None:
            # https://stackoverflow.com/a/19307329/13989043
            random.Random(seed).shuffle(self.tiles)

        # Create transformation that will be applied to each image
        transform_items = [
            # Convert Image or numpy array to tensor
            transforms.ToTensor()
        ]

        if normalise:
            transform_items.append(NormaliseTransform())

        self.transform = transforms.Compose(transform_items)

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        # TODO generate conditional info on the fly?

        tile = self.tiles[idx]
        tile_path = os.path.join(self.folder, tile)

        to_return = {}
        for key, value in self.index_mapping.items():
            if value is not None:
                img = load_image(os.path.join(tile_path, value))
                img = self.transform(img)
                to_return[key] = img

        # add data augmentation so that both conditional and target get same transform
        if self.data_augmentation:
            transform = RandomTerrainTransform.get_random_transform()
            for k in to_return:
                to_return[k] = RandomTerrainTransform.apply_transform(
                    to_return[k], transform)

        # Load tile metadata
        with open(os.path.join(tile_path, METADATA_FILE)) as fp:
            metadata = json.load(fp)

        metadata['resolution'] = LODS[metadata['zoom'] - metadata['factor']][0]
        to_return['metadata'] = metadata
        return to_return


METADATA_FILE = 'metadata.json'


def main():

    # +--------+-----------+------------+--------------+
    # | factor | img_size  | extent     | num_tiles    |
    # +--------+-----------+------------+--------------+
    # | 0      | 256x256   | 611m       | 1x1 = 1      |
    # | 1      | 512x512   | 1223m      | 2x2 = 4      |
    # | 2      | 1024x1024 | 2446m      | 4x4 = 16     | [1]
    # | 3      | 2048x2048 | 4892m      | 8x8 = 64     | [2]
    # | 4      | 4096x4096 | 9784m      | 16x16 = 256  |
    # | 5      | 8192x8192 | 19568m     | 32x32 = 1024 |
    # +--------+-----------+------------+--------------+

    from ..labelling.label import generate_conditioning
    from .elevation import concat_elevation_tiles, elevation_to_image, MissingTileException

    # TODO make args
    raw_elevation_folder = 'data/elevation/valid'

    # Script to generate processed dataset
    parser = CustomArgumentParser(
        ProcessDatasetArguments,
        description='Used to process and create a dataset'
    )

    process_args,  = parser.parse_args_into_dataclasses()

    zoom = process_args.zoom
    factor = process_args.factor

    # Specify minimum threshold
    # +--------+-----------+
    # | factor | threshold |
    # +--------+-----------+
    # |      1 |         5 |
    # |      2 |        10 |
    # |      3 |        20 |
    # |      4 |        40 |
    # +--------+-----------+
    difference_threshold = 5 * (2**(factor-1))

    # Must do at least one thing
    if process_args.generate_elevation or process_args.generate_sketch or process_args.generate_satellite:

        dirs_to_use = ('moderated/train', 'moderated/test',
                       'moderated/to_divide', 'ml_moderated')

        # Controls how many tiles to concatenate
        scale = 2 ** factor

        # Resolutions to output final images as
        image_sizes = set(process_args.output_sizes)

        # Base resolution of concatenated tile
        res = LODS[zoom - factor][0]

        # Folders which contain valid derivative tiles
        deriv_folders = [
            f'{process_args.derivative_folder}/{t}/{v.name}'
            for t in dirs_to_use
            for v in DerivativeFilterClass.valid()
        ]

        # Folders which contain valid satellite tiles
        sat_folders = [
            f'{process_args.satellite_folder}/{t}/{v.name}'
            for t in dirs_to_use
            for v in SatelliteFilterClass.valid()
        ]
        # Folders which contain invalid satellite tiles
        invalid_sat_folders = [
            f'{process_args.satellite_folder}/{t}/{v.name}'
            for t in dirs_to_use
            for v in SatelliteFilterClass.invalid_satellite()
        ]

        print('Scanning folders')
        derivative_tiles = scan_folders(deriv_folders, PNG_EXT, zoom=zoom)
        possible_keys = derivative_tiles.keys()

        if process_args.ignore_satellite:
            if process_args.generate_satellite:
                raise ValueError(
                    'cannot have --ignore_satellite and --generate_satellite')
            satellite_tiles = {}
            invalid_satellite_tiles = {}
        else:
            satellite_tiles = scan_folders(sat_folders, JPG_EXT, zoom=zoom)
            invalid_satellite_tiles = scan_folders(
                invalid_sat_folders, JPG_EXT, zoom=zoom)

            possible_keys &= satellite_tiles.keys()

        # Tiles that have valid elevation data
        possible_keys = list(possible_keys)
        random.shuffle(possible_keys)

        # Used to check if already generated
        dirs_to_check = [
            os.path.join(process_args.processed_folder, x)
            for x in ('to_divide', 'train', 'test', 'valid')
        ]
        to_divide_dir = dirs_to_check[0]

        # Next, process all tiles which have all necessary data.
        # For every tile given, use this as the top-left tile and see
        # if remaining tiles are available in any folder
        with tqdm(possible_keys) as progress:
            for tile_info in progress:
                tile_y, tile_x = tile_info

                # All components of the tile are valid, so we join them together
                # Use top-left as the index
                tile_name = f'{zoom}_{tile_y}_{tile_x}_{factor}'

                # TODO save metadata.json
                # Where to save processed files
                tile_folder = any_path_exists(
                    folders=dirs_to_check,
                    file=tile_name,
                    default=None
                )

                if tile_folder is None:
                    if process_args.generate_if_folder_exists:
                        continue
                    tile_folder = os.path.join(to_divide_dir, tile_name)

                metadata_path = os.path.join(tile_folder, METADATA_FILE)
                if os.path.exists(metadata_path):
                    with open(metadata_path) as fp:
                        metadata = json.load(fp)
                else:
                    metadata = None

                # Generate tiles that will be merged
                tile_components = generate_tile_components(
                    tile_y, tile_x, scale)
                # Check if data exists and is valid
                deriv_paths = [derivative_tiles.get(
                    x) for x in tile_components]
                if None in deriv_paths:
                    continue  # At least one tile is invalid

                if not process_args.ignore_satellite:
                    sat_paths = [satellite_tiles.get(
                        x) for x in tile_components]
                    if None in sat_paths:
                        continue  # At least one tile is invalid

                if process_args.generate_elevation or process_args.generate_sketch:

                    # Remove files that have already been generated
                    elev_files_to_generate = {}
                    if process_args.generate_elevation:
                        elev_files_to_generate = {
                            image_size: os.path.join(
                                tile_folder, f'elevation-{image_size}x{image_size}.{PNG_EXT}')
                            for image_size in image_sizes
                        }
                        elev_files_to_generate = {
                            k: v for k, v in elev_files_to_generate.items()
                            if not os.path.exists(v)
                        }

                    sketch_files_to_generate = {}
                    if process_args.generate_sketch:
                        sketch_files_to_generate = {
                            image_size: os.path.join(
                                tile_folder, f'sketch-{image_size}x{image_size}.{PNG_EXT}')
                            for image_size in [256]
                            # TODO support other sizes
                            # for image_size in image_sizes
                        }
                        sketch_files_to_generate = {
                            k: v for k, v in sketch_files_to_generate.items()
                            if not os.path.exists(v)
                        }

                    image_sizes_to_generate = elev_files_to_generate.keys(
                    ) | sketch_files_to_generate.keys()

                    if image_sizes_to_generate or not os.path.exists(metadata_path):
                        # Save/load elevation data
                        elev_paths = [
                            os.path.join(raw_elevation_folder,
                                         f'{zoom}_{y}_{x}.{NP_EXT}')
                            for y, x in tile_components
                        ]
                        try:
                            elevation_data = concat_elevation_tiles(
                                elev_paths, scale, scale, keep_overflow=True)
                        except MissingTileException as e:
                            # Ignore tile since it doesn't contain all info
                            continue

                        # Check variability
                        min_val = np.min(elevation_data)
                        max_val = np.max(elevation_data)
                        difference = max_val - min_val

                        if difference < difference_threshold:
                            continue

                        # TODO add more data
                        metadata = {
                            'range': difference,
                            'zoom': zoom,
                            'tile_y': tile_y,
                            'tile_x': tile_x,
                            'factor': factor,
                        }

                        os.makedirs(tile_folder, exist_ok=True)

                        # Update modified time (useful for sorting)
                        os.utime(tile_folder)

                        # Write metadata:
                        with open(metadata_path, 'w') as fp:
                            json.dump(metadata, fp)

                        if image_sizes_to_generate:  # Something to process

                            # Save elevation images at various sizes
                            r, c = elevation_data.shape
                            for image_size in image_sizes_to_generate:

                                step_y = r // image_size
                                step_x = c // image_size

                                # NOTE: this gives 2D array with some overflow
                                # e.g., 257x257 or 513x513
                                # (needed for derivative calculations)
                                scaled_elevation_data = elevation_data[::step_y, ::step_x]

                                if process_args.generate_elevation:
                                    output_elev_path = elev_files_to_generate.get(
                                        image_size)
                                    if output_elev_path:
                                        # Save at correct size
                                        cropped_elev_data = scaled_elevation_data[:image_size, :image_size]
                                        elev = elevation_to_image(
                                            cropped_elev_data)
                                        save_image(output_elev_path, elev)

                                # TODO add option to generate multiple label images
                                # TODO update label generation
                                if process_args.generate_sketch:
                                    output_sketch_path = sketch_files_to_generate.get(
                                        image_size)
                                    if output_sketch_path:
                                        sketch = generate_conditioning(scaled_elevation_data.astype(np.float32), resolution=res)

                                        sketch = sketch[:image_size,
                                                        :image_size]
                                        sketch_img = Image.fromarray(sketch)
                                        save_image(
                                            output_sketch_path, sketch_img)

                # Only generate satellite imagery if all tiles are valid
                if process_args.generate_satellite and all(x not in invalid_satellite_tiles for x in tile_components):
                    sat_img = get_image_at_factor(
                        sat_paths, factor, TILE_SIZE_IN_PIXELS)

                    for image_size in image_sizes:
                        output_satellite_path = os.path.join(
                            tile_folder, f'satellite-{image_size}x{image_size}.{PNG_EXT}')
                        if not os.path.exists(output_satellite_path):
                            resized_img = sat_img.resize(
                                (image_size, image_size))
                            save_image(output_satellite_path, resized_img)

                progress.set_description(f'Processed {tile_name}')

    if process_args.archive_name is not None and process_args.archive_files is not None:

        # TODO add elev range options
        # TODO include either all or none of a tile (depending if contains all needed)
        dirs_to_add = ('train', 'valid', 'test')
        archive_files = set(process_args.archive_files)
        archive_files.add('metadata.json')  # Always include metadata file

        archive_min_height = process_args.archive_min_height or 0
        archive_max_height = process_args.archive_max_height or float('inf')

        # Create tar.gz with selected files
        with tarfile.open(process_args.archive_name, 'w:gz') as tar:
            for d in dirs_to_add:
                d_path = os.path.join(process_args.processed_folder, d)
                tiles = os.listdir(d_path)
                for tile in tqdm(tiles):
                    paths = [
                        # Real path, archive path
                        (os.path.join(d_path, tile, x), os.path.join(d, tile, x))
                        for x in archive_files
                    ]
                    if any((not os.path.exists(x[0]) for x in paths)):
                        continue
                    
                    # Filter elevation based on size (if specified)
                    if exists(process_args.archive_min_file_size) and any(
                        'elevation-' in x[0] and os.path.getsize(x[0]) / 1e3 < process_args.archive_min_file_size
                        for x in paths
                    ):
                        continue
                    
                    with open(os.path.join(d_path, tile, 'metadata.json')) as fp:
                        metadata = json.load(fp)

                    if not (archive_min_height <= metadata['range'] < archive_max_height):
                        continue  # Out of bounds

                    for real_path, arc_path in paths:
                        tar.add(real_path, arcname=arc_path)


if __name__ == '__main__':
    main()
