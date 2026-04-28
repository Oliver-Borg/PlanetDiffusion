import os
from dataclasses import dataclass, field
import shutil
import hashlib

from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms as transforms

from .classes import TYPE_MAPPINGS
from ..utils import load_model
from ...core.dataclass_argparser import CustomArgumentParser
from ...core.utils import list_all_files, chunks, get_tiles_list
from ...core.terrain_transforms import NormaliseTransform


# Used to prevent mode collapse and filter out missing images
SINGLE_COLOUR_SIZE = 1652
MIN_SIZE = 2048

# Detect when image is not available
NOT_AVAILABLE_SIZE = 2418
NOT_AVAILABLE_HASH = '83151db1dd3da72dccf70eac385aace338df6b1a439f66fa735cf93c272aeb78'


@dataclass
class ClassifierArguments:

    classifiers_dir: str = field(
        default='./models/classifiers',
        metadata={
            'help': 'The directory to store the classifiers'
        }
    )

    classifier_type: str = field(
        default='derivative',
        metadata={
            'choices': TYPE_MAPPINGS.keys(),
            'help': 'Satellite or derivative classifier'
        }
    )

    # Filtering options:
    data_dir: str = field(
        default='data',
        metadata={
            'help': 'Directory to store data'
        }
    )

    training_dir: str = field(
        default='moderated',
        metadata={
            'help': 'Directory to store data for training'
        }
    )


@dataclass
class FilteringArguments(ClassifierArguments):
    dir_to_scan: str = field(
        default='raw',
        metadata={
            'help': 'Directory (within data_dir/classifier_type) to filter'
        }
    )
    move_to_dir: str = field(
        default='ml_moderated',
        metadata={
            'help': 'Directory (within data_dir/classifier_type) to move filtered items to'
        }
    )
    prefilter: str = field(
        default=False,
        metadata={
            'help': 'Whether to perform prefiltering (False = skip, 1 = based on size, 2 = based on other classifier, or both)'
        }
    )


def main():

    parser = CustomArgumentParser(
        FilteringArguments,
        description='Perform filtering using a pretrained classifier'
    )

    filter_args,  = parser.parse_args_into_dataclasses()

    c_dir = os.path.join(filter_args.data_dir,
                         filter_args.classifier_type)
    t_dir = os.path.join(c_dir, filter_args.training_dir)

    dir_to_scan = os.path.join(c_dir, filter_args.dir_to_scan)
    move_to_dir = os.path.join(c_dir, filter_args.move_to_dir)

    is_sat = filter_args.classifier_type == 'satellite'

    c_type, num_channels, img_ext = TYPE_MAPPINGS[filter_args.classifier_type]

    idx_to_class = c_type.idx_to_class()

    prefilter_pass_1 = filter_args.prefilter in ('1', 'both')
    prefilter_pass_2 = filter_args.prefilter in ('2', 'both')

    if prefilter_pass_1:
        # If satellite, first filter out "Map data not yet available" images
        print('Prefiltering')

        files = list(list_all_files(dir_to_scan, img_ext, True))

        moved_count = 0
        moved_progress = tqdm(files)
        for d, f in moved_progress:
            move_dir = None
            path = os.path.join(d, f)
            f_size = os.path.getsize(path)
            if is_sat and f_size == NOT_AVAILABLE_SIZE:
                # Must read to calculate hash
                with open(path, 'rb') as fp:
                    data = fp.read()

                hash = hashlib.sha256(data).hexdigest()

                if hash == NOT_AVAILABLE_HASH:
                    move_dir = os.path.join(
                        t_dir, 'ignored', 'not_available')

            if move_dir is None:
                # hash didn't match
                if is_sat and f_size == SINGLE_COLOUR_SIZE:
                    # min size (single colour)
                    move_dir = os.path.join(
                        t_dir, 'ignored', 'single_colour')
                elif f_size <= MIN_SIZE:
                    move_dir = os.path.join(
                        t_dir, 'ignored', 'below_min_size')

            if move_dir is not None and not os.path.samefile(d, move_dir):
                # Do not move, or not already in the folder
                os.makedirs(move_dir, exist_ok=True)
                try:
                    shutil.move(path, move_dir)
                except shutil.Error:
                    # File exists in new dir, delete it
                    os.remove(path)

                moved_count += 1

            moved_progress.set_description(
                f'Processed {f} (moved {moved_count})')

    if prefilter_pass_2:
        # Prefilter based on other's classifier
        move = 'derivative' if is_sat else 'satellite'
        c_class, _, ext = TYPE_MAPPINGS[move]

        # Move derivative tiles that correspond to invalid satellite images
        prefilter_move_dir = os.path.join(
            filter_args.data_dir, move)

        invalid_tiles = {}
        invalid_classes = c_class.invalid()
        for t in ('train', 'test', 'to_divide'):
            d = os.path.join(prefilter_move_dir,
                             filter_args.training_dir, t)

            for invalid_class in invalid_classes:
                d1 = os.path.join(d, invalid_class.name)

                for i in get_tiles_list(d1, ext, recursive=False):
                    invalid_tiles[i] = invalid_class.name

        other_ignored_files = get_tiles_list(os.path.join(
            prefilter_move_dir, filter_args.training_dir, 'ignored'), ext)
        for i in other_ignored_files:
            invalid_tiles[i] = 'OTHER'

        files = list(list_all_files(dir_to_scan, img_ext, True))
        print('Moving files that are ignored by other classifier')
        moved_count = 0
        moved_progress = tqdm(files)
        for d, f in moved_progress:
            tile = f.split('.')[0]
            if tile in invalid_tiles:
                move_dir = os.path.join(
                    t_dir, 'ignored', move, invalid_tiles[tile])

                os.makedirs(move_dir, exist_ok=True)
                if not os.path.samefile(d, move_dir):
                    # Not already in the folder
                    shutil.move(os.path.join(d, f), move_dir)
                    moved_count += 1

            moved_progress.set_description(
                f'Processed {f} (moved {moved_count})')

    model_dir = os.path.join(
        filter_args.classifiers_dir, filter_args.classifier_type)

    model = load_model(model_dir)
    model = model.cuda()

    # Do some filtering
    transform = transforms.Compose([
        transforms.ToTensor(),
        NormaliseTransform()
    ])

    # Get files again
    files = list(list_all_files(dir_to_scan, img_ext, True))
    # random.shuffle(files)

    batch_size = 32  # 64
    chunked_files = list(chunks(files, batch_size))

    for chunk in tqdm(chunked_files):

        paths = list(map(lambda x: os.path.join(*x), chunk))

        # Create batch
        imgs = torch.stack(
            list(map(lambda x: transform(Image.open(x)), paths))).cuda()

        with torch.no_grad():
            outputs = model(imgs)

        _, predictions = torch.max(outputs, dim=1)

        for path, pred in zip(paths, predictions):
            class_name = idx_to_class[pred.item()]

            pdir = os.path.join(move_to_dir, class_name)
            os.makedirs(pdir, exist_ok=True)
            if not os.path.samefile(move_to_dir, pdir):
                try:
                    shutil.move(path, pdir)
                except shutil.Error:
                    # File exists in new dir, delete it
                    os.remove(path)


if __name__ == '__main__':
    main()
