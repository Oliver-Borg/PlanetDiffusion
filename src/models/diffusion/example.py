from dataclasses import dataclass, field
import os
import json

import torch

from .model import TerrainDiffusionPipeline
from .inference import GenerationArguments
from ..shared.image_utils import tensor_to_img, preprocess_image
from ...core.utils import load_image
from ...labelling.encoding import GlobalTerrainStyle
from ...core.dataclass_argparser import CustomArgumentParser

@dataclass
class ExampleArguments(GenerationArguments):
    tile_dir: str = field(
        default='./data/examples/16_24959_10284_3',
        metadata={
            'help': 'Path to input folder'
        }
    )
    num_samples: int = field(
        default=1,
        metadata={
            'help': 'Number of images to generate at a time'
        }
    )


def main():
    parser = CustomArgumentParser(
        (
            ExampleArguments,
        ),
        description='Generate terrain from a sketch'
    )
    example_args, = parser.parse_args_into_dataclasses()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pipeline = TerrainDiffusionPipeline.from_pretrained(
        example_args.model_dir
    ).to(device)

    with open(os.path.join(example_args.tile_dir, 'metadata.json')) as fp:
        metadata = json.load(fp)

    desired_resolution = metadata['resolution']
    desired_range = metadata['range']

    output_dir = os.path.join(example_args.tile_dir, 'outputs')

    # Load sketch
    sketch_image = load_image(
        os.path.join(example_args.tile_dir, 'sketch.png'),
        convert_to='RGBA'
    )
    sketch_tensor = preprocess_image(sketch_image).to(device)

    # Load desired style
    style_image = load_image(os.path.join(example_args.tile_dir, 'style.png'))
    style_tensor = preprocess_image(style_image).to(device)

    # NOTE: Pipeline accepts batches. Expand based on num_samples
    # (c, h, w) -> (b, c, h, w)
    sketch_tensor = torch.cat([sketch_tensor] * example_args.num_samples)
    style_tensor = torch.cat([style_tensor] * example_args.num_samples)
    desired_range = [desired_range] * example_args.num_samples
    desired_resolution = [desired_resolution] * example_args.num_samples

    # Create style
    style = GlobalTerrainStyle(
        terrains=style_tensor,
        ranges=desired_range,
        resolutions=desired_resolution
    )

    with torch.autocast(device_type=device.type, enabled=example_args.use_fp16):
        outputs = pipeline(
            cond_image=sketch_tensor,
            terrain_style=style,
            num_inference_steps=example_args.timesteps,
            guidance_scale=example_args.guidance_scale,
            seed=example_args.seed
        ).images

    outputs = (outputs + 1)/2

    os.makedirs(output_dir, exist_ok=True)
    output_file_name = f'{example_args.seed}_{example_args.timesteps}_{example_args.guidance_scale}'
    for i, output in enumerate(outputs):
        output_path = os.path.join(output_dir, f'{output_file_name}_{i}.png')
        output_image = tensor_to_img(output, bit_depth=16)
        output_image.save(output_path)  # Save image


if __name__ == '__main__':
    main()
