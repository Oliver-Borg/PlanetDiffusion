
import json
from dataclasses import dataclass, field, asdict, replace
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import re
from PIL import Image

from PIL import Image as img

import cv2

from .model import TerrainDiffusionPipeline
from ..shared.image_utils import tensor_to_img
from ...core.dataclass_argparser import CustomArgumentParser
from ...core.terrain_dataset import TerrainDataset
from ...core.terrain_transforms import UnnormaliseTransform
from ...core.utils import array_to_image, load_image, estimate_noise, batched_noise_estimate
from app.sketch_manager import SketchManager, SketchTile

from planetAI.src.data.utils import PlanetConfig, random_masking, np_rgb
from planetAI.src.data.dataset import PlanetDataset
from planetAI.src.data.sketch_gen import paint_img, strahler_paint_river, dilate_paint
from planetAI.src.data.map_paster import get_tile, change_channels
from planetAI.src.data.data_creator import check_empty

from pytorch_CycleGAN_and_pix2pix.inference import call, from_pretrained

import sys

import wandb

from time import time

wandb.login()

@dataclass
class GenerationArguments:
    diffusion_model_dir: str = field(
        default='./models/diffusion/sketch-to-planet/',
        metadata={
            'help': 'Directory of diffusion models'
        }
    )

    gan_model_dir: str = field(
        default='./models/gan/sketch-to-planet/',
        metadata={
            'help': 'Directory of gan models'
        }
    )

    seed: int = field(
        default=None,
        metadata={
            'help': 'Seed for reproducibility'
        }
    )
    timesteps: int = field(
        default=250,
        metadata={
            'help': 'Number of timesteps used for sampling'
        }
    )
    guidance_scale: float = field(
        default=1.0,
        metadata={
            'help': 'Guidance scale for classifier-free guidance'
        }
    )

    use_fp16: bool = field(
        default=False,
        metadata={
            'help': 'Run in fp16 mode'
        }
    )

    enable_attention_slicing: bool = field(
        default=False,
        metadata={
            'help': 'Whether to enable attention slicing (more memory efficient, but slower)'
        }
    )

    attention_slice_size: str = field(
        default='auto',
        metadata={
            'help': 'If `enable_attention_slicing`, use this as the slice_size'
        }
    )

    normalise_outputs: bool = field(
        default=False,
        metadata={
            'help': 'Whether to normalise outputs'
        }
    )





@dataclass
class InferenceArguments(GenerationArguments):
    input_folder: str = field(
        default='./data/evaluation/inputs',
        metadata={
            'help': 'Path to input folder'
        }
    )
    output_folder: str = field(
        default='./data/evaluation/outputs',
        metadata={
            'help': 'Path to output folder'
        }
    )
    batch_size: int = field(
        default=1,
        metadata={
            'help': 'Number of images to generate at a time'
        }
    )

    cond_image: str = field(
        default='sketch-256x256.png',
        metadata={
            'help': 'Conditional image'
        }
    )
    target_image: str = field(
        default='elevation-256x256.png',
        metadata={
            'help': 'Target image'
        }
    )
    scale_factor: int = field(
        default=1,
        metadata={
            'help': 'Scale factor (for conditional image). Used for super-resolution'
        }
    )
    gen_images: bool = field(
        default=False,
        metadata={
            'help': 'Whether to generate images'
        }
    )

    clear_inputs: bool = field(
        default=False,
        metadata={
            'help': 'Whether to clear input folder'
        }
    )

    test_style: bool = field(
        default=False,
        metadata={
            'help': 'Whether to test style'
        }
    )

    compile_model: bool = field(
        default=False,
        metadata={
            'help': 'Whether to compile the model'
        }
    )
    num_workers: int = field(
        default=0,
        metadata={
            'help': 'Number of workers'
        }
    )
    paleo: bool = field(
        default=False,
        metadata={
            'help': 'Whether to use paleo dataset https://doi.org/10.5281/zenodo.5460860'
        }
    )
    # merge_mode: int=0 # 0: none, 1: alpha blend, 2: graph cut, 3: average
    # overlap_mode: int=0 # 0: blank, 1: interpolate, 2: oversize, 3: generate
    # use_poisson: bool=False
    # merge_mode: int = field(
    #     default=None,
    #     metadata={
    #         'help': 'Merge mode to use for tiling (0: none, 1: alpha blend, 2: graph cut, 3: average)'
    #     }
    # )
    overlap_mode: int = field(
        default=None,
        metadata={
            'help': 'Overlap mode to use for tiling (0: blank, 1: interpolate, 2: oversize, 3: generate)'
        }
    )
    # use_poisson: bool = field(
    #     default=None,
    #     metadata={
    #         'help': 'Whether to use poisson blending for tiling'
    #     }
    # )
    find_good_seed: bool = field(
        default=False,
        metadata={
            'help': 'Whether to find a good seed if seed is None'
        }
    )
    unconditional: bool = field(
        default=False,
        metadata={
            'help': 'Whether to use unconditional model'
        }
    )

class FolderDataset(Dataset):
    def __init__(self,
                 folder,
                 index_mapping,
                 ):

        self.folder = folder
        self.index_mapping = index_mapping

        all_files = os.listdir(folder)
        all_files.sort()
        self.tiles = {}
        self.length = None
        for k in index_mapping:
            val = index_mapping[k]
            self.tiles[k] = list(filter(lambda x: re.match(r"^\d+_\d+_\d+_" + val + r"$", x), all_files))
            if self.length is None:
                self.length = len(self.tiles[k])
            else:
                assert self.length == len(self.tiles[k])
                self.length = min(self.length, len(self.tiles[k]))

        self.transform = transforms.ToTensor()
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        to_return = {}
        for key, value in self.index_mapping.items():
            if value is not None:
                tile = self.tiles[key][idx]
                tile_path = os.path.join(self.folder, tile)
                if tile.endswith('.json'):
                    with open(tile_path) as fp:
                        to_return[key] = json.load(fp)
                    continue
                img = Image.open(tile_path)
                img = self.transform(img)*2 - 1
                to_return[key] = img
        return to_return

class SketchManagerDataset(Dataset):
    def __init__(self, sketch_manager: SketchManager, index_mapping, cond_image_channels=1, target_image_channels=1):
        self.sketch_manager = sketch_manager
        self.index_mapping = index_mapping
        self.cond_image_channels = cond_image_channels
        self.target_image_channels = target_image_channels

        self.length = len(sketch_manager.gen_list) + len(sketch_manager.overlap_gen_list)

        print(sketch_manager.gen_list)
        print(sketch_manager.overlap_gen_list)

        self.transform = transforms.ToTensor()

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        to_return = {}
        if idx < len(self.sketch_manager.gen_list):
            sketch_tile: SketchTile = self.sketch_manager.gen_list[idx]
        else:
            sketch_tile: SketchTile = self.sketch_manager.overlap_gen_list[idx - len(self.sketch_manager.gen_list)]
        cond_image = sketch_tile.sketch[:, :, :self.cond_image_channels]
        target_image = sketch_tile.style[:, :, :self.target_image_channels] if len(sketch_tile.style.shape) == 3 else np.dstack([sketch_tile.style]*self.target_image_channels)
        metadata = {
            'tile_y': sketch_tile.y,
            'tile_x': sketch_tile.x,
            'idx': idx,
        }
        if cond_image.dtype != np.uint8 and cond_image.max() <= 1.0:
            cond_image = (cond_image*255).astype(np.uint8)
        if target_image.dtype != np.uint8 and target_image.max() <= 1.0:
            target_image = (target_image*255).astype(np.uint8)
        to_return['cond_image'] = self.transform(cond_image)*2 - 1
        to_return['target_image'] = self.transform(target_image)*2 - 1
        to_return['metadata'] = metadata

        return to_return


def main():
    parser = CustomArgumentParser(
        (
            InferenceArguments,
            PlanetConfig,
        ),
        description='Run inference on a diffusion model'
    )

    (inference_args, planet_cfg) = parser.parse_args_into_dataclasses()


    inference_args.input_folder = os.path.join(inference_args.input_folder, str(planet_cfg.size))
    inference_args.output_folder = os.path.join(inference_args.output_folder, str(planet_cfg.size))

    sys.argv = sys.argv[:1]

    config = {**asdict(inference_args), **asdict(planet_cfg)}

    cond_channels = 1 + int(planet_cfg.size > 3)
    target_channels = 1

    dataset = PlanetDataset(
        planet_cfg, 
        target_image_channels=target_channels, 
        cond_image_channels=cond_channels, 
        conditioning_dropout=0,
        num_configurations=planet_cfg.num_configurations
    )

    dataloader = DataLoader(
        dataset,
        batch_size=inference_args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    os.makedirs(inference_args.input_folder, exist_ok=True)
    generated = len(os.listdir(inference_args.input_folder))//3
    images_wanted = planet_cfg.iters
    # planet_cfg = replace(planet_cfg, iters=planet_cfg.iters - generated if generated < planet_cfg.iters else 0)
    to_pil = transforms.ToPILImage()
    unnormalise_transform = UnnormaliseTransform()
    if inference_args.gen_images and 0 < planet_cfg.iters and not inference_args.paleo:
        if inference_args.clear_inputs:
            for file in os.listdir(inference_args.input_folder):
                os.remove(os.path.join(inference_args.input_folder, file))
        for batch in tqdm(dataloader):
            cond_tensor = batch['cond_image']
            target_tensor = batch['target_image']
            metadata = batch['metadata']
            for i in range(cond_tensor.shape[0]):
                cond_img = cond_tensor[i]
                target_img = target_tensor[i]
                tile_name = f"{metadata['tile_y'][i]}_{metadata['tile_x'][i]}_{metadata['idx'][i]}_"
                tile_path = inference_args.input_folder 
                os.makedirs(tile_path, exist_ok=True)
                to_pil(unnormalise_transform(cond_img)).save(os.path.join(tile_path, tile_name+inference_args.cond_image))
                to_pil(unnormalise_transform(target_img)).save(os.path.join(tile_path, tile_name+inference_args.target_image))
                with open(os.path.join(tile_path, tile_name+'metadata.json'), 'w') as fp:
                    meta = {k: metadata[k][i].item() for k in metadata}
                    json.dump(meta, fp)
    sketch = None
    dem = None
    if inference_args.paleo and inference_args.gen_images:
        paleo_tif_dir = './planetAI/paleo_tifs'
        files = os.listdir(paleo_tif_dir)
        w = 512*2**planet_cfg.size
        h = 256*2**planet_cfg.size
        x_tiles = 2*2**planet_cfg.size
        y_tiles = 2**planet_cfg.size
        i = 0
        coverage = 1.0 - planet_cfg.size/planet_cfg.mask_count
        for file in files:
            if not file.endswith('.tif'):
                continue
            if not file == 'tamriel_dem.tif':
                continue
            im = img.open(os.path.join(paleo_tif_dir, file)) 
            dem = np.array(im)
            dem = cv2.resize(dem, (w, h))
            dem[dem > 0.95] = 0.5
            max_val = np.max(dem)
            min_val = np.min(dem)
            dem = (dem/max_val * 75).astype(np.uint8)
            
            if planet_cfg.sketch_mode == "brush":
                sketch = paint_img(dem, planet_cfg)
            else:
                sketch = dilate_paint(dem, planet_cfg)
            if planet_cfg.use_rivers:
                landmask = (dem > 0).astype(np.uint8)*255
                sketch = np.dstack([sketch, strahler_paint_river(dem, landmask, landmask, planet_cfg)])
            if inference_args.overlap_mode is not None:
                if file == 'tamriel_dem.tif':
                    break
                continue 
            # Diverse evaluation examples
            # 'Map28_PALEOMAP_6min_Early_Cretaceous_125Ma.tif'
            # 'Map36_PALEOMAP_6min_Middle_Jurassic_165Ma.tif'
            # 'Map48_PALEOMAP_6min_Middle_Triassic_245Ma.tif'
            #
            # tamriel_dem.tif
            # westeros_dem.tif
            # middle_earth_dem.tif
            for y in range(y_tiles):
                for x in range(x_tiles):
                    sketch_tile = get_tile(sketch, 256*y, 256*x)
                    dem_tile = get_tile(dem, 256*y, 256*x)

                    if check_empty(dem_tile, coverage):
                        continue
                    if planet_cfg.image_mode == 'dem-inpainting':
                        inpainting = (random_masking(
                            dem_tile/127.5 - 1, mode=planet_cfg.inpainting_mode
                        )*127.5 + 127.5).astype(np.uint8)
                        sketch_tile = np.dstack([inpainting, sketch_tile])
                    tile_name = f"{y}_{x}_{i}_"
                    metadata = {
                        'range': int(np.max(dem_tile) - np.min(dem_tile)),
                        'zoom': 2**planet_cfg.size,
                        'tile_y': y,
                        'tile_x': x,
                        'factor': planet_cfg.size,
                        'resolution': 84375.0/(2**planet_cfg.size),
                        'idx': i,
                    }
                    tile_path = inference_args.input_folder
                    os.makedirs(tile_path, exist_ok=True)
                    img.fromarray(sketch_tile).save(os.path.join(tile_path, tile_name+inference_args.cond_image))
                    img.fromarray(dem_tile).save(os.path.join(tile_path, tile_name+inference_args.target_image))
                    with open(os.path.join(tile_path, tile_name+'metadata.json'), 'w') as fp:
                        json.dump(metadata, fp)
                    planet_cfg.randomize_parameters()
                    i += 1
                


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    output_file_name = f'diffusion_output_{inference_args.seed}_{inference_args.timesteps}_{inference_args.guidance_scale}.png'

    # (TODO make cli args)
    index_mapping = {}
    index_mapping['cond_image'] = inference_args.cond_image
    index_mapping['target_image'] = inference_args.target_image
    index_mapping['metadata'] = 'metadata.json'

    sketch_manager = SketchManager(
        size=planet_cfg.size,
        kwargs={
            'overlap_mode': inference_args.overlap_mode,
            'selected_x': 0,
            'selected_y': 0,
            'overlap_width': 64,
            'tile_border_size': 16,
            'automatic_style': True,
        }
    ) if inference_args.overlap_mode is not None else None

    if dem is None:
        dem = (sketch_manager.style*255).astype(np.uint8)
    if sketch is None:
        if planet_cfg.sketch_mode == "brush":
            sketch = paint_img(dem, planet_cfg)
        else:
            sketch = dilate_paint(dem, planet_cfg)
        if planet_cfg.use_rivers:
            landmask = (dem > 0).astype(np.uint8)*255
            sketch = np.dstack([sketch, strahler_paint_river(dem, landmask, landmask, planet_cfg)])
    if sketch_manager is not None:
        sketch_manager.style = dem.astype(np.float32)/255
        sketch_manager.set_sketch_window(change_channels(sketch, 3)/255)
        sketch_manager.dem = np.zeros_like(sketch_manager.dem)
        img.fromarray(
            (
                sketch_manager.style*255
            ).astype(np.uint8)
        ).save(os.path.join(inference_args.output_folder, 'target.png'))
        img.fromarray(
            (
                sketch_manager.sketch*255
            ).astype(np.uint8)
        ).save(os.path.join(inference_args.output_folder, 'sketch.png'))

    test_dataset = FolderDataset(
        folder=inference_args.input_folder,
        index_mapping=index_mapping,
    ) if inference_args.overlap_mode is None else SketchManagerDataset(
        sketch_manager=sketch_manager,
        index_mapping=index_mapping,
        cond_image_channels=cond_channels,
        target_image_channels=target_channels,
    )

    all_sketch_tiles = []
    if sketch_manager is not None:
        all_sketch_tiles = sketch_manager.gen_list + sketch_manager.overlap_gen_list

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=inference_args.batch_size,
        shuffle=True,
        num_workers=0,
        persistent_workers=False,
        generator=torch.Generator().manual_seed(42)
    )
    name = f'Zoom {planet_cfg.size}'

    diffusion_model_path = os.path.join(inference_args.diffusion_model_dir, name, 'final')

    pipeline = TerrainDiffusionPipeline.from_pretrained(
        diffusion_model_path
    )
    cgan_model_path = inference_args.gan_model_dir

    cgan_model = from_pretrained(cgan_model_path, name, input_nc=1+int(planet_cfg.size>2))
    # Move pipeline to device
    pipeline = pipeline.to(device)

    if inference_args.compile_model:
        pipeline.unet = torch.compile(pipeline.unet)

    if inference_args.enable_attention_slicing:
        pipeline.enable_attention_slicing(
            slice_size=inference_args.attention_slice_size
        )

    # # TODO add option to switch out scheduler
    # # DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models
    # from diffusers import DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler
    # pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    diffusion_timings = []
    gan_timings = []
    total_iters = 0
    
    
    table = None
    first_cond = None
    first_target = None
    seed = inference_args.seed
    if inference_args.find_good_seed and inference_args.seed is None:
        seed = 0
    for batch in tqdm(test_dataloader):
        if total_iters > images_wanted:
            break
        total_iters += inference_args.batch_size
        target_image = batch['target_image'].to(device)
        cond_image = batch['cond_image'].to(device)
        if inference_args.unconditional:
            cond_image[cond_image > -1.0] = -1.0
        style_image = batch['target_image'].to(device)
        if inference_args.test_style:
            if first_cond is None:
                first_cond = cond_image
            cond_image = first_cond
            if first_target is None:
                first_target = target_image
            target_image = first_target

        # Reshape metadata
        all_metadata = []
        for i in range(target_image.shape[0]):
            item = {}
            for k in batch['metadata']:
                item[k] = batch['metadata'][k][i].item()
            all_metadata.append(item)

        i = 0
        for metadata in all_metadata:
            tile_name = f"{metadata['tile_y']}_{metadata['tile_x']}_{metadata['idx']}_"
            gan_output_file_name = output_file_name.replace('diffusion', 'gan')
            if os.path.exists(os.path.join(inference_args.output_folder, tile_name+output_file_name)):
                # Remove from batch
                cond_image = torch.cat([cond_image[:i], cond_image[i+1:]])
                target_image = torch.cat([target_image[:i], target_image[i+1:]])
                style_image = torch.cat([style_image[:i], style_image[i+1:]])
                for k in batch['metadata']:
                    batch['metadata'][k] = torch.cat([batch['metadata'][k][:i] + batch['metadata'][k][i+1:]])
                i -= 1
            i += 1
        if i == 0:
            continue



        # Resize conditional image if needed
        if inference_args.scale_factor > 1:
            cond_image = F.interpolate(
                cond_image, scale_factor=inference_args.scale_factor)

        # Specify desired terrain style
        # In this case, we want it in the style of the target image
        terrain_style = pipeline.create_terrain_style(
            style_image, style_image, batch['metadata'])

        # Generate terrain from sketch, in the style of the target terrain
        mse = None
        mse_th = 0.01
        steps = 0
        while mse is None or (mse > mse_th and inference_args.find_good_seed and inference_args.seed is None):
            t1 = time()
            non_oceans = torch.where(cond_image > -1.0)
            if len(non_oceans[0]) == 0:
                diffusion_outputs = cond_image.clone()
            else:
                with autocast(device_type=device.type, enabled=inference_args.use_fp16):
                    diffusion_outputs = pipeline(
                        cond_image=cond_image,
                        terrain_style=terrain_style,
                        num_inference_steps=inference_args.timesteps,
                        guidance_scale=inference_args.guidance_scale,
                        seed=seed,
                        normalise_output=inference_args.normalise_outputs,
                    ).images
            
            outputs_copy = diffusion_outputs.clone()
            outputs_copy[non_oceans] = cond_image[non_oceans]
            num_ocean_pixels = len(torch.where(cond_image == -1.0)[0])
            total_pixels = len(cond_image.flatten())
            mse = F.mse_loss(outputs_copy, cond_image).item()
            if num_ocean_pixels > 0:
                mse = mse*total_pixels/num_ocean_pixels
                mse_th = 0.01
            else:
                num_colours = len(planet_cfg.colour_list())
                mse = F.mse_loss(diffusion_outputs, cond_image).item()*num_colours/4
                mse_th = 0.1
            if inference_args.unconditional:
                mse = batched_noise_estimate(diffusion_outputs).item()
            if mse > mse_th and seed is not None:
                seed += 1
            steps += 1
            if steps > 10:
                seed = 0
                break
            t2 = time()
        print(f"Seed: {seed}")
        gan_cond_image = cond_image.clone()
        gan_input_nc = 1
        if planet_cfg.size > 2:
            # Expand shape[1] from 1 to 2
            gan_input_nc = 2
            gan_cond_image = torch.cat([gan_cond_image, torch.zeros_like(gan_cond_image)-1], dim=1)
        gan_outputs = call(cgan_model, gan_cond_image, input_nc=gan_input_nc)
        t3 = time()
        diffusion_timings.append(t2-t1)
        gan_timings.append(t3-t2)

        if sketch_manager is not None:
            sketch_tiles = []
            for metadata in all_metadata:
                sketch_tiles.append(all_sketch_tiles[metadata['idx']]) 
            to_add = list((diffusion_outputs.cpu().numpy()+1)/2)
            to_add = [x[0, :, :] for x in to_add]
            sketch_manager.add_dem_tiles(to_add, sketch_tiles)

        target_image = unnormalise_transform(target_image)
        cond_image = unnormalise_transform(cond_image)
        diffusion_outputs = unnormalise_transform(diffusion_outputs)
        gan_outputs = unnormalise_transform(gan_outputs)
        style_image = unnormalise_transform(style_image)

        

        for metadata, cond_img, diffusion_output_img, gan_output_img, target_img, style_img in zip(all_metadata, cond_image, diffusion_outputs, gan_outputs, target_image, style_image):
            tile_name = f"{metadata['tile_y']}_{metadata['tile_x']}_{metadata['idx']}_"

            cond_img = to_pil(cond_img)
            diffusion_output_img = to_pil(diffusion_output_img)
            gan_output_img = to_pil(gan_output_img)
            target_img = to_pil(target_img)
            style_img = to_pil(style_img)

            tile_path = inference_args.output_folder
            os.makedirs(tile_path, exist_ok=True)
            gan_output_file_name = output_file_name.replace('diffusion', 'gan')
            cond_img.save(os.path.join(tile_path, tile_name+'input.png'))
            diffusion_output_img.save(os.path.join(tile_path, tile_name+output_file_name))
            gan_output_img.save(os.path.join(tile_path, tile_name+gan_output_file_name))
            target_img.save(os.path.join(tile_path, tile_name+'target.png'))
            style_img.save(os.path.join(tile_path, tile_name+'style.png'))

            
            data = [cond_img, diffusion_output_img, gan_output_img, target_img, style_img]
            captions = ["Input", "Diffusion Output", "GAN Output", "Target", "Style"]
            images = []
            image_captions = []
            for im, caption in zip(data, captions):
                im = np.array(im)
                if len(im.shape) == 2:
                    images.append(wandb.Image(img.fromarray(np_rgb(im, 'Greys')), caption=caption))
                    image_captions.append(caption)
                else:
                    for i in range(im.shape[2]):
                        images.append(wandb.Image(img.fromarray(np_rgb(im[:,:,i], 'Greys')), caption=f"{caption} {i}"))
                        image_captions.append(f"{caption} {i}")

            if table is None:
                table = wandb.Table(columns=["name", "noise"]+image_captions)


            # Estimate noise
            noise = estimate_noise(np.array(diffusion_output_img))
            table.add_data(
                tile_name,
                noise,
                *images
            )
            metadata['noise'] = noise
            with open(os.path.join(tile_path, tile_name+output_file_name+'metadata.json'), 'w') as fp:
                json.dump(metadata, fp)
            with open(os.path.join(tile_path, tile_name+gan_output_file_name+'metadata.json'), 'w') as fp:
                json.dump(metadata, fp)
    
    if sketch_manager is not None:
        sketch_manager.save_dem()
        for merge_mode in [0, 1, 2]:
            for use_poisson in [False, True]:
                sketch_manager.merge_mode = merge_mode
                sketch_manager.use_poisson = use_poisson
                full_sized = (sketch_manager.do_full_merge()*255).astype(np.uint8)
                name = f'Zoom_{planet_cfg.size}_Full_{inference_args.overlap_mode}_{merge_mode}_{use_poisson}.png'
                img.fromarray(full_sized).save(os.path.join(inference_args.output_folder, name))
    
    run_name = f"Zoom {planet_cfg.size} Seed {inference_args.seed}  Steps {inference_args.timesteps} Guidance {inference_args.guidance_scale} AMP {str(inference_args.use_fp16)[0]}"
    with wandb.init(project="planetAI-inference", config=config, name=run_name) as run:
        run.log({"table": table})
    print("Timesteps: ", inference_args.timesteps)
    print("Size: ", planet_cfg.size)
    print("Compile model: ", inference_args.compile_model)
    print("AMP: ", inference_args.use_fp16)
    num_params = 0
    for param in pipeline.unet.parameters():
        num_params += param.numel()
    print(f'Diffusion number of parameters: {num_params/1e6:.2f}M')
    num_params = 0
    for param in cgan_model.netG.parameters():
        num_params += param.numel()
    print(f'GAN number of parameters: {num_params/1e6:.2f}M')

    if inference_args.compile_model:
        compile_time = diffusion_timings[0] - np.mean(diffusion_timings[1:])
        print(f"Compile time: {compile_time}")
        diffusion_timings[0] -= compile_time
    print(f"Diffusion time: {np.mean(diffusion_timings)}")
    print(f"Diffusion time per timestep: {np.mean(diffusion_timings)/inference_args.timesteps}")
    print(f"GAN time: {np.mean(gan_timings)}")

if __name__ == '__main__':
    main()
