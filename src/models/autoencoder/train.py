
import os
from typing import List, Optional
from dataclasses import dataclass, field, fields

from diffusers.models import (
    AutoencoderKL,
)
from tqdm.auto import trange
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast
from torchvision.utils import make_grid, save_image
import numpy as np
import lpips

from ..shared.sampling import SamplingTrainer, SamplingArguments
from ...core.utils import exists, format_memory
from ...core.terrain_dataset import TerrainDataset
from ...core.dataclass_argparser import CustomArgumentParser
from ...training.trainer import (
    TrainingArguments,
    ModelInputs,
    run_if,
)
from planetAI.src.data.dataset import RAMDataset
from planetAI.src.data.utils import PlanetConfig, open_image_array

import wandb
try:
    wandb.login()
except:
    if 'WANDB_API_KEY' in os.environ:
        wandb.login(key=os.environ['WANDB_API_KEY'])
    else:
        raise Exception("WANDB_API_KEY not found in environment variables. https://wandb.ai/authorize")

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class DatasetArguments:
    dataset_folder: str = field(
        default='./data/processed/',
        metadata={
            'help': 'Folder containing train, valid, and test subfolder'
        }
    )
    image_path: Optional[str] = field(
        default='elevation-256x256.png',
        metadata={
            'help': 'Path to target image inside tile folder'
        }
    )


class AutoEncoderTrainer(SamplingTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Needed since AutoEncoder returns DecoderOutput object
        self.transform_outputs = lambda x: x.sample

    def extract(self, item):
        # Input and output should be the same
        target = item['target_image'].to(self.device)
        mask = torch.ones_like(target)
        return target, target, mask

    @run_if(lambda self: self.steps % self.sampling_args.sample_steps == 0 or self.steps == self.total_train_steps)
    def sample(self):
        self.model.eval()

        dl = iter(self.dataloaders['valid'])

        num_iters = min(self.sampling_args.num_samples, len(self.dataloaders['valid'])
                        ) // self.training_args.batch_size

        images = []
        sampling_info = f'Sampling | epoch={self.epoch:.2f}, steps={self.steps}'

        with (
            trange(num_iters, desc=sampling_info) as progress,
            torch.no_grad(),
            autocast(device_type=self.device.type, enabled=self.training_args.amp)
        ):
            for i in progress:
                model_inputs, targets, masks = self.extract_next(dl)

                outputs = self.model(
                    *model_inputs.args, **model_inputs.kwargs)

                if exists(self.transform_outputs):
                    outputs = self.transform_outputs(outputs)
                outputs = outputs * masks + targets.targets * (1.0 - masks)

                for input, output, target in zip(model_inputs.args[0], outputs, targets.targets):
                    images.extend([input[:3], input[3:4].repeat((3, 1, 1)), output[:3], output[3:4].repeat((3, 1, 1)), target[:3], target[3:4].repeat((3, 1, 1))])

        grid = self.unnormalise(make_grid(images, nrow=6))

        out_dir = os.path.join(
            self.training_args.results_dir, self.sampling_args.samples_dir)
        os.makedirs(out_dir, exist_ok=True)
        self.wandb_metrics.update({'samples': wandb.Image(grid.cpu().numpy().transpose((1, 2, 0)), caption=f'sample-{self.steps}')})
        save_image(grid, os.path.join(out_dir, f'sample-{self.steps}.png'))

@dataclass
class VAETrainingArguments(TrainingArguments):
    results_dir: str = field(
        default='./models/autoencoder/',
        metadata=TrainingArguments.__dataclass_fields__['results_dir'].metadata
    )
    tile_size: int = field(
        default=256,
        metadata={
            'help': 'Size of tiles to train on'
        }
    )
    use_accelerate: bool = field(
        default=True,
        metadata={
            'help': 'Whether to use accelerate for training'
        }
    )
    block_out_channels_factory: List[int] = field(
        default_factory=lambda: {
            2: [256, 512],
            3: [128, 256, 512],
            4: [64, 128, 256, 512],
            5: [64, 128, 256, 256, 512],
            6: [64, 128, 128, 256, 256, 512],
        },
        metadata={
            'help': 'Number of output channels for each block'
        }
    )
    layers_per_block: int = field(
        default=2,
        metadata={
            'help': 'Number of layers per block'
        }
    )
    latent_channels: int = field(
        default=2,
        metadata={
            'help': 'Number of latent channels'
        }
    )
    norm_num_groups: int = field(
        default=32,
        metadata={
            'help': 'Number of groups for normalization layers'
        }
    )
    num_blocks: int = field(
        default=3,
        metadata={
            'help': 'Number of blocks in the encoder and decoder'
        }
    )
    loss_fn: str = field(
        default='mse',
        metadata={
            'help': 'Loss function to use',
            'choices': ['mse', 'mixed']
        }
    )

    def __post_init__(self):
        for f in fields(self):
            k = getattr(self, f.name)
            if isinstance(k, dict):
                setattr(self, f.name[:-len('_factory')], k[self.num_blocks])



def main():
    parser = CustomArgumentParser(
        (
            DatasetArguments,
            VAETrainingArguments,
            SamplingArguments,
            PlanetConfig
        ),
        description='Train an autoencoder'
    )

    (
        dataset_args,
        training_args, 
        sampling_args,
        planet_cfg,
    ) = parser.parse_args_into_dataclasses()

    # Parse arguments and display using logger
    logger.info(f'{" Training with arguments ":*^40}')
    logger.info(dataset_args)
    logger.info(training_args)
    logger.info(sampling_args)
    logger.info('-'*40 + '\n')

    # Get device info and display using logger
    logger.info(f'{" Device information ":*^40}')
    device_count = torch.cuda.device_count()
    logger.info(f'device_count={device_count}')
    if device_count > 1:
        logger.warning('Multiple devices detected, choosing the first.')

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    logger.info(f'CUDA_VISIBLE_DEVICES={visible_devices}')

    # setting device on GPU if available, else CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    if device.type == 'cuda':  # additional info when using cuda
        device_properties = torch.cuda.get_device_properties(device)
        logger.info(f'Device name: {device_properties.name}')
        logger.info(
            f'Total memory: {format_memory(device_properties.total_memory)}')
        logger.info(
            f'Allocated memory: {format_memory(torch.cuda.memory_allocated(device))}')
        logger.info(
            f'Cached memory: {format_memory(torch.cuda.memory_reserved(device))}')

    logger.info('-'*40 + '\n')

    config_dict = {**vars(training_args), **vars(dataset_args), **vars(sampling_args), **vars(planet_cfg)}


    index_mapping = {
        'input': dataset_args.image_path
    }

    block_out_channels=training_args.block_out_channels
    print(f"Model info for {training_args.experiment_name}")
    print(f"Block out channels: {block_out_channels}")
    print(f"Layers per block: {training_args.layers_per_block}")
    print(f"Latent channels: {training_args.latent_channels}")
    print(f"Norm num groups: {training_args.norm_num_groups}")

    model = AutoencoderKL(
        in_channels=4,
        out_channels=4,
        down_block_types=['DownEncoderBlock2D'] * len(block_out_channels),
        up_block_types=['UpDecoderBlock2D'] * len(block_out_channels),
        block_out_channels=block_out_channels,
        layers_per_block=training_args.layers_per_block,
        latent_channels=training_args.latent_channels,
        norm_num_groups=training_args.norm_num_groups,
    )
    test_im = torch.randn(1, 4, 256, 256)
    latent_im = model.encoder(test_im)
    config_dict.update({'latent_shape': latent_im.shape, 
                        'block_out_channels': block_out_channels, 
                        'layers_per_block': training_args.layers_per_block, 
                        'latent_channels': training_args.latent_channels, 
                        'norm_num_groups': training_args.norm_num_groups,
                        'num_params': sum(p.numel() for p in model.parameters() if p.requires_grad)})


    # Move model to device
    model = model.to(device)
    wandb.init(project="PlanetAI", name=training_args.experiment_name, config=config_dict)
    if training_args.experiment_name is not None:
        training_args.results_dir = os.path.join(training_args.results_dir, training_args.experiment_name)

    print(f"Model info for {training_args.experiment_name}")
    print(f"Block out channels: {block_out_channels}")
    print(f"Layers per block: {training_args.layers_per_block}")
    print(f"Latent channels: {training_args.latent_channels}")
    print(f"Norm num groups: {training_args.norm_num_groups}")
    print(f" -> Latent tile shape: {latent_im.shape}")

    shared_dataset_kwargs = dict(
        index_mapping=index_mapping,
        seed=training_args.seed,
    )
    sat_dem = None
    land_temp = None

    if planet_cfg is not None:
        W = 512 * 2 ** planet_cfg.size
        H = W // 2

        dem = open_image_array(os.path.join(planet_cfg.data_dir, f'World_DEM_{W}x{H}.png'))
        sat = open_image_array(os.path.join(planet_cfg.data_dir, f'world.satellite.{W}x{H}.png'))
        sat_dem = np.dstack([sat, dem])
        land = open_image_array(os.path.join(planet_cfg.data_dir, f'World_LandCover_{W}x{H}.png'))
        temp = open_image_array(os.path.join(planet_cfg.data_dir, f'World_Temp_{W}x{H}.png'))
        land_temp = np.dstack([land, temp])


    datasets = dict(
        train=RAMDataset(
            planet_cfg,
            planet_cfg.output_channels(),
            planet_cfg.output_channels(),
            normalise=True,
            tile_size=256,
            mode='train',
            sat_dem=sat_dem,
            land_temp=land_temp,
        ),
        valid=RAMDataset(
            planet_cfg,
            planet_cfg.output_channels(),
            planet_cfg.output_channels(),
            normalise=True,
            tile_size=256,
            mode='val',
            sat_dem=sat_dem,
            land_temp=land_temp,
        ),
        test=RAMDataset(
            planet_cfg,
            planet_cfg.output_channels(),
            planet_cfg.output_channels(),
            normalise=True,
            tile_size=256,
            mode='test',
            sat_dem=sat_dem,
            land_temp=land_temp,
        )
    )

    metrics = []  # TODO add metrics

    # Define optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_args.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-5,  # https://stackoverflow.com/a/46597531/13989043
    )


    
    # Define loss function
    lpips_model = lpips.LPIPS(net='squeeze') # closer to "traditional" perceptual loss, when used for optimization
    # = lpips.LPIPS(net='alex') # best forward scores

    lpips_model = lpips_model.to(device)

    def lpips_fn(x, x_hat):
        _, c, _, _ = x.shape
        losses = []
        for i in range(c):
            a = x[:, i:i+1]
            b = x_hat[:, i:i+1]
            losses.append(lpips_model(a.repeat(1,3,1,1), b.repeat(1,3,1,1)).mean())
        return torch.stack(losses).mean()


    def criterion(a, b):
        if training_args.loss_fn == 'mse':
            return torch.nn.functional.mse_loss(a, b)
        elif training_args.loss_fn == 'mixed':
            return torch.nn.functional.mse_loss(a, b)  + lpips_fn(a, b)

    # Define scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=training_args.lr_scheduler_factor,
        patience=training_args.lr_scheduler_patience,
        threshold=training_args.lr_scheduler_threshold,
        min_lr=training_args.lr_scheduler_min_lr,
        verbose=training_args.lr_scheduler_verbose,
    )

    trainer = AutoEncoderTrainer(
        model=model,
        training_args=training_args,
        sampling_args=sampling_args,

        datasets=datasets,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        metrics=metrics,
    )
    trainer.train()


if __name__ == '__main__':
    main()
