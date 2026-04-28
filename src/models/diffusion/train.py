
import os
import re
from typing import Callable, List, Optional, Union
from dataclasses import dataclass, field, fields, replace

from diffusers.models import (
    UNet2DConditionModel,
    UNet2DModel,
    AutoencoderKL,
    ControlNetModel
)
from diffusers.schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
)
from .control import load_controlnet_and_model, ControlNetTrainWrapper
from tqdm.auto import trange
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from diffusers.pipelines import DiffusionPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.amp import autocast
from torchvision.utils import make_grid, save_image
import numpy as np
from PIL import Image as img

import lpips

from .model import TerrainDiffusionPipeline, exists
from ..shared.sampling import SamplingTrainer, SamplingArguments
from ...core.utils import (
    exists,
    tile_images,
    format_memory,
    estimate_noise, 
    batched_noise_estimate,
    NoiseLoss
)
from ...core.terrain_dataset import TerrainDataset
from planetAI.src.data.dataset import RAMDataset
from planetAI.src.data.utils import PlanetConfig
from planetAI.src.data.map_paster import setup
from ...core.dataclass_argparser import CustomArgumentParser
from ...training.trainer import (
    TrainingArguments,
    ModelInputs,
    run_if,
)
from ...training.metrics import TrainingMetrics
from ...labelling.encoding import (
    GlobalTerrainEncoder,
    SatelliteTerrainEncoder,
    PlanetEncoder,
)

from torchvision.transforms import ToPILImage

import wandb
if 'WANDB_API_KEY' in os.environ:
    wandb.login(key=os.environ['WANDB_API_KEY'])
else:
    raise Exception("WANDB_API_KEY not found in environment variables. https://wandb.ai/authorize")
import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class ImageArguments:
    target_image_channels: int = field(
        default=None,
        metadata={
            'help': 'Number of channels in the target image (in pixel space)'
        }
    )
    cond_image_channels: int = field(
        default=None,  # TODO
        metadata={
            'help': 'Number of channels in the conditional image'
        }
    )

    target_image_type: str = field(
        default='planet',
        metadata={
            'choices': ['elevation', 'satellite', 'planet'],
            'help': 'Type of image to generate',
        }
    )

    def update_channels(self, planet_cfg: PlanetConfig):
        if self.target_image_channels is None:
            self.target_image_channels = planet_cfg.output_channels()
        if self.cond_image_channels is None:
            self.cond_image_channels = planet_cfg.input_channels()


@dataclass
class DiffusionArguments:
    # Default params based on:
    # https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/unet/config.json

    unet_block_out_channels_factory: List[int] = field(
        default_factory=lambda: {
            'none': [
                64, 128, 256, 512
            ],
            'attn': [
                64, 128, 256, 512
            ],
            'crossattn': [
                64, 128, 256, 512
            ],
            'test': [
                128, 128, 256, 256, 512, 512
            ]
        },
            
        metadata={
            'help': 'List of block output channels',
            'nargs': '+'
        }
    )
    unet_down_block_types_factory: List[str] = field(
        default_factory=lambda: {
            'none': [
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
            ],
            'attn': [
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ],
            'crossattn': [
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ],
            'test': [
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "DownBlock2D",
            ]
        },
        metadata={
            'help': 'List of downsample block types.',
            'nargs': '+'
        }
    )
    unet_up_block_types_factory: List[str] = field(
        default_factory=lambda: {
            'none': [
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ],
            'attn': [
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ],
            'crossattn': [
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
            ],
            'test': [
                "UpBlock2D",  # a regular ResNet upsampling block
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ]

        },
        metadata={
            'help': 'List of upsample block types.',
            'nargs': '+'
        }
    )
    unet_layers_per_block: int = field(
        default=3,
        metadata={
            'help': 'The number of layers per block.'
        }
    )
    unet_attention_head_dim: int = field(
        default=8,
        metadata={
            'help': 'The attention head dimension.'
        }
    )

    unet_use_embeddings: bool = field(
        default=True,
        metadata={
            'help': 'Whether to further condition denoising on embedding.'
        }
    )

    unet_attn_type: str = field(
        default='none',
        metadata={
            'choices': ['none', 'attn', 'crossattn', 'test'],
            'help': 'Type of attention to use',
        }
    )

    unet_norm_eps: float = field(
        default=0.01,
        metadata={
            'help': 'The epsilon value for the normalization layers.'
        }
    )

    compile_unet: bool = field(
        default=False,
        metadata={
            'help': 'Whether to compile the UNet model for faster training.'
        }
    )

    latent_diffusion: bool = field(
        default=False,
        metadata={
            'help': 'Whether to use latent diffusion.'
        }
    )

    auto_encoder_model_path: Optional[str] = field(
        default=None,
        metadata={
            'help': 'Path to the autoencoder model.'
        }
    )

    sat_latent_scaling_factor: float = field(
        default=0.19472,
        metadata={
            'help': 'The scaling factor for the satellite image latent space.'
        }
    )
    dem_latent_scaling_factor: float = field(
        default=0.14028,
        metadata={
            'help': 'The scaling factor for the DEM latent space.'
        }
    )

    def __post_init__(self):
        for f in fields(self):
            k = getattr(self, f.name)
            if isinstance(k, dict):
                setattr(self, f.name[:-len('_factory')], k[self.unet_attn_type])


@dataclass
class SchedulerArguments:
    # TODO add support for other scheduler params

    num_train_timesteps: int = field(
        default=1000,
        metadata={
            'help': 'Number of diffusion steps used to train the model.'
        }
    )
    num_inference_steps: int = field(
        default=100,
        metadata={
            'help': 'Number of diffusion steps used when generating samples.'
        }
    )
    eta: float = field(
        default=1.0,
        metadata={
            'help': 'Corresponds to parameter eta (η) in the DDIM paper. Only applies to'
                    'DDIMScheduler, will be ignored for others.'
        }
    )
    beta_schedule: str = field(
        default='linear',
        metadata={
            'choices': ['linear', 'cosine'],
            'help': 'Type of noise scheduler to use.'
        }
    )
    beta_start: float = field(
        default=1e-4,
        metadata={
            'help': 'the starting `beta` value of inference.'
        }
    )
    beta_end: float = field(
        default=2e-2,
        metadata={
            'help': 'the final `beta` value.'
        }
    )
    clip_sample: bool = field(
        default=True,
        metadata={
            'help': 'option to clip predicted sample for numerical stability.'
        }
    )
    # TODO: Fix other schedulers. Currently only DDIM and DDPM work.
    noise_scheduler: str = field(
        default='DDIMScheduler',
        metadata={
            'choices': ['DDIMScheduler', 'DDPMScheduler', 'PNDMScheduler', 'LMSDiscreteScheduler', 'EulerDiscreteScheduler',
                        'EulerAncestralDiscreteScheduler'],
            'help': 'Type of noise scheduler to use.'
        }
    )


@dataclass
class DiffusionTrainingArguments(TrainingArguments):
    results_dir: str = field(
        default='./models/diffusion',
        metadata=TrainingArguments.__dataclass_fields__['results_dir'].metadata
    )
    use_controlnet: bool = field(
        default=False,
        metadata={
            'help': 'Whether to train a ControlNet adapter instead of the full U-Net.'
        }
    )


@dataclass
class DatasetArguments:
    dataset_folder: str = field(
        default='./data/processed/',
        metadata={
            'help': 'Folder containing train, valid, and test subfolder'
        }
    )
    target_image: Optional[str] = field(
        default='elevation.png',
        metadata={
            'help': 'Path to target image inside tile folder'
        }
    )
    cond_image: Optional[str] = field(
        default="sketch.png",  # 'elevation-64x64.png'
        metadata={
            'help': 'Path to conditional image inside tile folder. If None, train unconditionally'
        }
    )
    setup: Optional[bool] = field(
        default=False,
        metadata={
            'help': 'Setup the directories and generate the masks'
        }
    )


    




class DiffusionModelTrainer(SamplingTrainer):
    def __init__(self,
                 model: TerrainDiffusionPipeline,
                 generator: Optional[torch.Generator] = None,
                 scheduler_args: Optional[SchedulerArguments] = None,
                 planet_cfg: Optional[PlanetConfig] = None,
                 auto_encoder: Optional[AutoencoderKL] = None,
                 sat_latent_scaling_factor: float = 0.19472,
                 dem_latent_scaling_factor: float = 0.14028,
                 *args, **kwargs):

        self.pipeline = model
        self.planet_cfg = planet_cfg
        self.auto_encoder = auto_encoder
        self.sat_latent_scaling_factor = sat_latent_scaling_factor or 1.0
        self.dem_latent_scaling_factor = dem_latent_scaling_factor or 1.0

        # We set the U-Net to be the model to train
        from diffusers.configuration_utils import FrozenDict
        from torch.nn.modules.conv import Conv2d
        from diffusers.models.embeddings import Timesteps, TimestepEmbedding
        from torch.nn.modules.linear import Linear
        from torch.nn.modules.activation import SiLU
        from torch.nn.modules.container import ModuleList
        from diffusers.models.unets.unet_2d_blocks import (
            DownBlock2D, UpBlock2D, UNetMidBlock2D, UNetMidBlock2DCrossAttn, UNetMidBlock2DSimpleCrossAttn
        )
        import sys
        import diffusers.models.unets.unet_2d_condition
        import diffusers.models.unets.unet_2d_blocks
        from diffusers.models.transformers.transformer_2d import Transformer2DModel
        # Shim: Redirect the old module path to the new location
        sys.modules["diffusers.models.transformer_2d"] = diffusers.models.transformers.transformer_2d
        sys.modules["diffusers.models.unet_2d_condition"] = diffusers.models.unets.unet_2d_condition
        sys.modules["diffusers.models.unet_2d_blocks"] = diffusers.models.unets.unet_2d_blocks

        from diffusers.models.resnet import ResnetBlock2D, Downsample2D, Upsample2D
        from torch.nn.modules.normalization import GroupNorm, LayerNorm
        from torch.nn.modules.dropout import Dropout
        from diffusers.models.attention import BasicTransformerBlock, FeedForward, GEGLU, GELU, ApproximateGELU
        from diffusers.models.attention_processor import Attention, SlicedAttnProcessor, AttnProcessor2_0
        from builtins import getattr
        torch.serialization.add_safe_globals([
            UNet2DConditionModel, FrozenDict, Conv2d, Timesteps, TimestepEmbedding,
            Linear, SiLU, ModuleList, DownBlock2D, UpBlock2D, ResnetBlock2D, GroupNorm,
            Dropout, Downsample2D, Upsample2D, UNetMidBlock2D, UNetMidBlock2DCrossAttn,
            UNetMidBlock2DSimpleCrossAttn, Transformer2DModel, BasicTransformerBlock,
            LayerNorm, Attention, SlicedAttnProcessor, AttnProcessor2_0, FeedForward,
            GEGLU, GELU, ApproximateGELU, getattr
        ])

        # Determine if we are training ControlNet or Standard U-Net
        self.controlnet = getattr(model, 'controlnet', None)
        if self.controlnet:
            training_model = ControlNetTrainWrapper(model.unet, self.controlnet)
            logger.info("Initializing Trainer with ControlNet Wrapper (Frozen U-Net + Trainable ControlNet)")
        else:
            training_model = model.unet

        super().__init__(model=training_model, pipeline=model, *args, **kwargs)
        # Easier accessing
        self.terrain_encoder = model.terrain_encoder
        self.noise_scheduler = model.scheduler

        self.generator = generator

        if scheduler_args is None:
            scheduler_args = SchedulerArguments()
        self.scheduler_args = scheduler_args

        # Needed since UNet2DModel returns UNet2DOutput object
        self.transform_outputs = lambda x: x.sample

        self.latents_dtype = next(self.model.parameters()).dtype

    def train(self):
        # Run normal training loop
        super().train()
        if self.training_args.save_on_disk:
            # Save diffusers pipeline after training
            final_dir = os.path.join(self.training_args.results_dir, 'final')
            self.pipeline.save_pretrained(final_dir)

    def extract(self, item):

        # 1. Extract images
        target_image = item['target_image'].to(self.device)
        batch_size = target_image.shape[0]

        # 1.5 (optional) Encode target image into its latent space
        if self.auto_encoder is not None:
            y = target_image
            if self.auto_encoder.config['in_channels'] == 3:
                sat_targets = y[:, 0:3]
                dem_targets = y[:, 3:4].repeat(1, 3, 1, 1)
                # I might need different scaling factors for each latent space
                sat_latents = self.auto_encoder.encode(sat_targets).latent_dist.sample() * self.sat_latent_scaling_factor
                dem_latents = self.auto_encoder.encode(dem_targets).latent_dist.sample() * self.dem_latent_scaling_factor
                latent = torch.cat([sat_latents, dem_latents], dim=1)
            elif self.auto_encoder.config['in_channels'] == 4:
                latent = self.auto_encoder.encode(y).latent_dist.sample() * (self.sat_latent_scaling_factor + self.dem_latent_scaling_factor)/2
            else:
                raise ValueError('Autoencoder must have 3 or 4 input channels')

            target_image = latent

        # 2. Sample noise that we'll add to the target_image
        noise = torch.randn(
            target_image.shape,
            generator=self.generator,
            dtype=target_image.dtype,
            device=target_image.device
        )

        # 3. Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.device,
            generator=self.generator,
        ).long()

        # 4. Add noise to the target latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_target_image = self.noise_scheduler.add_noise(
            target_image, noise, timesteps)
        # Get next noisy target image for inpainting
        zero_indices = torch.where(timesteps == 0)
        next_noisy_target_image = noisy_target_image.clone()
        next_noisy_target_image[zero_indices] = target_image[zero_indices]
        nonzero_indices = torch.where(timesteps > 0)
        next_noisy_target_image[nonzero_indices] = self.noise_scheduler.add_noise(
            target_image[nonzero_indices], noise[nonzero_indices], timesteps[nonzero_indices]-1)

        
        cond_image = None
        inputs = [noisy_target_image]
        masks = None
        if exists(item.get('cond_image')):
            cond_image = item['cond_image'].to(self.device)
            if cond_image.shape[-2:] != noisy_target_image.shape[-2:]:
                # Resize conditional image if needed
                cond_image = F.interpolate(
                    cond_image,
                    size=noisy_target_image.shape[-2:]
                )
            mask_is = item['metadata'].get('mask_channel')
            if exists(mask_is):
                masks = torch.cat([cond_image[i, mask_i, :, :].unsqueeze(0) for i, mask_i in enumerate(mask_is)])
                cond_image = torch.cat([cond_image[i, mask_i+1:, :, :].unsqueeze(0) for i, mask_i in enumerate(mask_is)])
                masks += 1.0
                masks /= 2.0
                channels = target_image.shape[1]
                masks = masks.unsqueeze(1)
                masks = masks.repeat(1, channels, 1, 1)
                # noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy*mask+(1.-mask)*y_0], dim=1), sample_gammas)
                # loss = self.loss_fn(mask*noise, mask*noise_hat)
                inputs = [noisy_target_image * masks + (1.0 - masks) * next_noisy_target_image]

            # If NOT using controlnet, we append cond_image to inputs for concatenation
            if not self.controlnet:
                inputs.append(cond_image)

        # 5. Concat target and conditional latents in the channel dimension.
        # TODO Remove this
        # sample = torch.cat(inputs, dim=1)

        # 6. Prepare style, if needed
        other_model_inputs = {}
        terrain_style = self.pipeline.create_terrain_style(
            target_image, cond_image, item['metadata'])
        if exists(terrain_style):
            other_model_inputs['encoder_hidden_states'] = self.pipeline._encode_terrain_style(
                terrain_style,
                do_classifier_free_guidance=False
            )
        # other_model_inputs['masks'] = masks

        # 7. Handle Input Creation
        if self.controlnet:
            # For ControlNet, 'sample' is just the noisy image.
            # 'controlnet_cond' is passed via kwargs.
            sample = inputs[0]  # Just the noisy image
            other_model_inputs['controlnet_cond'] = cond_image
        else:
            # For Standard U-Net, 'sample' is the concatenation of noisy + cond
            sample = torch.cat(inputs, dim=1)

        inputs = ModelInputs(
            sample=sample,
            timestep=timesteps,
            **other_model_inputs
        )
        if torch.isnan(cond_image).any():
            print("Error: NaN detected in conditional image input")
        return inputs, noise, masks

    @run_if(lambda self: self.steps % self.sampling_args.sample_steps == 0 or self.steps == self.total_train_steps)
    def sample(self):
        self.model.eval()

        dl = iter(self.dataloaders['valid'])

        num_samples = min(self.sampling_args.num_samples, len(self.dataloaders['valid']))
        num_samples -= num_samples % self.training_args.sampling_batch_size  # Subtract offset

        num_iters = num_samples // self.training_args.sampling_batch_size

        # Create generator for reproducibility
        generator = None
        if exists(self.training_args.seed):
            generator = [torch.Generator(self.model.device) for _ in range(num_iters)]
            for i, g in enumerate(generator):
                g.manual_seed(self.training_args.seed + i)

        images = []
        sampling_info = f'Sampling | epoch={self.epoch:.2f}, steps={self.steps}'
        sample_loss = 0
        scaled_sample_loss = 0
        zero_loss = 0
        noise_level = 0
        with (
            trange(num_iters, desc=sampling_info) as progress,
            torch.no_grad(),
            autocast(device_type=self.device.type, enabled=self.training_args.amp)
        ):
            for i in progress:
                item = next(dl)
                target_image: torch.Tensor = item['target_image'].to(self.device)
                original_target_image = target_image.clone()
                if self.auto_encoder is not None:
                    y = target_image
                    if self.auto_encoder.config['in_channels'] == 3:
                        sat_targets = y[:, 0:3]
                        dem_targets = y[:, 3:4].repeat(1, 3, 1, 1)
                        sat_latents = self.auto_encoder.encode(sat_targets).latent_dist.sample() * self.sat_latent_scaling_factor
                        dem_latents = self.auto_encoder.encode(dem_targets).latent_dist.sample() * self.dem_latent_scaling_factor
                        latent = torch.cat([sat_latents, dem_latents], dim=1)
                    elif self.auto_encoder.config['in_channels'] == 4:
                        latent = self.auto_encoder.encode(y).latent_dist.sample() * (self.sat_latent_scaling_factor + self.dem_latent_scaling_factor)/2
                    else:
                        raise ValueError('Autoencoder must have 3 or 4 input channels')

                    target_image = latent
                to_zip = []
                cond_image = None
                pipeline_inputs = {}
                # (b) encode conditional image into its latent space
                if exists(item.get('cond_image')):
                    cond_image = item['cond_image'].to(self.device)

                    resized_inputs = cond_image
                    if resized_inputs.shape[-2:] != target_image.shape[-2:]:
                        # Resize conditional image if needed
                        resized_inputs = F.interpolate(
                            resized_inputs, size=target_image.shape[-2:])

                    to_zip.append(torch.clamp(resized_inputs, -1, 1))

                    pipeline_inputs['cond_image'] = resized_inputs

                else:
                    pipeline_inputs['output_size'] = target_image.shape

                terrain_style = self.pipeline.create_terrain_style(
                    target_image, cond_image, item['metadata'])

                # Pass inputs through the pipeline
                outputs = self.pipeline(
                    num_inference_steps=self.scheduler_args.num_inference_steps,
                    eta=self.scheduler_args.eta,
                    generator=generator[i] if isinstance(generator, list) else generator,
                    normalise_output=self.sampling_args.normalise_output,
                    terrain_style=terrain_style,
                    target_image=target_image,
                    **pipeline_inputs,
                ).images
                
                if self.auto_encoder is not None:
                    if self.auto_encoder.config['in_channels'] == 3:
                        sat_latents = outputs[:, 0:4] / self.sat_latent_scaling_factor
                        dem_latents = outputs[:, 4:8] / self.dem_latent_scaling_factor
                        sat_outputs = self.auto_encoder.decode(sat_latents).sample
                        dem_outputs = self.auto_encoder.decode(dem_latents).sample
                        dem_outputs = dem_outputs[:, 0:1]
                        outputs = torch.cat([sat_outputs, dem_outputs], dim=1)
                    elif self.auto_encoder.config['in_channels'] == 4:
                        outputs = outputs / ((self.sat_latent_scaling_factor + self.dem_latent_scaling_factor)/2)
                        outputs = self.auto_encoder.decode(outputs).sample
                    else:
                        raise ValueError('Autoencoder must have 3 or 4 input channels')
                    target_image = original_target_image

                # For now resize the first 4 channels of the output to the target image size
                outputs = F.interpolate(outputs[:, 0:4], size=original_target_image.shape[-2:])
                target_image = F.interpolate(target_image[:, 0:4], size=original_target_image.shape[-2:])

                to_zip.append(outputs)
                # if exists(cond_image) or exists(self.terrain_encoder):
                # # Display target image if conditional image or encoding provided
                to_zip.append(target_image)
                sample_loss += self.criterion(outputs, target_image).cpu()
                mask_ratios = item['metadata'].get('mask_ratio', 1.0) 
                scaled_sample_loss += (self.criterion(outputs, target_image, reduction='none').mean(axis=(1,2,3))/mask_ratios).mean().cpu()
                zero_loss += torch.nn.functional.mse_loss(torch.zeros_like(outputs)-1, outputs).cpu()
                for j in range(outputs.shape[0]):
                    noise_level += estimate_noise(outputs[0][0].squeeze().cpu().numpy()/2+0.5)
                if self.planet_cfg is None:
                    to_zip = tile_images(to_zip)
                    for row in zip(*to_zip):
                        images.extend(row)
                else:
                    # Why did I do this?
                    # cond_image[:, :4] = original_target_image[:, :4]
                    row = self.planet_cfg.output_display(cond_image, outputs, target_image)
                    images.append(row)
        self.last_sample_loss = scaled_sample_loss/num_iters
        self.wandb_metrics.update({'scaled_sample_loss': scaled_sample_loss/num_iters})
        self.wandb_metrics.update({'sample_loss': sample_loss/num_iters})
        self.wandb_metrics.update({'zero_loss': zero_loss/num_iters})
        self.wandb_metrics.update({'noise_level': noise_level/num_samples})
        self.sample_losses.append(sample_loss/num_iters)
        # Add std and mean of losses
        if len(self.sample_losses) > 1:
            self.wandb_metrics.update({'sample_loss_std': torch.stack(self.sample_losses).std()})
            self.wandb_metrics.update({'sample_loss_mean': torch.stack(self.sample_losses).mean()})


        # Get line of best fit of losses and add gradient of line
        if len(self.sample_losses) > 1:
            x = torch.arange(len(self.sample_losses))
            y = torch.stack(self.sample_losses).to(torch.float32)
            slope, intercept = np.polyfit(x, y, 1)
            self.wandb_metrics.update({'sample_loss_gradient': slope})
        nrow = len(images) // num_samples

        # TODO remove code duplication

        grid = self.unnormalise(make_grid(images, nrow=nrow)) if self.planet_cfg is None else self.unnormalise(torch.concatenate(images, dim=1))
        grid = torch.clamp(grid, 0, 1)
        out_dir = self.sampling_args.samples_dir
        os.makedirs(out_dir, exist_ok=True)
        self.wandb_metrics.update({'samples': wandb.Image(grid.cpu().numpy().transpose((1, 2, 0)), caption=f'sample-{self.steps}')})
        if self.training_args.save_on_disk:
            save_image(grid, os.path.join(out_dir, f'sample-{self.steps}.png'))


def loss_fn_selector(loss_fn: str, device, lpips_scale_factor: float) -> Callable:
    if loss_fn == 'lpips':
        # assert image_args.target_image_channels == 3 or image_args.target_image_channels == 1, 'LPIPS can only be used on RGB images'
        lpips_model = lpips.LPIPS(net='vgg') # Can only be used on RGB images
        lpips_model = lpips_model.to(device)
        reshape = lambda x: x if x.shape[1] == 3 else torch.cat([x]*3, dim=1)

        def lpips_loss(x, y, reduction='mean'):
            channels = x.shape[1]
            losses = []
            num_rgb = channels//3
            num_grayscale = channels % 3
            for i in range(num_rgb):
                x_img = x[:, i*3:(i+1)*3]
                y_img = y[:, i*3:(i+1)*3]
                losses.append(lpips_model(x_img, y_img))
            for i in range(num_grayscale):
                x_img = x[:, num_rgb*3+i: num_rgb*3+i+1]
                y_img = y[:, num_rgb*3+i: num_rgb*3+i+1]
                losses.append(lpips_model(reshape(x_img), reshape(y_img)))
            loss = torch.stack(losses, dim=1).mean(dim=1) * lpips_scale_factor
            return {'mean': loss.mean(), 'sum': loss.sum(), 'none': loss}[reduction]
        criterion = lpips_loss
    elif loss_fn == 'mse':
        criterion = F.mse_loss
    elif loss_fn == 'mae':
        criterion = F.l1_loss
    elif loss_fn == 'wmse':
        def weighted_mse_loss(x, y, reduction='mean'):
            loss = F.mse_loss(x, y, reduction='none')
            for i in range(loss.shape[1] % 3):
                loss[:, loss.shape[1] // 3 + i] *= 3
            return {'mean': loss.mean(), 'sum': loss.sum(), 'none': loss}[reduction]
        criterion = weighted_mse_loss
    elif loss_fn == 'huber':
        criterion = lambda x, y, reduction='mean': F.huber_loss(x, y, reduction=reduction, delta=0.5)
    elif loss_fn == 'noise':
        criterion = NoiseLoss()
    else:
        raise ValueError(f'Invalid `loss_fn` specified: {loss_fn}')
    return criterion

def main(
    dataset_args: DatasetArguments,
    image_args: ImageArguments,
    training_args: DiffusionTrainingArguments,
    sampling_args: SamplingArguments,
    diffusion_args: DiffusionArguments,
    scheduler_args: SchedulerArguments,
    planet_cfg: PlanetConfig,
    params: dict = {},
):
    # TODO https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
    # https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/latent_diffusion/pipeline_latent_diffusion_superresolution.py
    wandb_mode = 'online' if training_args.use_wandb else 'disabled'
    run = wandb.init(project="PlanetAI", config=params, mode=wandb_mode, name=training_args.experiment_name) if not wandb.run else wandb.run
    params = dict(wandb.config) if params == {} and wandb.run else params
    if params:
        for args in [dataset_args, image_args, training_args, sampling_args, diffusion_args, scheduler_args, planet_cfg]:
            for field in fields(args):
                if field.name in params:
                    setattr(args, field.name, params[field.name])
            if hasattr(args, '__post_init__'):
                args.__post_init__()
    bs = training_args.sampling_batch_size or training_args.batch_size
    if sampling_args.num_samples < bs or sampling_args.num_samples % bs != 0:
        sampling_args.num_samples = bs

    if training_args.amp:
        # For some reason clip_grad_norm must be true for amp to work
        # The gradients are probably overflowing or underflowing
        training_args.clip_grad_norm = True
    image_args.update_channels(planet_cfg)

    for args in [dataset_args, image_args, training_args, sampling_args, diffusion_args, scheduler_args, planet_cfg]:
        for field in fields(args):
            if getattr(args, field.name) != field.default and "factory" not in field.name:
                print(f"{args.__class__.__name__}.{field.name} is set to {getattr(args, field.name)} (default={field.default})")
                wandb.config[field.name] = getattr(args, field.name)

    height = 256 * 2 ** planet_cfg.size
    width = 2 * height
    run_name = wandb.run.name
    folder = str(planet_cfg)
    
    suffix = f"{folder}/{width}x{height}/" if image_args.target_image_type == 'planet' else ''
    print(f"Sweep ID: {wandb.run.sweep_id}")
    suffix = training_args.experiment_name or run_name or folder
    training_args.results_dir = os.path.join(training_args.results_dir, suffix)
    sampling_args.samples_dir = os.path.join(sampling_args.samples_dir, suffix)

    print(f"Results dir: {training_args.results_dir}")
    print(f"Samples dir: {sampling_args.samples_dir}")
       
    # logger.info('Generating transformations. This will only be done once per configuration.')
    # setup(planet_cfg=planet_cfg, output_dir=dataset_args.dataset_folder, force=dataset_args.setup)
            

    # Parse arguments and display using logger
    logger.info(f'{" Training with arguments ":*^40}')
    logger.info(dataset_args)
    logger.info(image_args)
    logger.info(training_args)
    logger.info(sampling_args)
    logger.info(diffusion_args)
    logger.info(scheduler_args)
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

    # Trainer code
    trainer_class = None
    trainer_kwargs = {}

    index_mapping = {}

    # In image space
    unet_out_channels = unet_in_channels = image_args.target_image_channels

    auto_encoder = None
    if diffusion_args.latent_diffusion:
        if diffusion_args.auto_encoder_model_path is not None:
            auto_encoder = AutoencoderKL.from_pretrained(
                diffusion_args.auto_encoder_model_path
            ).to(device)
            test_im = torch.randn((1, unet_out_channels, 256, 256)).to(device)
            latent_shape = auto_encoder.encode(test_im).latent_dist.sample().shape
            unet_in_channels = latent_shape[1]
            unet_out_channels = latent_shape[1]
        else:
            repo_id = "stabilityai/stable-diffusion-2-base"
            vae = AutoencoderKL.from_pretrained(repo_id, subfolder='vae').to(device)   
            auto_encoder = vae.to(device)
            test_im = torch.randn((1, 3, 256, 256)).to(device)
            latent_shape = auto_encoder.encode(test_im).latent_dist.sample().shape
            # We have to use double the channels because the stable diffusion vae only accepts 3 channel images
            unet_in_channels = latent_shape[1]*2
            unet_out_channels = latent_shape[1]*2

    # Add additional channels if conditional generation
    if exists(dataset_args.cond_image) and not training_args.use_controlnet:
        unet_in_channels += image_args.cond_image_channels

    model_kwargs = dict(
        in_channels=unet_in_channels,
        out_channels=unet_out_channels,
        block_out_channels=diffusion_args.unet_block_out_channels,
        down_block_types=diffusion_args.unet_down_block_types,
        up_block_types=diffusion_args.unet_up_block_types,
        layers_per_block=diffusion_args.unet_layers_per_block,
        attention_head_dim=diffusion_args.unet_attention_head_dim,
        norm_eps=diffusion_args.unet_norm_eps,
    )

    terrain_encoder = None
    if diffusion_args.unet_attn_type == 'crossattn' or diffusion_args.unet_use_embeddings:
        model_class = UNet2DConditionModel

        if diffusion_args.unet_use_embeddings:
            if image_args.target_image_type == 'elevation':
                terrain_encoder = GlobalTerrainEncoder()
            elif image_args.target_image_type == 'satellite':
                terrain_encoder = SatelliteTerrainEncoder()
            elif image_args.target_image_type == 'planet':
                terrain_encoder = PlanetEncoder(planet_cfg)
            else:
                raise ValueError(
                    f'Invalid `--target_image_type` specified: {image_args.target_image_type}')

            unet_cross_attention_dim = terrain_encoder.cross_attention_dim
        else:
            # TODO is this valid?
            unet_cross_attention_dim = 48

        model_kwargs.update(
            cross_attention_dim=unet_cross_attention_dim
        )
    else:
        model_class = UNet2DModel

    model = model_class(**model_kwargs)

    controlnet = None
    if training_args.use_controlnet:
        assert isinstance(model, UNet2DConditionModel)
        controlnet, model = load_controlnet_and_model(model, training_args.results_dir, image_args.cond_image_channels)
        logger.info("Freezing U-Net and enabling ControlNet training...")

        # Freeze U-Net
        model.requires_grad_(False)
        controlnet.train()
        controlnet.to(device)

    if diffusion_args.compile_unet:
        logging.info("Compiling model")
        model = torch.compile(model, mode='reduce-overhead')

    num_params = 0
    trainable_params = []
    if controlnet:
        for param in controlnet.parameters():
            num_params += param.numel()
            trainable_params.append(param)
    else:
        for param in model.parameters():
            num_params += param.numel()
            trainable_params.append(param)

    logger.info(f'Number of parameters: {num_params/1e6:.2f}M')
    wandb.config['num_params'] = num_params
    noise_scheduler_kwargs = dict(
        beta_start=scheduler_args.beta_start,
        beta_end=scheduler_args.beta_end,
        beta_schedule='squaredcos_cap_v2' if scheduler_args.beta_schedule == 'cosine' else scheduler_args.beta_schedule,
    )
    if scheduler_args.noise_scheduler == 'DDIMScheduler' or scheduler_args.noise_scheduler == 'DDPMScheduler':
        noise_scheduler_kwargs.update(
            clip_sample=scheduler_args.clip_sample,
        )

    # TODO do tests with other schedulers
    noise_scheduler = eval(scheduler_args.noise_scheduler)(**noise_scheduler_kwargs)
    noise_scheduler.config.num_train_timesteps = scheduler_args.num_train_timesteps
    noise_scheduler.set_timesteps(scheduler_args.num_inference_steps)

    pipeline = TerrainDiffusionPipeline(
        unet=model,
        controlnet=controlnet,
        terrain_encoder=terrain_encoder,
        scheduler=noise_scheduler,
        bin_dir=os.path.join(dataset_args.dataset_folder, 'train', '_bins'),
    )
    pipeline = pipeline.to(device)

    trainer_class = DiffusionModelTrainer
    trainer_kwargs = dict(
        model=pipeline,

        # Additional arguments
        training_args=training_args,
        scheduler_args=scheduler_args,
        sampling_args=sampling_args,
        planet_cfg=planet_cfg,
        auto_encoder=auto_encoder,
        sat_latent_scaling_factor=diffusion_args.sat_latent_scaling_factor,
        dem_latent_scaling_factor=diffusion_args.dem_latent_scaling_factor,
    )

    index_mapping['cond_image'] = dataset_args.cond_image
    index_mapping['target_image'] = dataset_args.target_image

    # Move model to device
    model = model.to(device)

    shared_dataset_kwargs = dict(
        index_mapping=index_mapping,
        seed=training_args.seed,
    )
    if image_args.target_image_type != 'planet':
        datasets = dict(
            train=TerrainDataset(
                folder=os.path.join(dataset_args.dataset_folder, 'train'),
                **shared_dataset_kwargs
            ),
            valid=TerrainDataset(
                folder=os.path.join(dataset_args.dataset_folder, 'valid'),
                data_augmentation=False,
                **shared_dataset_kwargs
            ),
            test=TerrainDataset(
                folder=os.path.join(dataset_args.dataset_folder, 'test'),
                data_augmentation=False,
                **shared_dataset_kwargs
            )
        )
    else:
        train_cfg = replace(planet_cfg)
        train_cfg.iters = int(train_cfg.iters*0.8)
        valid_cfg = replace(planet_cfg)
        valid_cfg.iters = min(int(valid_cfg.iters*0.1), 10000)
        valid_cfg.randomize_steps = 0
        if valid_cfg.planet_seed is not None:
            valid_cfg.planet_seed += 10
        test_cfg = replace(planet_cfg)
        test_cfg.iters = min(int(test_cfg.iters*0.1), 10000)
        test_cfg.randomize_steps = 0
        if test_cfg.planet_seed is not None:
            test_cfg.planet_seed += 20
        
        datasets = dict(
            train=RAMDataset(
                train_cfg,
                image_args.target_image_channels,
                image_args.cond_image_channels,
                normalise=True,
                conditioning_dropout=training_args.conditioning_dropout,
                tile_size=training_args.tile_size,
                mode='train',
                auto_encoder=None,
            ),
            valid=RAMDataset(
                valid_cfg,
                image_args.target_image_channels,
                image_args.cond_image_channels,
                normalise=True,
                conditioning_dropout=training_args.conditioning_dropout,
                tile_size=training_args.sample_tile_size,
                mode='val',
                auto_encoder=None,
            ),
            test=RAMDataset(
                test_cfg,
                image_args.target_image_channels,
                image_args.cond_image_channels,
                normalise=True,
                conditioning_dropout=training_args.conditioning_dropout,
                tile_size=training_args.sample_tile_size,
                mode='test',
                auto_encoder=None,
            )
        )

    metrics = []  # TODO add metrics
    if training_args.extra_metrics:
        metrics.append(TrainingMetrics(device=device))

    # Define optimizer
    optimizer = optim.AdamW(
        trainable_params, # Only optimize ControlNet if use_controlnet is True
        lr=training_args.lr,
        # betas=(0.9, 0.999), # These are default anyway
        weight_decay=training_args.weight_decay,  # https://stackoverflow.com/a/46597531/13989043
    )

    # Define loss function
    criterion = loss_fn_selector(training_args.loss_fn, device, training_args.lpips_scale_factor)
    perceptual_criterion = loss_fn_selector(training_args.perceptual_loss_fn, device, training_args.lpips_scale_factor)


    # Define scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=training_args.lr_scheduler_factor,
        patience=training_args.lr_scheduler_patience,
        threshold=training_args.lr_scheduler_threshold,
        min_lr=training_args.lr_scheduler_min_lr,
        verbose=training_args.lr_scheduler_verbose,
    ) if training_args.lr_scheduler_type == 'plateau' else get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=training_args.num_warmup_steps,
        num_training_steps=(len(datasets['train']) * training_args.num_epochs),
    )

    trainer = trainer_class(
        datasets=datasets,
        optimizer=optimizer,
        criterion=criterion,
        perceptual_criterion=perceptual_criterion,
        scheduler=scheduler,
        metrics=metrics,
        load_model_weights=not training_args.use_controlnet,
        **trainer_kwargs
    )

    trainer.train()


if __name__ == '__main__':
    parser = CustomArgumentParser(
        (
            DatasetArguments,

            ImageArguments,

            DiffusionTrainingArguments, SamplingArguments,

            # Diffusion arguments
            DiffusionArguments, SchedulerArguments,

            PlanetConfig
        ),
        description='Train a diffusion model'
    )

    args: tuple[
        DatasetArguments,
        ImageArguments,
        DiffusionTrainingArguments,
        SamplingArguments,
        DiffusionArguments,
        SchedulerArguments,
        PlanetConfig
    ] = parser.parse_args_into_dataclasses()

    main(*args)
    # sweep_id = wandb.sweep(sweep=sweep_configuration, project="PlanetAI")
    # wandb.agent(sweep_id, count=10, function=main)
