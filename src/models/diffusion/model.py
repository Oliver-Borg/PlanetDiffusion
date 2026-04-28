from dataclasses import dataclass
import inspect
from typing import Optional, Tuple, Union, List, Generator
import os

import torch
import torch.utils.checkpoint
import numpy as np

from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.models import (
    UNet2DModel,
    UNet2DConditionModel,
    ControlNetModel,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput
from ...labelling.encoding import (
    TerrainEncoder,
    TerrainStyle,
    GlobalTerrainEncoder,
    GlobalTerrainStyle,
    SatelliteTerrainEncoder,
    SatelliteTerrainStyle,
    PlanetEncoder,
    PlanetStyle
)
from ...core.utils import exactly_one_exists, exists
from torchvision.transforms import ToPILImage
from planetAI.src.data.utils import create_overlap_mask

import sys
import diffusers.models.unets.unet_2d_condition
import diffusers.models.unets.unet_2d_blocks
import diffusers.models.transformers.transformer_2d
# Shim: Redirect the old module path to the new location
sys.modules["diffusers.models.transformer_2d"] = diffusers.models.transformers.transformer_2d
sys.modules["diffusers.models.unet_2d_condition"] = diffusers.models.unets.unet_2d_condition
sys.modules["diffusers.models.unet_2d_blocks"] = diffusers.models.unets.unet_2d_blocks


TERRAIN_ENCODER_FOLDER = 'terrain_encoder'
TERRAIN_ENCODER_NAME = 'terrain_encoder.bin'
CONTROLNET_FOLDER = 'controlnet'


def return_feedback_decorator(function):
    def wrapper(*args, **kwargs):
        kwargs['return_feedback'] = kwargs.pop('return_feedback', False)
        result = function(*args, **kwargs)
        if not kwargs['return_feedback']:
            result = next(result)
        return result
    return wrapper


@dataclass
class TerrainDiffusionPipelineOutput(BaseOutput):
    """
    Output class for Terrain Diffusion pipelines.

    Args:
        images (`torch.Tensor`)
            Denoised images as torch.Tensor array of shape `(batch_size, height, width, num_channels)`
    """

    images: torch.Tensor
    step: int
    num_inference_steps: int


class TerrainDiffusionPipeline(DiffusionPipeline):
    r"""
    A pipeline for image-to-image translation using Latent Diffusion

    This class inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        unet ([`UNet2DConditionModel`]): U-Net architecture to denoise the encoded image.
        controlnet ([`ControlNetModel`]):
            Provides additional conditioning to the `unet` during the denoising process. If you set this parameter,
            the `unet` parameter must be of type `UNet2DConditionModel`.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], [`EulerDiscreteScheduler`],
            [`EulerAncestralDiscreteScheduler`], or [`PNDMScheduler`].
        terrain_encoder ([`TerrainEncoder`]):
            Frozen terrain-encoder. Used for stylisation of terrain.
        bin_dir (str):
            Path to the directory containing the encoder channel bin files.
    """
    unet: Union[UNet2DModel, UNet2DConditionModel]
    controlnet: Optional[ControlNetModel]
    _optional_components = ["controlnet"]

    def __init__(
        self,
        unet: Union[UNet2DModel, UNet2DConditionModel],
        scheduler: Union[
            DDIMScheduler,
            DDPMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
        ],
        controlnet: Optional[ControlNetModel] = None,
        terrain_encoder: Optional[TerrainEncoder] = None,
        bin_dir: Optional[str] = '',
    ):
        if isinstance(unet, UNet2DConditionModel) and terrain_encoder is None:
            raise ValueError(
                '`terrain_encoder` must be specified when using `UNet2DConditionModel`')

        super().__init__()

        self.register_modules(
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
        )
        self.terrain_encoder = terrain_encoder
        self.bin_dir = bin_dir

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        safe_serialization: bool = False,
    ):
        # Workaround to save terrain_encoder separately
        super().save_pretrained(save_directory, safe_serialization)

        if exists(self.terrain_encoder):
            terrain_encoder_folder = os.path.join(
                save_directory, TERRAIN_ENCODER_FOLDER)
            os.makedirs(terrain_encoder_folder, exist_ok=True)
            torch.save(self.terrain_encoder, os.path.join(
                terrain_encoder_folder, TERRAIN_ENCODER_NAME))

        if exists(self.controlnet):
            controlnet_folder = os.path.join(save_directory, CONTROLNET_FOLDER)
            os.makedirs(controlnet_folder, exist_ok=True)
            self.controlnet.save_pretrained(controlnet_folder, safe_serialization=safe_serialization)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs) -> 'TerrainDiffusionPipeline':
        terrain_encoder_path = os.path.join(pretrained_model_name_or_path, TERRAIN_ENCODER_FOLDER, TERRAIN_ENCODER_NAME)
        controlnet_path = os.path.join(pretrained_model_name_or_path, CONTROLNET_FOLDER)

        if os.path.exists(terrain_encoder_path):
            kwargs[TERRAIN_ENCODER_FOLDER] = torch.load(terrain_encoder_path)

        if os.path.exists(controlnet_path):
            # 1. Load Config & Instantiate Standard Model
            config = ControlNetModel.load_config(controlnet_path)
            controlnet = ControlNetModel.from_config(config)

            # 2. Load Weights Manually
            model_file = os.path.join(controlnet_path, "diffusion_pytorch_model.bin")
            state_dict = torch.load(model_file, map_location="cpu")

            # 3. Detect & Re-apply Surgery
            # The custom model has 'controlnet_cond_embedding.weight' (Conv2d style)
            # The standard model expects 'controlnet_cond_embedding.blocks.0.weight'
            is_custom_embedding = 'controlnet_cond_embedding.weight' in state_dict

            if is_custom_embedding:
                print("Detected custom ControlNet embedding (Surgery). Re-applying Conv2d layer...")

                # Infer dimensions from the saved weight tensor
                # Shape is [out_channels, in_channels, k, k]
                weight = state_dict['controlnet_cond_embedding.weight']
                out_c, in_c, k, _ = weight.shape

                # Replace the complex block with the simple Conv2d
                controlnet.controlnet_cond_embedding = torch.nn.Conv2d(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=k,
                    padding=k // 2
                )

            # 4. Load State Dict
            controlnet.load_state_dict(state_dict)
            kwargs[CONTROLNET_FOLDER] = controlnet

        return super().from_pretrained(
            pretrained_model_name_or_path,
            **kwargs
        )

    def prepare_inputs(self,
                       batch_size, num_noise_channels, unet_input_height, unet_input_width,
                       dtype, device, generator,
                       ):

        shape = (batch_size, num_noise_channels,
                 unet_input_height, unet_input_width)

        inputs = randn_tensor(shape, generator=generator,
                             device=device, dtype=dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        inputs = inputs * self.scheduler.init_noise_sigma
        return inputs

    def create_terrain_style(self, target_image, condition=None,  metadata={}):
        if self.terrain_encoder is None:
            return None

        if isinstance(self.terrain_encoder, GlobalTerrainEncoder):
            terrain_style = GlobalTerrainStyle(
                terrains=target_image,
                ranges=metadata['range'],
                resolutions=metadata['resolution'],
            )
        elif isinstance(self.terrain_encoder, SatelliteTerrainEncoder):
            terrain_style = SatelliteTerrainStyle(terrains=target_image)
        elif isinstance(self.terrain_encoder, PlanetEncoder):
            terrain_style = PlanetStyle(sketches=condition, dems=target_image, metadata=metadata)
        else:
            terrain_style = TerrainStyle(terrains=target_image)

        return terrain_style

    def _encode_terrain_style(self,
                              terrain_style: TerrainStyle,
                              do_classifier_free_guidance=False
                              ):
        # TODO add negative_prompts

        # Generate encodings based on normalised target_image
        encoder_hidden_states = self.terrain_encoder(terrain_style).to(self.device)

        if do_classifier_free_guidance:
            baseline = self.terrain_encoder.baseline(
                batch_size=terrain_style.terrains.shape[0]).to(self.device)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            encoder_hidden_states = torch.cat(
                [baseline, encoder_hidden_states])

        encoder_hidden_states = encoder_hidden_states.to(self.device)
        return encoder_hidden_states

    @return_feedback_decorator
    @torch.no_grad()
    def __call__(
        self,

        # Typically a sketch or low-res image
        cond_image: Optional[torch.Tensor] = None,
        output_size: Optional[Tuple[int]] = None,

        # Further condition on exemplars (which will be encoded using `self.terrain_encoder`)
        terrain_style: Optional[TerrainStyle] = None,

        eta: Optional[float] = 1.0,
        generator: Optional[Union[torch.Generator,
                                  List[torch.Generator]]] = None,
        seed: Optional[Union[int, List[int]]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 1,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        normalise_output=True,
        return_feedback=False,

        progress_bar=True,
        input_has_noise: bool = False,
        target_image: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Generator[TerrainDiffusionPipelineOutput, None, None], TerrainDiffusionPipelineOutput]:
        r"""
        Args:
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 1):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the `terrain_style`,
                usually at the expense of lower image quality.
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the ControlNet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original `unet`. If multiple ControlNets are specified in `init`, you can set
                the corresponding scale as a list.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
        Returns:
            `Union[Generator[TerrainDiffusionPipelineOutput], TerrainDiffusionPipelineOutput]`
        """
        # TODO move comments to docstrings
        # TODO remove code duplication with trainer forward method
        # TODO add negative styles

        # 1. Check inputs. Raise error if not correct
        if not exactly_one_exists(cond_image, output_size):
            raise ValueError(
                'Must specify exactly one of `cond_image` or `output_size`.'
            )

        if isinstance(self.unet, UNet2DConditionModel) and terrain_style is None:
            raise ValueError(
                '`terrain_style` must be specified when using `UNet2DConditionModel`')

        # 2. Define call parameters
        device = self.device
        dtype = next(self.unet.parameters()).dtype
        if exists(seed):
            if exists(generator):
                raise ValueError(
                    'May not specify `generator` and `seed` at the same time'
                )

            # Seed specified, but generator not specified.
            if isinstance(seed, list):
                # Create list of generators
                generator = [torch.Generator(
                    device).manual_seed(s) for s in seed]
            else:
                generator = torch.Generator(device).manual_seed(seed)

        # 3. Set unet parameters
        if exists(cond_image):
            if not isinstance(cond_image, torch.Tensor):
                raise ValueError(
                    f"If specified, `cond_image` has to be of type `torch.Tensor` but is {type(cond_image)}"
                )

            batch_size, _, height, width = cond_image.shape

            # Ensure RGBA image
            # if cond_channels != 4:
            #     raise ValueError(
            #         f'Invalid number of conditional channels. Expected 4 but got {cond_channels}.')

        else:  # Unconditional generation
            if len(output_size) != 3:
                batch_size, _, height, width = output_size
                # raise ValueError(
                #     '`output_size` must be of the form (num_samples, height, width)')
            else:
                batch_size, height, width = output_size
            depth = self.unet.config.in_channels - self.unet.config.out_channels

            # Construct "empty" conditional image
            cond_image = torch.cat(
                [
                    torch.full(
                        (batch_size, 3, height, width),
                        fill_value=-1,
                        device=device, dtype=dtype
                    ),
                    torch.full(
                        (batch_size, 1, height, width),
                        fill_value=1,
                        device=device, dtype=dtype
                    ),
                ],
                dim=1  # Concatenate on channel
            ) if depth == 4 else torch.full(
                (batch_size, depth, height, width),
                fill_value=-1,
                device=device, dtype=dtype
            )

        num_noise_channels = self.unet.config.out_channels

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 4. Encode `terrain_styles` if needed
        unet_kwargs = {}
        if exists(self.terrain_encoder):
            unet_kwargs['encoder_hidden_states'] = self._encode_terrain_style(
                terrain_style, do_classifier_free_guidance
            )

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        # 6. Prepare inputs (noise) for unet
        # (i.e., what we will be denoising)
        unet_inputs = self.prepare_inputs(
            batch_size,
            num_noise_channels,
            height,
            width,
            dtype,
            device,
            generator
        )
        self.starting_image = unet_inputs

        original_noise = unet_inputs.clone()
        masks = None
        mask_is = [m.get('mask_channel') for m in terrain_style.metadata] if exists(terrain_style) else None
        if all([exists(m_i) for m_i in mask_is]):
            masks = torch.cat([cond_image[i, mask_i, :, :].unsqueeze(0) for i, mask_i in enumerate(mask_is)])
            masked_target_image = torch.cat([cond_image[i, :mask_i, :, :].unsqueeze(0) for i, mask_i in enumerate(mask_is)])
            cond_image = torch.cat([cond_image[i, mask_i+1:, :, :].unsqueeze(0) for i, mask_i in enumerate(mask_is)])
            masks += 1.0
            masks /= 2.0
            channels = unet_inputs.shape[1]
            masks = masks.unsqueeze(1)
            masks = masks.repeat(1, channels, 1, 1)
            # So we can use noise generated globally
            if input_has_noise:
                unet_inputs = masked_target_image
            if target_image is None:
                target_image = masked_target_image
            # noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy*mask+(1.-mask)*y_0], dim=1), sample_gammas)
            # loss = self.loss_fn(mask*noise, mask*noise_hat)

        # 7. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8. Add progress bar if wanted
        if progress_bar:
            progress_timesteps = self.progress_bar(timesteps)
        else:
            progress_timesteps = timesteps

        # 9. Denoising loop
        output = None
        self.step_images = [unet_inputs]
        for i, t in enumerate(progress_timesteps):
            # concat target and conditional latents in the channel dimension.
            # NOTE: will throw error if latent width and height are not the same
            # (number of channels may differ)
            if masks is not None:
                b, c, h, w = unet_inputs.shape
                if i == num_inference_steps - 1:
                    noisy_target_image = target_image
                    temp_masks = masks
                else:
                    noisy_target_image = self.scheduler.add_noise(
                        target_image, original_noise, timesteps[i+1].long())
                    temp_masks = [torch.stack([torch.tensor(create_overlap_mask((h, w)))]*c) for _ in range(b)]
                    temp_masks = torch.stack(temp_masks).to(device)/255.0
                # Remove masks for sampling
                temp_masks *= 0.0
                temp_masks += 1.0
                    
                unet_inputs = unet_inputs * temp_masks + (1.0 - temp_masks) * noisy_target_image

            if self.controlnet is not None:
                # 1. Prepare Inputs
                # Check if surgery was performed (Backbone expects 4 channels)
                if self.unet.config.in_channels == unet_inputs.shape[1]:
                    # Unconditional Backbone (New Efficient Mode)
                    model_input = unet_inputs
                else:
                    # Conditional Backbone (Legacy Mode - requires zero padding)
                    zeros = torch.full_like(cond_image, 0.0)
                    if do_classifier_free_guidance:
                        zeros = torch.cat([zeros] * 2)
                    model_input = torch.cat([unet_inputs, zeros], dim=1)

                # Expand conditioning for CFG
                cond_input = torch.cat([cond_image] * 2) if do_classifier_free_guidance else cond_image

                # Expand model input for CFG
                latents_input = torch.cat([model_input] * 2) if do_classifier_free_guidance else model_input
                latents_input = self.scheduler.scale_model_input(latents_input, t)

                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    sample=latents_input,
                    timestep=t,
                    encoder_hidden_states=unet_kwargs.get('encoder_hidden_states'),
                    controlnet_cond=cond_input,
                    conditioning_scale=controlnet_conditioning_scale,
                    return_dict=False,
                )

                # Predict noise
                noise_pred = self.unet(
                    sample=latents_input,
                    timestep=t,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    **unet_kwargs
                ).sample

            else:
                # Standard Concatenation Logic (Original)
                inputs = [unet_inputs]
                if cond_image is not None:
                    inputs.append(cond_image)
                # noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy*mask+(1.-mask)*y_0], dim=1), sample_gammas)
                # loss = self.loss_fn(mask*noise, mask*noise_hat)
                model_inputs = torch.cat(inputs, dim=1)

                # expand the model_inputs if we are doing classifier free guidance
                model_inputs = torch.cat(
                    [model_inputs] * 2) if do_classifier_free_guidance else model_inputs
                model_inputs = self.scheduler.scale_model_input(model_inputs, t)

                # predict the noise residual
                noise_pred = self.unet(
                    sample=model_inputs,
                    timestep=t,
                    **unet_kwargs
                ).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * \
                    (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            # stepped = self.scheduler.step(
            #     noise_pred, t, unet_inputs, **extra_step_kwargs)
            stepped = self.step(
                unet_inputs, cond_image, t, unet_kwargs, extra_step_kwargs)
            unet_inputs = stepped.prev_sample
            if masks is not None:
                unet_inputs = unet_inputs * temp_masks + (1.0 - temp_masks) * noisy_target_image
            self.step_images.append(unet_inputs)

            # Post-process original if user wants all outputs or it is the final iteration
            if return_feedback or t == 0:
                pred_original_sample = self.postprocess(
                    stepped.pred_original_sample, cond_images=cond_image, normalise_output=normalise_output)

                output = TerrainDiffusionPipelineOutput(
                    images=pred_original_sample,
                    step=i,
                    num_inference_steps=num_inference_steps
                )

                # Yield current prediction of start if user wants
                if return_feedback:
                    yield output

        # Have final yield outside of the progress bar
        if exists(output) and not return_feedback:
            yield output

    @torch.no_grad()
    def _step(
        self,
        unet_inputs: torch.Tensor,
        cond_image: torch.Tensor,
        t: torch.Tensor,
        unet_kwargs: dict,
        extra_step_kwargs: dict,
        guidance_channel: int | None = None,
        guidance_scale: float = 1.0,
    ):
        do_classifier_free_guidance = (
            guidance_scale > 1.0
            and guidance_channel is not None
            and (cond_image[:, guidance_channel, :, :] > -1.0).any()
        )

        if self.controlnet is not None:
            unet_inputs_scaled = self.scheduler.scale_model_input(unet_inputs, t)

            # Auto-detect backbone channels
            if self.unet.config.in_channels == unet_inputs_scaled.shape[1]:
                backbone_input = unet_inputs_scaled
                controlnet_input = unet_inputs_scaled
            else:
                raise ValueError("Channels should match")
                zeros = torch.zeros_like(cond_image)
                backbone_input = torch.cat([unet_inputs_scaled, zeros], dim=1)
                controlnet_input = backbone_input  # ControlNet usually takes same as backbone in diffusers

            down, mid = self.controlnet(
                sample=controlnet_input, timestep=t, encoder_hidden_states=unet_kwargs.get("encoder_hidden_states"), controlnet_cond=cond_image, return_dict=False
            )
            noise_pred = self.unet(
                sample=backbone_input, timestep=t, encoder_hidden_states=unet_kwargs.get("encoder_hidden_states"),
                down_block_additional_residuals=down, mid_block_additional_residual=mid
            ).sample
        else:
            # Standard One-step logic
            model_inputs = torch.cat([unet_inputs, cond_image], dim=1)
            model_inputs = self.scheduler.scale_model_input(model_inputs, t)
            noise_pred = self.unet(
                sample=model_inputs,
                timestep=t,
                encoder_hidden_states=unet_kwargs.get("encoder_hidden_states"),
            ).sample

        # Second pass - unconditional (if doing classifier-free guidance)
        if do_classifier_free_guidance:
            uncond_image = cond_image.clone()
            uncond_image[:, guidance_channel, :, :] = -1.0

            # Bit hacky but we can just run step again without a guidance channel to prevent another recursion
            noise_pred_uncond = self._step(unet_inputs, uncond_image, t, unet_kwargs, extra_step_kwargs, None, 1.0)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

        return noise_pred

    def step(
        self,
        unet_inputs: torch.Tensor,
        cond_image: torch.Tensor,
        t: torch.Tensor,
        unet_kwargs: dict,
        extra_step_kwargs: dict,
        guidance_channel: int | None = None,
        guidance_scale: float = 1.0,
    ) -> DDIMSchedulerOutput:
        noise_pred = self._step(
            unet_inputs, cond_image, t, unet_kwargs, extra_step_kwargs, guidance_channel, guidance_scale
        )
        stepped: DDIMSchedulerOutput = self.scheduler.step(
            noise_pred, t, unet_inputs, **extra_step_kwargs
        )
        return stepped

    def postprocess(self, outputs, cond_images=None, normalise_output=True):
        if normalise_output:
            batched_views = outputs.view(outputs.shape[0], -1)
            min_vals, _ = torch.min(batched_views, dim=1)
            max_vals, _ = torch.max(batched_views, dim=1)
            if cond_images is not None:
                batched_cond_views = cond_images.view(cond_images.shape[0], -1)
                min_cond_vals, _ = torch.min(batched_cond_views, dim=1)
                max_cond_vals, _ = torch.max(batched_cond_views, dim=1)
                min_diffs = min_cond_vals - min_vals
                max_diffs = max_cond_vals - max_vals
                # Scale to between min and max of conditional image
                image_ranges = (max_vals - min_vals)[:, None, None, None]
                cond_ranges = (max_cond_vals - min_cond_vals)[:, None, None, None]
                outputs += min_diffs[:, None, None, None] + 1.0
                outputs *= cond_ranges / image_ranges
                outputs -= 1.0
            else:
                image_ranges = (max_vals - min_vals)[:, None, None, None]

                outputs = (
                    2 * (outputs - min_vals[:, None, None, None]) / image_ranges) - 1

        else:
            outputs = torch.clamp(outputs, -1.0, 1.0)

        return outputs

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        extra_step_kwargs = {}

        scheduler_params = set(inspect.signature(
            self.scheduler.step).parameters.keys())

        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = 'eta' in scheduler_params
        if accepts_eta:
            extra_step_kwargs['eta'] = eta

        # check if the scheduler accepts generator
        accepts_generator = 'generator' in scheduler_params
        if accepts_generator:
            extra_step_kwargs['generator'] = generator
        return extra_step_kwargs
