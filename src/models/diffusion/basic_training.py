from dataclasses import dataclass


@dataclass
class TrainingConfig:
    image_size = 32  # the generated image resolution
    train_batch_size = 64
    eval_batch_size = 8  # how many images to sample during evaluation
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-5
    lr_warmup_steps = 500
    save_image_epochs = 1
    save_model_epochs = 10
    dataset_size = 100000  # how many images to use from the dataset
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "diffusion-test"  # the model name locally and on the HF Hub

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0


config = TrainingConfig()
from planetAI.src.data.dataset import PlanetDataset
from planetAI.src.data.utils import PlanetConfig
from planetAI.src.data.map_paster import setup
train_cfg = PlanetConfig(iters=int(config.dataset_size*0.8), use_mask_store=True, planet_seed=config.seed, data_dir='./planetAI/data')
setup(train_cfg)
train_dataset = PlanetDataset(train_cfg, 1, 1, tile_size=config.image_size)

config.dataset_name = "planet_cfg"
dataset = train_dataset


import torch

train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

from diffusers import UNet2DModel

model = UNet2DModel(
    sample_size=config.image_size,  # the target image resolution
    in_channels=2,  # the number of input channels, 3 for RGB images
    out_channels=1,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)

"""It is often a good idea to quickly check the sample image shape matches the model output shape:"""

sample_image = dataset[0]['target_image'].unsqueeze(0)
sample_cond = dataset[0]['cond_image'].unsqueeze(0)
print("Input shape:", sample_image.shape)


print("Output shape:", model(torch.cat([sample_image, sample_cond], dim=1), timestep=0).sample.shape)
import torch
from PIL import Image
from diffusers import DDIMScheduler
import numpy as np

noise_scheduler = DDIMScheduler(num_train_timesteps=100)
noise = torch.randn(sample_image.shape)
timesteps = torch.LongTensor([50])
noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)
noisy_image = torch.cat([noisy_image, sample_cond], dim=1)
numpy_image = ((noisy_image.permute(0, 2, 3, 1) + 1.0) * 127.5).type(torch.uint8).numpy()[0]
while numpy_image.shape[2] < 3:
    numpy_image = np.dstack([numpy_image, numpy_image[:, :, -1]])
Image.fromarray(numpy_image)


import torch.nn.functional as F

noise_pred = model(noisy_image, timesteps).sample
loss = F.mse_loss(noise_pred, noise)

from diffusers.optimization import get_cosine_schedule_with_warmup

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)
from diffusers import DDIMPipeline
import math
import os

from typing import List, Optional, Tuple, Union

import torch

from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipeline_utils import ImagePipelineOutput

def tensor_to_pil(tensor: torch.Tensor) -> Image:
    output_image = (tensor / 2 + 0.5).clamp(0, 1)
    output_image = output_image.cpu().numpy()
    output_image = Image.fromarray((output_image[0][0] * 255).astype(np.uint8))
    return output_image

class ConditionalDDIMPipeline(DDIMPipeline):
    r"""
    This model inherits from [`DDIMPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    def __init__(self, unet, scheduler):
        super().__init__(unet, scheduler)

        # make sure scheduler can always be converted to DDIM
        scheduler = DDIMScheduler.from_config(scheduler.config)

        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        images: torch.Tensor,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        use_clipped_model_output: Optional[bool] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Args:
            image ([`torch.Tensor`]):
                The image to denoise. Must be of shape `(batch_size, channels, height, width)`.
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            eta (`float`, *optional*, defaults to 0.0):
                The eta parameter which controls the scale of the variance (0 is DDIM and 1 is one type of DDPM).
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            use_clipped_model_output (`bool`, *optional*, defaults to `None`):
                if `True` or `False`, see documentation for `DDIMScheduler.step`. If `None`, nothing is passed
                downstream to the scheduler. So use `None` for schedulers which don't support this argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if `return_dict` is
            True, otherwise a `tuple. When returning a tuple, the first element is a list with the generated images.
        """
        image_channels = images.shape[1]
        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            noise_shape = (
                batch_size,
                self.unet.config.in_channels - image_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            noise_shape = (batch_size, self.unet.config.in_channels - image_channels, *self.unet.config.sample_size)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        noise = randn_tensor(noise_shape, generator=generator, device=self.device, dtype=self.unet.dtype)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_input = torch.cat([noise, images], dim=1)
            model_output = self.unet(model_input, t).sample

            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            stepped = self.scheduler.step(
                model_output, t, images, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator
            )
            noise = stepped.prev_sample
            output_image = stepped.pred_original_sample
            to_show = torch.cat([output_image, images, noise, model_output], dim=3)

        output_image = (to_show / 2 + 0.5).clamp(0, 1)
        output_image = output_image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            output_image = self.numpy_to_pil(output_image)

        if not return_dict:
            return (output_image,)

        return ImagePipelineOutput(images=output_image)


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def evaluate(config, eval_images, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        images=eval_images,
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
    ).images

    # Make a grid out of the images
    image_grid = make_grid(images, rows=config.eval_batch_size, cols=1)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")

from accelerate import Accelerator
from huggingface_hub import HfFolder, Repository, whoami
from tqdm.auto import tqdm
from pathlib import Path
import os


def get_full_repo_name(model_id: str, organization: str = None, token: str = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and wandb logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="wandb",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.push_to_hub:
            repo_name = get_full_repo_name(Path(config.output_dir).name)
            repo = Repository(config.output_dir, clone_from=repo_name)
        elif config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["target_image"]
            cond_images = batch["cond_image"]
            # Sample noise to add to the images
            noise = randn_tensor(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            model_inputs = torch.cat([noisy_images, cond_images], dim=1)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(model_inputs, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            to_show = tensor_to_pil(torch.cat([clean_images, noisy_images, noisy_images - noise_pred.detach(), noisy_images - noise, noise, noise_pred.detach()], dim=3))
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = ConditionalDDIMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                eval_images = [train_dataloader.dataset[i] for i in range(config.eval_batch_size)]
                eval_images = [eval_image['cond_image'] for eval_image in eval_images]
                eval_images = torch.stack(eval_images).to(accelerator.device)

                evaluate(config, eval_images, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=True)
                else:
                    pipeline.save_pretrained(config.output_dir)

from accelerate import notebook_launcher

args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

notebook_launcher(train_loop, args, num_processes=1)
import glob

sample_images = sorted(glob.glob(f"{config.output_dir}/samples/*.png"))
Image.open(sample_images[-1]).show()
