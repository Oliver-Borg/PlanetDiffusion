import os
import logging

from diffusers.models import UNet2DConditionModel, ControlNetModel
import torch
import torch.nn as nn

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ControlNetTrainWrapper(nn.Module):
    """
    Wraps a frozen U-Net and a trainable ControlNet.
    Automatically handles input concatenation based on the backbone's expected channels.
    """

    def __init__(self, unet: UNet2DConditionModel, controlnet: ControlNetModel):
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet
        self.config = unet.config

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: dict | None = None,
        controlnet_cond: torch.Tensor | None = None,
        **kwargs,
    ):
        in_channels = self.config.in_channels
        sample_channels = sample.shape[1]

        if in_channels > sample_channels:
            # Pad inputs to match the backbone's expected channel count (e.g. 9)
            channels_needed = in_channels - sample_channels
            shape = list(sample.shape)
            shape[1] = channels_needed

            zeros = torch.full(shape, 0.0, device=sample.device, dtype=sample.dtype)

            backbone_input = torch.cat([sample, zeros], dim=1)
            controlnet_sample = backbone_input
        else:
            backbone_input = sample
            controlnet_sample = sample

        # 2. Get ControlNet Residuals
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            sample=controlnet_sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_cond,
            return_dict=False,
        )

        # 3. Run Backbone with Residuals
        return self.unet(
            sample=backbone_input,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            **kwargs,
        )

    def save_pretrained(
        self,
        save_directory: str,
        safe_serialization: bool = False,
    ):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # 1. Save ControlNet
        if self.controlnet is not None:
            self.controlnet.save_pretrained(
                os.path.join(save_directory, "controlnet"), safe_serialization=safe_serialization
            )

        # 2. Save U-Net (The backbone)
        # We must save this because we sliced the input channels
        # from 8 to 4. Standard checkpoints won't match this config anymore.
        if self.unet is not None:
            self.unet.save_pretrained(os.path.join(save_directory, "unet"), safe_serialization=safe_serialization)


def load_controlnet_and_model(model: UNet2DConditionModel, checkpoint_dir: str, cond_image_channels: int):
    """
    Load a full controlnet by performing weight surgery from the saved conditional
    unet or loading the `ControlNetTrainWrapper` directly. This function will also load the weights for the unet
    and set the model weights in place.

    :param model: An unitialized unet instance
    :type model: UNet2DConditionModel
    :param checkpoint_dir: Directory where the checkpoints were saved
    :type checkpoint_dir: str
    :param cond_image_channels: Number of channels for the ControlNet conditioning
    :type cond_image_channels: int
    """
    logger.info("Initializing ControlNet...")
    import glob

    # Flags to track what we loaded
    loaded_wrapper = False
    sketch_weights = None
    controlnet = None

    # 1. Try to find and load an existing checkpoint
    if os.path.exists(checkpoint_dir):
        checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint-*")))
        if checkpoints:
            latest_ckpt = checkpoints[-1]
            ckpt_path = os.path.join(latest_ckpt, "model.pt")

            if os.path.exists(ckpt_path):
                logger.info(f"Analyzing checkpoint {ckpt_path}...")
                loaded_obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)

                # If we loaded the wrapper, we have both U-Net and ControlNet ready to go.
                if isinstance(loaded_obj, ControlNetTrainWrapper):
                    logger.info(" - Detected ControlNetTrainWrapper. Resuming training directly.")
                    model = loaded_obj.unet
                    controlnet = loaded_obj.controlnet
                    loaded_wrapper = True

                # If we loaded a U-Net, we need to check for surgery.
                else:
                    logger.info(" - Detected U-Net/StateDict.")

                    # Extract dictionary if it's a model object
                    if isinstance(loaded_obj, nn.Module):
                        sd = loaded_obj.state_dict()
                    else:
                        sd = loaded_obj

                    # Check if the checkpoint is > 4-channels (Combined)
                    # TODO Change this 4 to out_channels
                    sd_in_channels = sd["conv_in.weight"].shape[1]
                    if "conv_in.weight" in sd and sd_in_channels > 4:
                        logger.info(f" - Found {sd_in_channels}-channel weights. Performing Weight Surgery...")
                        full_weight = sd["conv_in.weight"]

                        # Save sketch weights for transplant
                        sketch_weights = full_weight[:, 4:, :, :]

                        # Slice U-Net weights to 4 channels
                        sd["conv_in.weight"] = full_weight[:, :4, :, :]
                        model.load_state_dict(sd, strict=False)
                    else:
                        # Standard 4-channel load
                        logger.info(" - Found standard 4-channel weights.")
                        model.load_state_dict(sd, strict=False)

    # 2. Initialize ControlNet (Only if we didn't load the wrapper)
    if not loaded_wrapper or controlnet is None:
        logger.info("Creating fresh ControlNet model...")
        controlnet = ControlNetModel.from_unet(
            model,
            conditioning_channels=cond_image_channels,
            load_weights_from_unet=True,  # Copy the trained backbone weights
        )

        # Setup the custom embedding layer
        embedding_dim = model.config.block_out_channels[0]
        controlnet.controlnet_cond_embedding = nn.Conv2d(
            in_channels=cond_image_channels, out_channels=embedding_dim, kernel_size=3, padding=1
        )

        # 3. Transplant Weights (Surgery) or Zero-Init
        if sketch_weights is not None:
            logger.info(" - Transplanting sliced sketch weights to Adapter.")
            controlnet.controlnet_cond_embedding.weight.data = sketch_weights
            nn.init.zeros_(controlnet.controlnet_cond_embedding.bias)
        else:
            logger.info(" - Zero-initializing Adapter.")
            nn.init.zeros_(controlnet.controlnet_cond_embedding.weight)
            nn.init.zeros_(controlnet.controlnet_cond_embedding.bias)

    return controlnet, model
