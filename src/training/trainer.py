
from functools import wraps
import os
import json
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import logging

import re
import shutil

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler

from diffusers import DDIMScheduler, DDPMScheduler

try:
    from line_profiler import profile
except:
    from planetAI.src.data.utils import profile

from tqdm.auto import tqdm, trange

from .ema import EMA
from .metrics import Metric
from ..core.utils import pad_or_crop_tensor
from ..core.shared import ArgsKwargsWrapper
import wandb
from accelerate import Accelerator
import psutil
from time import time
from torchvision.transforms import ToPILImage

# Default file names
MODEL_FILE_NAME = 'model.pt'
OPTIMIZER_FILE_NAME = 'optimizer.pt'
SCHEDULER_FILE_NAME = 'scheduler.pt'
EMA_FILE_NAME = 'ema.pt'
SCALER_FILE_NAME = 'scaler.pt'

TRAINER_STATE_FILE_NAME = 'trainer_state.json'

CHECKPOINT_PREFIX = 'checkpoint-'
CHECKPOINT_REGEX = fr'{CHECKPOINT_PREFIX}(\d+)'


def run_if(predicate):
    # https://stackoverflow.com/a/16358316/13989043
    def wrapper(f):
        @wraps(f)
        def wrapped(self, *f_args, **f_kwargs):
            if predicate(self):
                f(self, *f_args, **f_kwargs)
        wrapped.is_callback = True
        return wrapped

    return wrapper


def cycle(it):
    while True:
        yield from it


def regex_search(text, pattern, group=1, default=None):
    match = re.search(pattern, text)
    return match.group(group) if match else default


def list_checkpoints(model_dir):
    checkpoints = {}
    for f in os.listdir(model_dir):
        checkpoint_dir = os.path.join(model_dir, f)
        if not os.path.isdir(checkpoint_dir):
            continue

        match = regex_search(f, CHECKPOINT_REGEX)
        if match is None:
            continue  # No match, ignore

        l = int(match)
        checkpoints[l] = checkpoint_dir

    return checkpoints


def get_latest_checkpoint(model_dir):
    if model_dir is None or not os.path.exists(model_dir):
        return None

    checkpoints = list_checkpoints(model_dir)

    if not checkpoints:
        return None

    return checkpoints[max(checkpoints)]


@dataclass
class TrainingArguments:
    num_workers: int = field(
        default=1,
        metadata={
            'help': 'How many subprocesses to use for data loading'
        }
    )

    persistent_workers: bool = field(
        default=True,
        metadata={
            'help': 'Whether to keep workers alive between data loader iterations'
        }
    )

    lr: float = field(
        default=1e-4,
        metadata={
            'help': 'Learning rate'
        }
    )

    weight_decay: float = field(
        default=1e-2,
        metadata={
            'help': 'Weight decay for optimizer. Set to > 0 for L2 regularization'
        }
    )

    loss_fn: str = field(
        default='mse',
        metadata={
            'help': 'Loss function to use (loss for noisy high remaining timesteps)',
            'choices': ['mse', 'lpips', 'huber', 'noise', 'mae']
        }
    )

    perceptual_loss_fn: str = field(
        default='mse',
        metadata={
            'help': 'Second loss function to use (loss for clearer low remaining timesteps)',
            'choices': ['mse', 'lpips', 'huber', 'noise', 'mae']
        }
    )

    weighted_loss_fns: bool = field(
        default=False,
        metadata={
            'help': 'Whether to use timestep weighted combined loss functions'
        }
    )

    lpips_scale_factor: float = field(
        default=0.1,
        metadata={
            'help': 'Scale factor for LPIPS loss to put it more in line with MSE loss'
        }
    )

    num_epochs: int = field(
        default=10000,
        metadata={
            'help': 'Number of training epochs'
        }
    )
    auto_num_epochs: int = field(
        default=None,
        metadata={
            'help': 'Base number of training epochs. Used to automatically set num_epochs based on tile size'
        }
    )
    save_steps: int = field(
        default=5000,
        metadata={
            'help': 'Save checkpoint every X steps'
        }
    )
    upload_steps: int = field(
        default=250000,
        metadata={
            'help': 'Upload checkpoint every X steps'
        }
    )
    save_limit: Optional[int] = field(
        default=3,
        metadata={
            'help': 'Maximum number of checkpoints to store at a given time (None = no limit)'
        }
    )
    save_on_disk: bool = field(
        default=True,
        metadata={
            'help': 'Whether to save checkpoints and samples on disk'
        }
    )
    log_steps: int = field(
        default=1000,
        metadata={
            'help': 'Log every X steps'
        }
    )
    eval_steps: int = field(
        default=100000,
        metadata={
            'help': 'Run an evaluation every X steps'
        }
    )

    seed: Optional[int] = field(
        default=None,
        metadata={
            'help': 'Set a seed'
        }
    )

    batch_size: int = field(
        default=1,
        metadata={
            'help': 'The batch size'
        }
    )

    sampling_batch_size: int = field(
        default=None,
        metadata={
            'help': 'The sampling batch size'
        }
    )

    results_dir: str = field(
        default='results_dir',
        metadata={
            'help': 'Where to save results'
        }
    )

    checkpoint: str = field(
        default='latest',
        metadata={
            'help': 'Load model weights and information from this checkpoint'
        }
    )

    amp: bool = field(
        default=True,
        metadata={
            'help': 'Whether to train using automatic mixed precision'
        }
    )

    # Additional training arguments
    use_ema: bool = field(
        default=False,
        metadata={
            'help': 'Whether to train using an exponential moving average model'
        }
    )
    ema_update_every: int = field(
        default=10,
        metadata={
            'help': 'Update EMA model (if specified) every X steps'
        }
    )
    ema_decay: float = field(
        default=0.995,
        metadata={
            'help': 'EMA decay rate'
        }
    )
    lr_scheduler_type: str = field(
        default='plateau',
        metadata={
            'help': 'Learning rate scheduler type',
            'choices': ['plateau', 'cosine']
        }
    )
    # Learning rate scheduler arguments
    lr_scheduler_factor: float = field(
        default=0.8,
        metadata={
            'help': 'Factor by which the learning rate will be reduced'
        }
    )
    lr_scheduler_patience: int = field(
        default=10,
        metadata={
            'help': 'Number of epochs with no improvement after which learning rate will be reduced'
        }
    )
    lr_scheduler_threshold: float = field(
        default=1e-4,
        metadata={
            'help': 'Threshold for measuring the new optimum, to only focus on significant changes'
        }
    )
    lr_scheduler_min_lr: float = field(
        default=1e-6,
        metadata={
            'help': 'A lower bound on the learning rate'
        }
    )
    lr_scheduler_verbose: bool = field(
        default=True,
        metadata={
            'help': 'Whether to print out when learning rate changes'
        }
    )
    num_warmup_steps: int = field(
        default=500,
        metadata={
            'help': 'Number of steps for the warmup phase. Only applicable for cosine scheduler.'
        }
    )
    clip_grad_norm: bool = field(
        default=True,
        metadata={
            'help': 'Whether to clip gradients'
        }
    )
    conditioning_dropout: float = field(
        default=0.1,
        metadata={
            'help': 'Dropout probability for conditioning images. Used to train classifier free guidance.'
        }
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            'help': 'Number of gradient accumulation steps. Used to increase effective batch size.'
        }
    )
    extra_metrics: bool = field(
        default=False,
        metadata={
            'help': 'Whether to calculate extra metrics (e.g. FID, LPIPS) during testing'
        }
    )
    pretrained_model_dir: Optional[str] = field(
        default=None,
        metadata={
            'help': 'Path to pretrained model directory. If None, check results_dir or train from scratch.'
        }
    )
    use_wandb: bool = field(
        default=True,
        metadata={
            'help': 'Whether to use wandb for logging'
        }
    )
    experiment_name: str = field(
        default=None,
        metadata={
            'help': 'Name of experiment for wandb'
        }
    )
    tile_size: int = field(
        default=256,
        metadata={
            'help': 'Size of tiles to train on'
        }
    )
    sample_tile_size: int = field(
        default=256,
        metadata={
            'help': 'Size of tiles to sample'
        }
    )
    auto_batch_size: int = field(
        default=None,
        metadata={
            'help': 'The auto batch size to use to adjust batch size based on tile size'
        }
    )
    use_accelerate: bool = field(
        default=True,
        metadata={
            'help': 'Whether to use accelerate for training'
        }
    )

    def __post_init__(self):
        compute_factor = (self.tile_size ** 2) / 256 ** 2
        if self.auto_batch_size is not None:
            self.batch_size = int(round(self.auto_batch_size / compute_factor))
            print(
                f"Batch size adjusted {self.auto_batch_size} -> {self.batch_size} because tile size is {self.tile_size}"
            )
        if self.auto_num_epochs is not None:
            self.num_epochs = int(round(self.auto_num_epochs / compute_factor))
            print(
                f"Num epochs adjusted {self.auto_num_epochs} -> {self.num_epochs} because tile size is {self.tile_size}"
            )

        sample_batch_size = self.batch_size
        training_batch_scale = self.batch_size * self.tile_size ** 2

        while (
            sample_batch_size * self.sample_tile_size ** 2 >= training_batch_scale
            and sample_batch_size > 1
        ):
            sample_batch_size = sample_batch_size // 2
        if self.sampling_batch_size != sample_batch_size:
            print(
                f"Sampling batch size adjusted {self.sampling_batch_size} -> {sample_batch_size} "
                f"because sampling tile size is {self.sample_tile_size}"
            )
        self.sampling_batch_size = sample_batch_size


class ModelInputs(ArgsKwargsWrapper):
    pass


class ModelTargets:
    def __init__(self, targets) -> None:
        self.targets = targets


class BaseTrainer:

    def __init__(self,
                 model: nn.Module,
                 datasets: Dict[str, Dataset],
                 optimizer: optim.Optimizer,
                 criterion,
                 perceptual_criterion,
                 pipeline=None,
                 scheduler=None,

                 metrics=None,
                 #  transform_input=None,
                 #  transform_targets=None,
                 transform_output=None,
                 load_model_weights: bool = True,
                 training_args: Optional[TrainingArguments] = None
                 ) -> None:

        self.model = model
        self.pipeline = pipeline

        self.optimizer = optimizer
        self.criterion = criterion
        self.perceptual_criterion = perceptual_criterion
        self.scheduler = scheduler

        # self.transform_inputs = transform_input
        # self.transform_targets = transform_targets
        self.transform_outputs = transform_output

        self.wandb_metrics = {}
        self.sample_losses = []
        self.last_sample_loss = None
        self.metrics = []
        if metrics is not None:
            for met in metrics:
                if isinstance(met, Metric):
                    self.metrics.append(met)
                elif isinstance(met, type) and issubclass(met, Metric):
                    # Instantiate object of class
                    self.metrics.append(met())
                else:
                    raise ValueError(
                        f'Metric ({met}) is not an instance or subclass of the Metric class')

        # Use default training arguments if not specified
        if training_args is None:
            training_args = TrainingArguments()
        self.training_args = training_args

        self.ema_model = None
        if training_args.use_ema:
            self.ema_model = EMA(self.model,
                                 beta=training_args.ema_decay,
                                 update_every=training_args.ema_update_every)

        self.scaler = GradScaler(enabled=training_args.amp)

        # Set datasets
        if not isinstance(datasets, dict):
            self.datasets = {'train': datasets}
        self.datasets = datasets

        self.dataloaders = {}
        for key, dataset in self.datasets.items():
            self.dataloaders[key] = DataLoader(
                dataset,
                batch_size=(
                    self.training_args.batch_size
                    if key == 'train'
                    else self.training_args.sampling_batch_size
                    ),
                shuffle=True, # key == 'train'
                num_workers=self.training_args.num_workers,
                persistent_workers=self.training_args.num_workers > 0 and self.training_args.persistent_workers,
                generator=torch.Generator().manual_seed(
                self.training_args.seed) if self.training_args.seed is not None else None
            )
        self.accelerator = None
        if self.training_args.use_accelerate:
            self.accelerator = Accelerator(gradient_accumulation_steps=self.training_args.gradient_accumulation_steps,
                                mixed_precision='fp16' if self.training_args.amp else 'no')
            self.model, self.optimizer, self.dataloaders['train'], self.dataloaders['valid'], self.dataloaders['test'], self.scheduler = self.accelerator.prepare(
                self.model, 
                self.optimizer, 
                self.dataloaders['train'], 
                self.dataloaders['valid'], 
                self.dataloaders['test'], 
                self.scheduler
            )
            if training_args.use_ema:
                self.ema_model = self.accelerator.prepare(self.ema_model)
        checkpoint = training_args.checkpoint
        # For training, or if no model found in model_dir
        use_trainer_data = False
        if checkpoint == 'latest':  # Infer checkpoint
            # 1. Try finding subfolders in pretrained_model_dir
            checkpoint = get_latest_checkpoint(self.training_args.pretrained_model_dir)

            # 2. If subfolder search failed, check if the dir is the checkpoint
            if checkpoint is None and self.training_args.pretrained_model_dir:
                if os.path.exists(os.path.join(self.training_args.pretrained_model_dir, MODEL_FILE_NAME)):
                    checkpoint = self.training_args.pretrained_model_dir

            # 3. If still None, look in results_dir (Resuming)
            if checkpoint is None:
                checkpoint = get_latest_checkpoint(self.training_args.results_dir)
                use_trainer_data = True

        self.load_checkpoint(checkpoint, use_trainer_data, load_model=load_model_weights)

        self.total_train_steps = self.training_args.num_epochs * \
            len(self.dataloaders['train'])
        self._register_callbacks()

    @property
    def device(self):
        if self.accelerator is not None:
            return self.accelerator.device
        return next(self.model.parameters()).device

    def _register_callbacks(self):
        self._callbacks = []
        for name in dir(self):
            item = getattr(self, name)
            if callable(item) and hasattr(item, '__self__') and hasattr(item, 'is_callback') and item.is_callback:
                self._callbacks.append(item)

    def _run_callbacks(self):
        for cb in self._callbacks:
            cb()

    def try_load_state_dict(self, path, obj, try_transfer_learning=True):
        if not os.path.exists(path) or obj is None:
            return

        print(' - loading', path)

        # Load model parameters
        data = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(data, nn.Module):
            data = data.state_dict()

        self._load_state_dict(obj, data, try_transfer_learning)

    def _load_state_dict(self, obj, state_dict, try_transfer_learning):
        if isinstance(obj, nn.Module):
            if try_transfer_learning:
                missing_keys = []
                resized_keys = []
                remapped_keys = []

                for name, param in obj.named_parameters():
                    key_to_load = name

                    # 1. Remap old prefixes
                    if name not in state_dict:
                        # If model expects "unet.conv_in" but checkpoint has "conv_in"
                        if name.startswith("unet.") and name.replace("unet.", "") in state_dict:
                            key_to_load = name.replace("unet.", "")
                            remapped_keys.append(f"{name} <-- {key_to_load}")

                        # If model expects "controlnet.conv_in" but checkpoint has "conv_in"
                        # (Useful if you ever load a standalone ControlNet checkpoint)
                        elif name.startswith("controlnet.") and name.replace("controlnet.", "") in state_dict:
                            key_to_load = name.replace("controlnet.", "")
                            remapped_keys.append(f"{name} <-- {key_to_load}")

                    # 2. Load the data
                    if key_to_load in state_dict:
                        # Check shape mismatch
                        if state_dict[key_to_load].shape != param.shape:
                            resized_keys.append(f"{name}: {state_dict[key_to_load].shape} -> {param.shape}")
                            # Resize
                            for dim, num in enumerate(param.shape):
                                state_dict[key_to_load] = pad_or_crop_tensor(
                                    tensor=state_dict[key_to_load],
                                    dimension=dim,
                                    desired_num_values=num,
                                )

                        # Apply to the object
                        # We must access the parameter directly because 'name' in obj might differ
                        # from 'key_to_load' in state_dict
                        with torch.no_grad():
                            param.copy_(state_dict[key_to_load])

                    else:
                        # 3. Just use original init otherwise instead of noise
                        # We assume manual initialization (from_pretrained / from_unet) has already happened.
                        missing_keys.append(name)
                        # state_dict[name] = torch.randn(param.shape)

                # --- Reporting ---
                if len(remapped_keys) > 0:
                    print(f"\n[Loader] REMAPPED {len(remapped_keys)} keys (Prefix Mismatch Solved):")
                    for k in remapped_keys[:5]:
                        print(f"  - {k}")

                if len(resized_keys) > 0:
                    print(f"\n[Loader] RESIZED {len(resized_keys)} layers:")
                    for k in resized_keys[:5]:
                        print(f"  - {k}")

                if len(missing_keys) > 0:
                    print(f"\n[Loader] SKIPPED {len(missing_keys)} keys (Preserved Manual Init):")
                    # This is expected for ControlNet layers when loading a UNet-only checkpoint
                    print(f"  - Example: {missing_keys[0]}")
                    print("-" * 60)

            else:
                # Standard strict loading
                obj.load_state_dict(state_dict, strict=True)

        elif isinstance(obj, optim.Optimizer):
            try:
                obj.load_state_dict(state_dict)
            except ValueError:
                if try_transfer_learning:
                    print('WARNING | Ignoring optimizer due to parameter mismatch')
                else:
                    raise
        else:
            obj.load_state_dict(state_dict)

    def load_checkpoint(self, checkpoint, use_trainer_data=True, load_model=True):

        trainer_data = None
        if checkpoint is not None:  # Checkpoint specified, or found above
            print('Loading checkpoint:', checkpoint)

            if os.path.isdir(checkpoint):
                model_path = os.path.join(checkpoint, MODEL_FILE_NAME)

                for file, module in (
                    (OPTIMIZER_FILE_NAME, self.optimizer),
                    (SCHEDULER_FILE_NAME, self.scheduler),
                    (EMA_FILE_NAME, self.ema_model),
                    (SCALER_FILE_NAME, self.scaler),
                ):
                    self.try_load_state_dict(
                        os.path.join(checkpoint, file), module)

            else:
                model_path = checkpoint

            # Only load model weights if explicitly requested
            if load_model:
                self.try_load_state_dict(model_path, self.model)
            else:
                print("Skipping model weight loading (load_model=False)")

            # Try to load trainer state
            trainer_state = os.path.join(checkpoint, TRAINER_STATE_FILE_NAME)
            if os.path.exists(trainer_state):
                with open(trainer_state) as fp:
                    trainer_data = json.load(fp)

        if trainer_data is not None and use_trainer_data:
            self.steps = trainer_data['steps']
            self.epoch = trainer_data['epoch']
            self.history = trainer_data['history']

        else:  # No trainer_data, set to defaults
            self.steps = 0
            self.epoch = 0
            self.history = []

    def save(self):
        if self.training_args.save_limit is not None and os.path.exists(self.training_args.results_dir):
            # Remove if saving this model will exceed save_limit
            checkpoints = list_checkpoints(self.training_args.results_dir)
            if len(checkpoints) >= self.training_args.save_limit:
                to_remove = len(checkpoints) - self.training_args.save_limit + 1
                to_remove = min(to_remove, len(checkpoints))
                for k in sorted(checkpoints)[:to_remove]:
                    shutil.rmtree(checkpoints[k])
        if not self.training_args.save_on_disk:
            return None

        checkpoint_dir = os.path.join(
            self.training_args.results_dir, f'{CHECKPOINT_PREFIX}{self.steps}')
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save model separately
        # Note: We save the full model for easier inference later on (stores creation parameters)
        model_path = os.path.join(checkpoint_dir, MODEL_FILE_NAME)
        try:
            torch.save(self.model, model_path)
        except:
            torch.save(self.accelerator.unwrap_model(self.model), model_path)

        # Save optimizer
        # https://stackoverflow.com/questions/70768868/pytorch-whats-the-purpose-of-saving-the-optimizer-state
        optimizer_path = os.path.join(checkpoint_dir, OPTIMIZER_FILE_NAME)
        torch.save(self.optimizer.state_dict(), optimizer_path)

        # Save scheduler
        if self.scheduler is not None:
            scheduler_path = os.path.join(checkpoint_dir, SCHEDULER_FILE_NAME)
            torch.save(self.scheduler.state_dict(), scheduler_path)

        # Save EMA
        if self.ema_model is not None:
            ema_model_path = os.path.join(checkpoint_dir, EMA_FILE_NAME)
            torch.save(self.ema_model.state_dict(), ema_model_path)

        # Save Scaler
        if self.scaler is not None:
            scaler_path = os.path.join(checkpoint_dir, SCALER_FILE_NAME)
            torch.save(self.scaler.state_dict(), scaler_path)

        # Save trainer state
        trainer_state = os.path.join(checkpoint_dir, TRAINER_STATE_FILE_NAME)
        with open(trainer_state, 'w') as fp:
            json.dump({
                'steps': self.steps,
                'epoch': self.epoch,
                'history': self.history,
                # 'lr': self.train_lr,
                # 'total_num_epochs': self.num_epochs
            }, fp)

        return checkpoint_dir

    @property
    def current_lr(self):
        # https://discuss.pytorch.org/t/get-current-lr-of-optimizer-with-adaptive-lr/24851/3
        if self.scheduler is not None and hasattr(self.scheduler, 'get_last_lr'):
            return self.scheduler.get_last_lr()

        for param_group in self.optimizer.param_groups:
            return param_group['lr']

        return 0  # No learning rate found (should never reach here)

    def add_to_history(self, type, **kwargs):
        self.history.append({
            'type': type,
            'epoch': self.epoch,
            'steps': self.steps,
            'lr': self.current_lr,
            **kwargs
        })

    def extract(self, item):
        raise NotImplementedError

    def extract_next(self, dataloader) -> Tuple[ModelInputs, ModelTargets, torch.Tensor]:
        model_input, target, mask = self.extract(next(dataloader))
        if not isinstance(model_input, ModelInputs):
            model_input = ModelInputs(model_input)
        if not isinstance(target, ModelTargets):
            target = ModelTargets(target)

        # TODO move both to correct device

        return model_input, target, mask

    def step(self, dataloader, phase='train'):
        update_gradients = phase == 'train'
        update_metrics = phase != 'train'

        if update_gradients:
            # Zero gradients for every batch
            self.optimizer.zero_grad()

        with torch.set_grad_enabled(update_gradients):
            model_inputs, targets, masks = self.extract_next(dataloader)
                        

            # https://pytorch.org/docs/stable/notes/amp_examples.html#typical-mixed-precision-training
            # Runs the forward pass with autocasting.
            with autocast(device_type=self.device.type, enabled=self.training_args.amp):
                outputs = self.model(
                *model_inputs.args, **model_inputs.kwargs)
                if self.transform_outputs is not None:
                    outputs = self.transform_outputs(outputs)
                mask_ratios = torch.ones(outputs.shape[0], device=outputs.device)
                if masks is not None:
                    masks = masks.to(outputs.device)
                    outputs = outputs * masks + (1 - masks) * targets.targets
                    mask_ratios = masks[:, 0].mean(axis=(1, 2))
                    # Minimum should be 0.2
                    mask_ratios[mask_ratios < 0.2] = 1.0
                    # targets.targets = targets.targets * masks
                loss = self.criterion(outputs, targets.targets)
                try:
                    scaled_loss = (self.criterion(outputs, targets.targets, reduction='none').mean(axis=(1,2,3)) / mask_ratios).mean()
                except:
                    scaled_loss = loss
                loss = scaled_loss

            if update_metrics:
                for metric in self.metrics:
                    # Add outputs and labels to each metric
                    metric.add(
                        outputs=outputs,
                        targets=targets.targets
                    )

            if update_gradients:
                # Scales loss. Calls backward() on scaled loss to create scaled gradients.
                # Backward ops run in the same dtype autocast chose for corresponding forward ops.
                self.scaler.scale(loss).backward()

                # scaler.step() first unscales the gradients of the optimizer's assigned params.
                # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                # otherwise, optimizer.step() is skipped.
                self.scaler.step(self.optimizer)

                # Updates the scale for next iteration.
                self.scaler.update()

                if self.ema_model is not None:
                    self.ema_model.update()

        return loss.item()

    def get_best_loss(self) -> float:
        min_loss_json = os.path.join(self.training_args.results_dir, 'min_loss.json')
        if not os.path.exists(min_loss_json):
            return float("infinity")
        try:
            with open(min_loss_json, 'r') as f:
                return float(json.load(f).get("best_loss"))
        except Exception:
            return float("infinity")

    def save_best_loss(self, loss: float):
        min_loss_json = os.path.join(self.training_args.results_dir, 'min_loss.json')
        with open(min_loss_json, 'w') as f:
            json.dump({"best_loss": loss}, f)

    def upload_artifact(self, paths: list[str]):
        artifact = wandb.Artifact(name="PlanetAI-Model", type="model")
        for path in paths:
            artifact.add_dir(path)
        wandb.log_artifact(artifact)
        artifact.wait()

    def predict_x0_from_noise(self, x_t, t, pred_noise):
        """
        x_t        : Noisy image at step t
        t          : Timesteps (tensor of shape [B])
        pred_noise : Model prediction of ε
        """
        scheduler: DDIMScheduler | DDPMScheduler = self.noise_scheduler
        alphas_cumprod = scheduler.alphas_cumprod.to(self.device)
        sqrt_alpha_t = alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_t = (1 - alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
        x0_pred = (x_t[:, :pred_noise.shape[1]] - sqrt_one_minus_alpha_t * pred_noise) / sqrt_alpha_t
        return x0_pred

    @profile
    def train(self):
        train_dataloader = self.dataloaders['train']

        if self.steps >= self.total_train_steps:
            return

        do_eval = 'valid' in self.dataloaders
        do_test = 'test' in self.dataloaders

        loss_steps = 0
        running_losses = []

        dl = cycle(train_dataloader)
        with tqdm(initial=self.steps, total=self.total_train_steps) as progress:

            while self.steps < self.total_train_steps:
                t1 = time()
                self.model.train()  # Set model to training mode
                if self.accelerator is None:
                    loss = self.step(dl)
                    running_losses.append(loss)
                else:
                    with self.accelerator.accumulate(self.model):
                        # running_loss += self.step(dl)
                        dlt1 = time()
                        model_inputs, targets, masks = self.extract_next(dl)
                        dlt2 = time()
                        outputs = self.model(*model_inputs.args, **model_inputs.kwargs)
                        if self.transform_outputs is not None:
                            outputs = self.transform_outputs(outputs)
                        if self.training_args.loss_fn == "lpips" or self.training_args.perceptual_loss_fn == "lpips":
                            targets.targets = self.predict_x0_from_noise(
                                model_inputs.kwargs["sample"], model_inputs.kwargs['timestep'], targets.targets
                            )
                            outputs = self.predict_x0_from_noise(
                                model_inputs.kwargs["sample"], model_inputs.kwargs['timestep'], outputs
                            )
                        mask_ratios = torch.ones(outputs.shape[0], device=outputs.device)
                        if masks is not None:
                            masks = masks.to(outputs.device)
                            outputs = outputs * masks + (1 - masks) * targets.targets
                            mask_ratios = masks[:, 0].mean(axis=(1, 2))
                            # Minimum should be 0.2
                            mask_ratios[mask_ratios < 0.2] = 1.0
                            # targets.targets = targets.targets * masks
                        if self.training_args.weighted_loss_fns:
                            # Pass identity function as reduction
                            loss = self.criterion(outputs, targets.targets, reduction='none')
                            while len(loss.shape) > 1:
                                loss = loss.mean(axis=-1)
                            loss_2 = self.perceptual_criterion(outputs, targets.targets, reduction='none')
                            while len(loss_2.shape) > 1:
                                loss_2 = loss_2.mean(axis=-1)
                            total_steps = self.noise_scheduler.config.num_train_timesteps
                            timesteps = model_inputs.kwargs['timestep']
                            timestep_weights = timesteps/total_steps
                            weighted_loss = (loss * timestep_weights + loss_2 * (1 - timestep_weights)).mean()
                            loss = weighted_loss
                        else:
                            loss = (
                                self.criterion(outputs, targets.targets)
                                + self.perceptual_criterion(outputs, targets.targets)
                            ) / 2

                        running_losses.append(loss.item())
                        self.accelerator.backward(loss)
                        if self.training_args.clip_grad_norm and self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                        loss_steps += 1
                        if self.ema_model is not None:
                            self.ema_model.update()
                        self.optimizer.zero_grad()
                starting_index = max(len(running_losses) - self.training_args.log_steps, 0)
                running_losses = running_losses[starting_index:]
                
                avg_loss = sum(running_losses) / len(running_losses)
                if avg_loss == float('nan'):
                    print('Loss is nan. Stopping training.')
                    break
                progress.set_description(
                    f'Training | epoch={self.epoch:.2f}, loss={avg_loss:.5f}')
                # This shouldn't be done with reduce on plateau lr scheduler
                # if self.scheduler is not None and not isinstance(self.scheduler, ReduceLROnPlateau):
                #     self.scheduler.step()
                # Update progress
                progress.update()
                self.epoch = self.steps / len(train_dataloader)
                self.steps += 1
                log_lr = self.current_lr
                if type(log_lr) == list:
                    log_lr = log_lr[0]   
                t2 = time() 
                total_time = t2 - t1
                dataloader_time = dlt2 - dlt1
                dataloader_pcnt = dataloader_time / total_time * 100
                factor = (self.sat_latent_scaling_factor + self.dem_latent_scaling_factor) / 2
                self.wandb_metrics = {'train_loss': avg_loss, 'epoch': self.epoch, 'lr': log_lr, 
                                      'latent_scaled_train_loss': avg_loss / (factor or 1),
                                      'gpu_memory': torch.cuda.memory_allocated()/1024**3,
                                      'RAM': psutil.Process(os.getpid()).memory_info().rss/1024**3,
                                      'time': t2 - t1, 'speed': 1/(t2-t1), 'dataloader_time': dataloader_time,
                                      'dataloader_pcnt': dataloader_pcnt} 
                last_step = self.steps == self.total_train_steps

                # TODO register callbacks:
                if last_step or self.steps % self.training_args.log_steps == 0:
                    self.add_to_history('log', loss=avg_loss)

                    # Reset loss
                    running_loss = 0
                    loss_steps = 0

                if do_eval and (last_step or self.steps % self.training_args.eval_steps == 0):
                    eval_metrics = self.eval()

                    self.add_to_history('eval', **eval_metrics)

                    # Reduce learning rate if no improvement in eval loss
                    if self.scheduler is not None and isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(eval_metrics['loss'])

                do_artifact_upload = self.steps % self.training_args.upload_steps == 0
                if do_test and last_step:
                    # Run testing just before saving final model
                    test_metrics = self.eval(mode='test')
                    self.add_to_history('test', **test_metrics)

                if (
                    (last_step or self.steps % self.training_args.save_steps == 0 or do_artifact_upload) and
                    (self.training_args.save_on_disk)
                ):
                    progress.set_description('Saving model')
                    self.save()
                    final_model_dir = os.path.join(self.training_args.results_dir, 'final-model')
                    final_dir = os.path.join(self.training_args.results_dir, 'final')
                    best_dir = os.path.join(self.training_args.results_dir, 'best')
                    if self.model is not None:
                        self.model.save_pretrained(final_model_dir)
                    if self.pipeline is not None:
                        self.pipeline.save_pretrained(final_dir)
                        try:
                            min_loss = self.get_best_loss()
                            if self.last_sample_loss is not None and self.last_sample_loss < min_loss:
                                self.save_best_loss(float(self.last_sample_loss))
                                self.pipeline.save_pretrained(best_dir)
                        except Exception as e:
                            print('Error saving best model: ', e)
                    if do_artifact_upload:
                        self.upload_artifact([final_model_dir, final_dir])
                self._run_callbacks()
                wandb.log(self.wandb_metrics)

    def eval(self, mode='valid'):
        self.model.eval()

        dl = iter(self.dataloaders[mode])
        running_loss = 0
        info_header = 'Testing' if mode == 'test' else 'Validation'
        eval_info = f'{info_header} | epoch={self.epoch:.2f}, steps={self.steps}'

        # Reset metrics
        for metric in self.metrics:
            metric.reset()

        with trange(len(self.dataloaders[mode])) as progress, torch.no_grad():
            for i in range(len(self.dataloaders[mode])):
                running_loss += self.step(dl, phase='eval')

                eval_metrics = {
                    'loss': running_loss/(i+1),
                }

                if i == len(self.dataloaders[mode]) - 1:
                    # Only add these metrics at the end
                    for metric in self.metrics:
                        eval_metrics.update(metric.total())

                progress.set_description(
                    f'{eval_info}, {self.format_metrics(eval_metrics)}')
                progress.update()
                factor = (self.sat_latent_scaling_factor + self.dem_latent_scaling_factor) / 2
                self.wandb_metrics.update({f'{mode}_loss': eval_metrics['loss'], 
                                           f'latent_scaled_{mode}_loss': eval_metrics['loss'] / (factor or 1),
                                           'epoch': self.epoch})
        
        if self.training_args.extra_metrics and mode == 'test':
            extra_metrics = eval_metrics
            self.wandb_metrics.update(extra_metrics)
        return eval_metrics

    @staticmethod
    def format_metrics(metrics_dict: Dict):
        """Helper method to format metrics dictionary"""
        items = [
            f'{k}={v:.5f}'
            for k, v in metrics_dict.items()
        ]
        return ', '.join(items)
