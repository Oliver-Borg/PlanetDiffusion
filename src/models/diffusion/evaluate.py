from line_profiler import profile
import lpips
from pytorch_msssim import ssim
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import (
    calculate_frechet_distance,
)
from torch.nn.functional import adaptive_avg_pool2d
from dataclasses import dataclass, field
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from datetime import datetime
import cv2
from PIL import Image

from planetAI.src.data.atlas_loader import AtlasLoader
from planetAI.src.data.landcover_utils import land_to_gray
from planetAI.src.data.modal_sketch import ModalSketch


from ...core.dataclass_argparser import CustomArgumentParser
from ...core.utils import array_to_image
from ...evaluation.metrics import mse, mae
from ...core.folder_dataset import FolderDataset, FullPlanetFolderDataset
from torchmetrics.image.fid import FrechetInceptionDistance
from planetAI.src.data.utils import np_rgb, np_to_tensor, tensor_to_np, timing, PlanetConfig
from planetAI.src.data.sketch_gen import dilate_paint, DEFAULT_BUCKETS, get_buckets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fid_model = FrechetInceptionDistance(feature=2048).to(device)


def upload_artifact(paths: list[str], experiment_name: str = ""):
    import wandb
    try:
        wandb.login()
    except Exception:
        if 'WANDB_API_KEY' in os.environ:
            wandb.login(key=os.environ['WANDB_API_KEY'])
        else:
            raise Exception("WANDB_API_KEY not found in environment variables. https://wandb.ai/authorize")
    wandb.init(project="PlanetAI", name=f"{experiment_name} evaluation")
    artifact = wandb.Artifact(name="Evaluation_results", type="evaluation")
    for path in paths:
        artifact.add_dir(path)
    wandb.log_artifact(artifact)
    artifact.wait()
    print("Done uploading")


@timing
def fid(input_tensors: torch.FloatTensor, output_tensors: torch.FloatTensor):
    # Convert from -1-1 to 0-255
    input_tensors = (input_tensors + 1) * 127.5
    output_tensors = (output_tensors + 1) * 127.5
    # Cast to uint8
    input_tensors = input_tensors.to(torch.uint8).to(device)
    output_tensors = output_tensors.to(torch.uint8).to(device)
    fid_model.update(input_tensors, real=False)
    fid_model.update(output_tensors, real=True)
    # fid_model.update(input_tensors, real=False)
    # fid_model.update(output_tensors, real=True)
    return fid_model.compute().item()


@dataclass
class EvaluationArguments:
    eval_folder: str = field(
        default="./data/evaluation/outputs", metadata={"help": "Path to input folder"}
    )
    use_real_target: bool = field(
        default=True,
        metadata={"help": "Use real terrain as the target rather than PaleoDEM"},
    )
    inference_args_filter: str = field(
        default="",
        metadata={"help": "Filter by inference args key:value pairs (comma separated)"}
    )
    planet_cfg_filter: str = field(
        default="",
        metadata={"help": "Filter by planet config key:value pairs (comma separated)"}
    )
    experiment_name: str = field(
        default="", metadata={"help": "Experiment name for the file name"}
    )
    full_planet_timestamp: str = field(
        default=None, metadata={"help": "Full planet timestamp for evaluation."}
    )
    output_type: str = field(
        default=None, metadata={"help": "Output type", "choices": ["dem", "sat"]}
    )
    use_subfolders: bool = field(
        default=False, metadata={"help": "Use subfolders"}
    )
    max_images: int = field(
        default=10000, metadata={"help": "Maximum number of images to use for evaluation"}
    )
    shuffle_images: bool = field(
        default=True, metadata={"help": "Shuffle image dataloader"}
    )
    do_diversity_eval: bool = field(
        default=False, metadata={"help": "Shuffle the target data for diversity calculation"}
    )


def calculate_fid_given_dataset(
    output_type: str, dataset, batch_size, device, dims, num_workers=1
):
    """Calculates the FID of two paths"""

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    m1, s1 = calculate_activation_statistics_of_dataset(
        dataset, model, f"{output_type}_output", batch_size, dims, device, num_workers
    )
    m2, s2 = calculate_activation_statistics_of_dataset(
        dataset, model, f"{output_type}_real", batch_size, dims, device, num_workers
    )
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


class ElevationDataset(torch.utils.data.Dataset):
    def __init__(self, items, transforms=None):
        self.items = items
        self.transforms = transforms

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def get_activations(dataset, model, image_name, batch_size=50, dims=2048, device='cpu',
                    num_workers=1):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- dataset     : Dataset containing real and fake data
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > len(dataset):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    pred_arr = np.empty((len(dataset), dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch[image_name].to(device)
        batch = torch.clamp((batch + 1.0) / 2.0, 0.0, 1.0)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_activation_statistics_of_dataset(
    dataset, model, image_name, batch_size=50, dims=2048, device="cpu", num_workers=1
):
    act = get_activations(
        dataset, model, image_name, batch_size, dims, device, num_workers
    )
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def elev_to_img(x):
    x = x.cpu().numpy()
    x = np.transpose(x, (1, 2, 0))  # (c, h, w) -> (h, w, c)
    return array_to_image(x, bit_depth=16)


@profile
def main():
    parser = CustomArgumentParser(
        (EvaluationArguments,), description="Run evaluation on a dataset"
    )

    args: tuple[EvaluationArguments] = parser.parse_args_into_dataclasses()
    eval_args: EvaluationArguments = args[0]

    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

    loss_fn_alex = lpips.LPIPS(net="alex").to(device)
    output_type = eval_args.output_type
    if output_type is None:
        raise ValueError("Output type must be provided")

    output_file_name = f"{output_type}_output"
    target_file_name = f"{output_type}_real"

    # Initialize metrics
    running_mse = 0
    running_mae = 0
    running_ssim = 0
    running_ms_ssim = 0
    running_lpips = 0
    running_sketch_loss = 0

    output_tensors = []
    target_tensors = []

    is_blended = (
        "blended" in eval_args.eval_folder.lower() or
        "blended" in eval_args.experiment_name.lower()
    )

    dataset = (
        FullPlanetFolderDataset(eval_args.eval_folder, eval_args.full_planet_timestamp)
        if eval_args.full_planet_timestamp else
        FolderDataset(
            eval_args.eval_folder,
            inference_args_filter=eval_args.inference_args_filter,
            planet_cfg_filter=eval_args.planet_cfg_filter,
            required_files=[
                "args.json",
                f"{output_type}_output.png",
                f"{output_type}_real.png",
                "display.png",
            ],
            is_blended=is_blended,
            use_subfolders=eval_args.use_subfolders,
            max_len=eval_args.max_images,
            force_hflip=False,
        )
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=eval_args.shuffle_images,
        num_workers=0,
        pin_memory=True,
    )

    real_dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    ) if eval_args.do_diversity_eval else dataloader

    dataloader_iter = iter(dataloader)
    real_dataloader_iter = iter(real_dataloader)

    planet_cfg = PlanetConfig()

    planet_cfg = PlanetConfig()
    modal_sketcher = ModalSketch(planet_cfg)
    atlas_loader = AtlasLoader(planet_cfg)

    down_mars_dem = atlas_loader.down_mars_dem
    down_mars_buckets = (
        get_buckets(down_mars_dem, planet_cfg)
        if planet_cfg.bucketing_mode == "uniform"
        else None
    )
    count = 0
    for i in tqdm(range(len(dataset))):
        gen_item = next(dataloader_iter)
        real_item = next(real_dataloader_iter) if eval_args.do_diversity_eval else gen_item
        valid = gen_item["valid"][0].item() and real_item["valid"][0].item()
        if not valid:
            print("Skipping invalid item due to size mismatch")
            continue

        output = gen_item[output_file_name].to(device)
        target = real_item[target_file_name].to(device)

        # sketch = (
        #     item["downsketch"].to(device)
        #     if output_type == "dem"
        #     else item["downmodal_sketch"].to(device)
        # )
        w = output.shape[-1]
        if "display" in gen_item:
            display = gen_item["display"].to(device)
        else:
            display = None
        inference_args = gen_item["args"]
        delta = 2 ** inference_args["planet_cfg"]["downscale_offset"].item()

        if output_type == "dem":

            if display is not None:
                sketch = display[:, :, :, w * 2: w * 3]
                sketch_np: np.ndarray = tensor_to_np(sketch[0])
            else:
                sketch = tensor_to_np(gen_item["sketch"][0, 0])
                sketch_np = np_rgb(sketch, cmap="viridis")

            output_np = tensor_to_np(output[0, 0])
            output_np = cv2.resize(
                output_np, fx=1 / delta, fy=1 / delta, dsize=None, interpolation=cv2.INTER_LANCZOS4
            )

            target_np = tensor_to_np(target[0, 0])
            target_np = cv2.resize(
                target_np, fx=1 / delta, fy=1 / delta, dsize=None, interpolation=cv2.INTER_LANCZOS4
            )

            output_sketch = dilate_paint(
                output_np,
                planet_cfg,
                buckets=DEFAULT_BUCKETS,
            )
            output_sketch = cv2.resize(
                output_sketch, fx=delta, fy=delta, dsize=None, interpolation=cv2.INTER_NEAREST
            )

            mars_output_sketch = dilate_paint(
                output_np,
                planet_cfg,
                buckets=down_mars_buckets,
            )
            mars_output_sketch = cv2.resize(
                mars_output_sketch, fx=delta, fy=delta, dsize=None, interpolation=cv2.INTER_NEAREST
            )

            target_sketch = dilate_paint(
                target_np,
                planet_cfg,
                buckets=DEFAULT_BUCKETS,
            )
            target_sketch = cv2.resize(
                target_sketch, fx=delta, fy=delta, dsize=None, interpolation=cv2.INTER_NEAREST
            )
            # In the display, the sketch is viridis so we just use this
            output_sketch = np_rgb(output_sketch, cmap="viridis")
            mars_output_sketch = np_rgb(mars_output_sketch, cmap="viridis")
            sketch_match = sketch_np == output_sketch
            mars_sketch_match = sketch_np == mars_output_sketch
            running_sketch_loss += min(np.mean(~sketch_match), np.mean(~mars_sketch_match))
        else:

            if display is not None:
                rgb_landcover = tensor_to_np(display[0, :, :, :w])
                temp = tensor_to_np(display[0, 0, :, w: w * 2])
                landcover = land_to_gray(rgb_landcover)
                temp = cv2.resize(
                    temp, fx=1 / delta, fy=1 / delta, dsize=None, interpolation=cv2.INTER_NEAREST
                )
                landcover = cv2.resize(
                    landcover, fx=1 / delta, fy=1 / delta, dsize=None, interpolation=cv2.INTER_NEAREST
                )

                modal_sketch = modal_sketcher.get_sketch(
                    landcover, temp, mars_mask=landcover == 255
                )
                modal_sketch = cv2.resize(
                    modal_sketch, fx=delta, fy=delta, dsize=None, interpolation=cv2.INTER_NEAREST
                )
                sketch = np_to_tensor(modal_sketch).unsqueeze(0).to(device)
            else:
                sketch = gen_item["modal_sketch"].to(device)
            # Just take loss between modal sketch and output
            running_sketch_loss += torch.mean((sketch - output) ** 2).item()

        output_tensors.append(output)
        target_tensors.append(target)

        current_mse = mse(output, target).item()
        current_mae = mae(output, target).item()
        current_ssim = ssim(output, target, data_range=1).item()
        current_ms_ssim = 0  # ms_ssim(output, target, data_range=1).item()
        current_lpips = loss_fn_alex(output, target).item()

        running_mse += current_mse
        running_mae += current_mae
        running_ssim += current_ssim
        running_ms_ssim += current_ms_ssim
        running_lpips += current_lpips
        count += 1

    if count > 0:
        if not eval_args.do_diversity_eval:
            try:
                if count < 2048:
                    fid_score = fid(
                        torch.cat(output_tensors, dim=0), torch.cat(target_tensors, dim=0)
                    )
                else:
                    del output_tensors
                    del target_tensors
                    with torch.no_grad():
                        torch.cuda.empty_cache()

                    fid_score = calculate_fid_given_dataset(
                        output_type,
                        dataset,
                        batch_size=25,
                        device=device,
                        dims=2048,
                    )
            except Exception as e:
                print(f"An error occurred while calculating FID {e}")
                fid_score = 1000
        else:
            fid_score = 1000

        print(f"Evaluated {count} images:")
        print(f"{output_type} MSE: {running_mse/count:.4f}")
        print(f"{output_type} MAE: {running_mae/count:.4f}")
        print(f"{output_type} SSIM: {running_ssim/count:.3f}")
        print(f"{output_type} MS-SSIM: {running_ms_ssim/count:.3f}")
        print(f"{output_type} LPIPS: {running_lpips/count:.3f}")
        print(f"{output_type} Sketch Loss: {running_sketch_loss/count:.4f}")
        print(f"{output_type} FID: {fid_score:.3f}")
        os.makedirs("evaluation_results", exist_ok=True)
        # Save results to a json file
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"evaluation_results/{eval_args.experiment_name}_{run_timestamp}_{output_type}_metrics.json"
        with open(file_name, "w") as f:
            json.dump({
                "experiment_name": eval_args.experiment_name,
                "eval_folder": eval_args.eval_folder,
                "output_type": output_type,
                "num_images": count,
                "inference_args_filter": eval_args.inference_args_filter,
                "planet_cfg_filter": eval_args.planet_cfg_filter,
                "mse": running_mse/count,
                "mae": running_mae/count,
                "ssim": running_ssim/count,
                "ms_ssim": running_ms_ssim/count,
                "lpips": running_lpips/count,
                "sketch_loss": running_sketch_loss/count,
                "fid": fid_score
            }, f, indent=4)
        upload_artifact(["evaluation_results"], experiment_name=eval_args.experiment_name)


if __name__ == "__main__":
    main()
