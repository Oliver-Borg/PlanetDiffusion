import itertools
import torch
from torch.utils.data import DataLoader
from ...core.folder_dataset import FolderDataset
from planetAI.src.data.utils import PlanetConfig
from planetAI.src.data.dataset import RAMDataset
from .evaluate import mse, mae, fid
from pytorch_msssim import ssim
import lpips
from dataclasses import dataclass, field
from ...core.dataclass_argparser import CustomArgumentParser
from tqdm import tqdm


@dataclass
class UpscaleRegArgs:
    planet_size_2_folder: str = field(
        default="./evaluation/Planet-size-2-rgb-rivers/final",
        metadata={"help": "Path to Planet-size-2 experiment outputs"},
    )
    output_file_name: str = field(
        default="dem_output", metadata={"help": "Output file type to evaluate"}
    )
    target_file_name: str = field(
        default="dem_real", metadata={"help": "Target file type to evaluate"}
    )
    num_samples: int = field(
        default=200, metadata={"help": "Number of samples to evaluate"}
    )


def evaluate_upscale_strategy(planet_cfg: PlanetConfig, args: UpscaleRegArgs):
    """Compare different upscale regularization strategies"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn_alex = lpips.LPIPS(net="alex").to(device)

    # Load Planet-size-2 experiment outputs
    dataset = FolderDataset(
        args.planet_size_2_folder,
        inference_args_filter="tile_size:32",
        required_files=[
            "args.json",
            f"{args.output_file_name}.png",
            f"{args.target_file_name}.png",
        ],
        resize_shape=(256, 256),
    )

    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True
    )

    # Configure strategies to test
    blur_radii = [5, 10, 20]
    blur_amounts = [5, 10, 20, 40]

    modes = ["median", "median+gauss"]
    strategies = list(itertools.product(modes, blur_radii, blur_amounts))
    strategies.append(("none", 5, 5))
    metrics = {
        str(s): {
            "sat_mse": 0,
            "sat_mae": 0,
            "sat_ssim": 0,
            "sat_lpips": 0,
            "sat_fid": 0,
            "dem_mse": 0,
            "dem_mae": 0,
            "dem_ssim": 0,
            "dem_lpips": 0,
            "dem_fid": 0,
        }
        for s in strategies
    }

    fid_accumulators = {
        str(s): {
            "sat_outputs": [],
            "sat_inputs": [],
            "dem_outputs": [],
            "dem_inputs": [],
        }
        for s in strategies
    }

    # Create RAM dataset for input data
    ram_dataset = RAMDataset(
        planet_cfg,
        target_image_channels=planet_cfg.output_channels(),
        cond_image_channels=planet_cfg.input_channels(),
        normalise=True,
        tile_size=256,
        mode="test",
    )

    count = 0
    for item in tqdm(dataloader, total=min(len(dataloader), args.num_samples)):
        if count >= args.num_samples:
            break

        sat_output = item["sat_output"].to(device)
        dem_output = item["dem_output"].to(device)

        # Test each strategy
        for strategy in strategies:
            mode, blur_radius, blur_amount = strategy
            planet_cfg.upscale_reg_strat = mode
            planet_cfg.max_blur_radius = blur_radius
            planet_cfg.max_blur_amount = blur_amount
            strategy = str(strategy)

            # Get upscaled input using current strategy
            ram_item = ram_dataset[count]
            upscale_input: torch.Tensor = ram_item["cond_image"].to(device)
            sat_real = upscale_input[:3].unsqueeze(0)
            dem_real = upscale_input[3:4].unsqueeze(0)

            metrics[strategy]["sat_mse"] += mse(sat_output, sat_real).item()
            metrics[strategy]["sat_mae"] += mae(sat_output, sat_real).item()
            metrics[strategy]["sat_ssim"] += ssim(
                sat_output, sat_real, data_range=1
            ).item()
            metrics[strategy]["sat_lpips"] += loss_fn_alex(sat_output, sat_real).item()
            metrics[strategy]["dem_mse"] += mse(dem_output, dem_real).item()
            metrics[strategy]["dem_mae"] += mae(dem_output, dem_real).item()
            metrics[strategy]["dem_ssim"] += ssim(
                dem_output, dem_real, data_range=1
            ).item()
            metrics[strategy]["dem_lpips"] += loss_fn_alex(dem_output, dem_real).item()

            # Accumulate tensors for FID calculation
            fid_accumulators[strategy]["sat_outputs"].append(
                sat_output.expand(-1, 3, -1, -1)
            )
            fid_accumulators[strategy]["sat_inputs"].append(
                sat_real.expand(-1, 3, -1, -1)
            )
            fid_accumulators[strategy]["dem_outputs"].append(
                dem_output.expand(-1, 3, -1, -1)
            )
            fid_accumulators[strategy]["dem_inputs"].append(
                dem_real.expand(-1, 3, -1, -1)
            )

        count += 1

    # Average metrics and calculate FID
    for strategy in tqdm(strategies):
        # Calculate FID scores
        strategy = str(strategy)
        sat_outputs = torch.cat(fid_accumulators[strategy]["sat_outputs"], dim=0)
        sat_inputs = torch.cat(fid_accumulators[strategy]["sat_inputs"], dim=0)
        dem_outputs = torch.cat(fid_accumulators[strategy]["dem_outputs"], dim=0)
        dem_inputs = torch.cat(fid_accumulators[strategy]["dem_inputs"], dim=0)

        metrics[strategy]["sat_fid"] = fid(sat_outputs, sat_inputs)
        metrics[strategy]["dem_fid"] = fid(dem_outputs, dem_inputs)
        print(f"\n{strategy}:")
        value = metrics[strategy]["sat_fid"]
        print(f"sat_fid: {value:.4f}")
        value = metrics[strategy]["dem_fid"]
        print(f"dem_fid: {value:.4f}")

        # Average other metrics
        for metric in metrics[strategy]:
            if not metric.endswith("fid"):
                metrics[strategy][metric] /= count

    # Print results
    print(f"\nResults over {count} samples:")
    for strategy in strategies:
        strategy = str(strategy)
        print(f"\n{strategy}:")
        for metric, value in metrics[strategy].items():
            if "fid" not in metric:
                continue
            print(f"{metric}: {value:.4f}")


def main():
    parser = CustomArgumentParser(
        (PlanetConfig, UpscaleRegArgs),
        description="Evaluate upscale regularization strategies",
    )

    planet_cfg, args = parser.parse_args_into_dataclasses()
    evaluate_upscale_strategy(planet_cfg, args)


if __name__ == "__main__":
    main()
