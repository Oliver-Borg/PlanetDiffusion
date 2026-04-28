import argparse
from copy import deepcopy
import logging
from pathlib import Path
import json
import os
import math

import wandb

from .inference_tests import TileGenerator, recursive_merge, generate_configs
try:
    wandb.login()
except Exception:
    if 'WANDB_API_KEY' in os.environ:
        wandb.login(key=os.environ['WANDB_API_KEY'])
    else:
        raise Exception("WANDB_API_KEY not found in environment variables. https://wandb.ai/authorize")


logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Terrain Generation Interface")
    parser.add_argument(
        "--wandb_config",
        action="store_true",
        help="Download the latest config from W&B instead of using local file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="inference_config.json",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--num_images", type=int, default=100, help="Number of images to generate"
    )
    parser.add_argument(
        "--experiment_names",
        type=str,
        default=["Planet"],
        help="Names of the experiments",
        nargs="+",
    )
    parser.add_argument(
        "--check_num_outputs",
        type=bool,
        default=False,
        help="Check the number of already generated outputs",
    )
    parser.add_argument(
        "--catch_errors",
        type=bool,
        default=False,
        help="Catch errors to prevent crashes",
    )
    parser.add_argument(
        "--base_output_folder",
        type=str,
        help="Base output folder to add to store results",
        default="./evaluation",
    )
    parser.add_argument(
        "--base_model_folder",
        type=str,
        help="Base model folder to load models from",
        default="./models",
    )
    parser.add_argument(
        "--batch_size_factor",
        type=float,
        help="Factor to multiply the batch size by for inference",
        default=1.0,
    )
    parser.add_argument(
        "--no_upload",
        action="store_true",
        help="Don't upload results"
    )
    parser.add_argument(
        "--upload_only",
        action="store_true",
        help="Only upload results"
    )
    parser.add_argument(
        "--radius_power",
        type=int,
        help="The power of two to change the radius by",
        default=0,
    )
    parser.add_argument(
        "--real_sketch_names",
        type=str,
        default=[],
        nargs="+",
    )
    parser.add_argument(
        "--no_two_phase",
        action="store_true",
        help="Disable two phase generation",
    )
    parser.add_argument(
        "--no_sketch_rivers",
        action="store_true",
        help="Disable derivation of rivers from sketch",
    )

    args = parser.parse_args()

    use_wandb_config = args.wandb_config

    if use_wandb_config:
        logging.info("Fetching latest config from W&B Artifacts...")
        api = wandb.Api()
        artifact = api.artifact("PlanetAI/inference-configs:latest")
        download_dir = artifact.download()
        config_path = Path(download_dir) / "inference_config.json"
    else:
        artifact = None
        config_path = Path(args.config)

    # Load config and get experiment parameters
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    for experiment_name in args.experiment_names:
        merged_config = recursive_merge(
            config["default"], config["experiments"], experiment_name
        )
        run = wandb.init(project="PlanetAI", name=f"{experiment_name}-inference", config=merged_config)
        if use_wandb_config and run and artifact:
            run.use_artifact(artifact)

        all_experiment_configs = generate_configs(merged_config)
        for i, merged_config in enumerate(all_experiment_configs):
            experiment_params = merged_config["experiment"]
            merged_config = deepcopy(merged_config)

            merged_config["inference_args"]["output_folder"] = os.path.join(
                args.base_output_folder, merged_config["inference_args"]["output_folder"]
            )
            merged_config["inference_args"]["diffusion_model_dir"] = os.path.join(
                args.base_model_folder, merged_config["inference_args"]["diffusion_model_dir"]
            )
            merged_config["upscaling_args"]["output_folder"] = os.path.join(
                args.base_output_folder, merged_config["upscaling_args"]["output_folder"]
            )
            merged_config["upscaling_args"]["diffusion_model_dir"] = os.path.join(
                args.base_model_folder, merged_config["upscaling_args"]["diffusion_model_dir"]
            )
            merged_config["inference_args"]["batch_size"] = max(
                int(merged_config["inference_args"]["batch_size"] * args.batch_size_factor), 1
            )
            merged_config["upscaling_args"]["batch_size"] = max(
                int(merged_config["upscaling_args"]["batch_size"] * args.batch_size_factor), 1
            )

            print(f"Normal output folder: {merged_config['inference_args']['output_folder']}")
            print(f"Upscaling output folder: {merged_config['upscaling_args']['output_folder']}")
            print(f"Normal model folder: {merged_config['inference_args']['diffusion_model_dir']}")
            print(f"Upscaling model folder: {merged_config['upscaling_args']['diffusion_model_dir']}")
            wandb.log({"inference_progress": i/len(all_experiment_configs)})
            try:
                generator = TileGenerator(
                    experiment_name=experiment_name,
                    config=merged_config,
                    output_size=experiment_params["output_size"],
                    seed=experiment_params["seed"],
                    mars=experiment_params["mars"],
                    shuffle_landcover=experiment_params["shuffle_landcover"],
                    temp_variance=experiment_params["temp_variance"],
                    do_upscaling=experiment_params["do_upscaling"],
                    num_images=int(math.ceil(args.num_images / len(all_experiment_configs))),
                    upload_results=(i == len(all_experiment_configs) - 1) and not args.no_upload,
                    upload_only=args.upload_only,
                    check_num_outputs=args.check_num_outputs,
                    size_offset=args.radius_power,
                )
                if experiment_params["real_sketches"] is True:
                    generator.generate_real_sketches(
                        "./user_sketches",
                        args.real_sketch_names,
                        two_phase=not args.no_two_phase,
                        sketch_rivers=not args.no_sketch_rivers
                    )
                else:
                    generator.generate_grid()
            except Exception as e:
                if args.catch_errors:
                    logging.error(e)
                else:
                    raise e
        wandb.log({"inference_progress": 1.0})
        if run:
            run.finish()
