from .train import main
import os
from dataclasses import dataclass, field
import wandb
if 'WANDB_API_KEY' in os.environ:
    wandb.login(key=os.environ['WANDB_API_KEY'])
else:
    raise Exception("WANDB_API_KEY not found in environment variables. https://wandb.ai/authorize")
import json
import traceback

from ...core.dataclass_argparser import CustomArgumentParser
from ..shared.sampling import SamplingArguments
from .train import DatasetArguments, ImageArguments, DiffusionTrainingArguments, DiffusionArguments, SchedulerArguments
from planetAI.src.data.utils import PlanetConfig


def main_with_catch(args):
    def run():
        try:
            main(*args)
        except Exception as e:
            print(traceback.format_exc())
            print(e)
            wandb.log({"error": str(e)})
            exit(1)
        finally:
            wandb.finish()
    return run


@dataclass
class SweepArguments:
    config_folder: str = field(
        default="./sweep_configs",
        metadata={
            "help": "Folder containing sweep configurations."
        }
    )
    sweep_name: str = field(
        default="architecture",
        metadata={
            "help": "Name of the sweep configuration and run."
        }
    )


if __name__ == '__main__':
    parser = CustomArgumentParser(
        (
            DatasetArguments,
            ImageArguments,
            DiffusionTrainingArguments,
            SamplingArguments,
            DiffusionArguments,
            SchedulerArguments,
            PlanetConfig,
            SweepArguments,
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
        PlanetConfig,
        SweepArguments,
    ] = parser.parse_args_into_dataclasses()

    sweep_id_file = os.path.join(args[-1].config_folder, 'sweep_ids.json')

    if not os.path.exists(sweep_id_file):
        with open(sweep_id_file, 'w') as f:
            json.dump({}, f)
    with open(sweep_id_file, 'r') as f:
        sweep_ids: dict = json.load(f)

    config = args[-1].sweep_name

    with open(os.path.join(args[-1].config_folder, f"{config}.json"), "r") as f:
        sweep_config = json.load(f)

    sweep_id = sweep_ids.get(config, None)
    if sweep_id is None:
        sweep_id = wandb.sweep(sweep=sweep_config, project="PlanetAI")
    sweep_ids[config] = sweep_id
    with open(sweep_id_file, 'w') as f:
        json.dump(sweep_ids, f)
    wandb.agent(sweep_id, count=1, function=main_with_catch(args[:-1]), project="PlanetAI")
