"""Package a trained model with only the files needed for inference and upload it to the HuggingFace Hub.

Example:
    python -m src.misc.upload_weights --model_dir results/Planet-size-5-old --name PlanetDiffusion

Authenticate first with `huggingface-cli login` or the HF_TOKEN environment variable.
Use --dry_run to only build the staging folder.
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import torch
from huggingface_hub import HfApi
from safetensors.torch import save_file

from ..core.model_hub import HF_REPO_ID

MODEL_CARD = Path(__file__).parent / "model_card.md"


def stage_model(model_dir: Path, name: str, staging: Path) -> Path:
    """Copy the inference files to `staging/<name>/final`, converting the UNet to safetensors."""
    src = model_dir / "final"
    dst = staging / name / "final"
    dst.mkdir(parents=True)

    shutil.copy2(src / "model_index.json", dst)
    shutil.copytree(src / "scheduler", dst / "scheduler")

    (dst / "unet").mkdir()
    shutil.copy2(src / "unet" / "config.json", dst / "unet")
    state_dict = torch.load(src / "unet" / "diffusion_pytorch_model.bin", map_location="cpu", weights_only=True)
    save_file({k: v.contiguous() for k, v in state_dict.items()}, dst / "unet" / "diffusion_pytorch_model.safetensors")

    encoder = src / "terrain_encoder"
    if encoder.exists():
        shutil.copytree(encoder, dst / "terrain_encoder")
    return dst


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_dir", required=True, help="Training output folder containing `final/`.")
    parser.add_argument("--name", required=True, help="Subfolder name in the Hub repo.")
    parser.add_argument("--repo_id", default=HF_REPO_ID)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        stage_model(Path(args.model_dir), args.name, staging)
        shutil.copy2(MODEL_CARD, staging / "README.md")
        for f in sorted(staging.rglob("*")):
            if f.is_file():
                print(f"{f.relative_to(staging)}  {f.stat().st_size / 1e6:.1f} MB")
        if args.dry_run:
            return
        api = HfApi()
        print(f"Uploading as {api.whoami()['name']} to {args.repo_id}")
        api.create_repo(args.repo_id, private=args.private, exist_ok=True)
        api.upload_folder(
            repo_id=args.repo_id,
            folder_path=staging,
            commit_message=f"Upload {args.name}",
        )


if __name__ == "__main__":
    main()
