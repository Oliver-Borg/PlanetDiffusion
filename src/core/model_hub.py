"""Download pretrained model weights from the HuggingFace Hub on demand.

All released models live in one repo with one subfolder per model, and each subfolder
mirrors the local layout (`models/<name>/final/...`).
"""

import os
from pathlib import Path

HF_REPO_ID = "otrolie/PlanetDiffusion"

# Local model name -> subfolder in the HF repo. Aliases cover older names used in configs.
MODELS = {
    "PlanetDiffusion": "PlanetDiffusion",
    "Planet-size-5-old": "PlanetDiffusion",
}


def ensure_model(model_final_dir: str | os.PathLike) -> str:
    """Return `model_final_dir` (a `<models>/<name>/final` folder), downloading it from the Hub if it is missing.

    Folders that do not belong to a released model are returned unchanged.
    """
    final_dir = Path(model_final_dir)
    if final_dir.exists():
        return str(final_dir)
    name = final_dir.parent.name
    if final_dir.name != "final" or name not in MODELS:
        return str(final_dir)

    from huggingface_hub import snapshot_download

    subfolder = MODELS[name]
    models_root = final_dir.parent.parent
    print(f"{final_dir} not found, downloading '{subfolder}' from https://huggingface.co/{HF_REPO_ID}")
    snapshot_download(
        repo_id=HF_REPO_ID,
        allow_patterns=[f"{subfolder}/final/*"],
        local_dir=models_root,
    )
    downloaded = models_root / subfolder / "final"
    if downloaded != final_dir:
        # Alias: place the download under the requested name
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(downloaded.resolve(), final_dir, target_is_directory=True)
    return str(final_dir)
