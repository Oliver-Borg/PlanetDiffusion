---
license: mit
library_name: diffusers
pipeline_tag: image-to-image
tags:
  - terrain
  - planet
  - diffusion
  - procedural-generation
---

# PlanetDiffusion

Pretrained weights for *Authoring Terrestrial Planets with Diffusion Models*
([doi:10.1111/cgf.70390](http://doi.org/10.1111/cgf.70390)).
Code: https://github.com/Oliver-Borg/PlanetDiffusion

## Models

| Folder | Description |
|---|---|
| `PlanetDiffusion/final` | Tile diffusion model (`TerrainDiffusionPipeline`, UNet + DDIM scheduler + terrain encoder), planet size 5. |

Each folder mirrors the local layout `models/<name>/final`. Only inference files are included (no optimizer state).

## Usage

The code downloads the weights automatically when `models/PlanetDiffusion/final` is missing:

```python
from src.models.diffusion.inpaint_inference import load_pipeline

pipeline = load_pipeline("models/PlanetDiffusion/final")
```

To download manually:

```sh
huggingface-cli download otrolie/PlanetDiffusion --include "PlanetDiffusion/final/*" --local-dir models
```

The model is loaded through the custom `TerrainDiffusionPipeline` in the code repository and cannot be loaded with the stock `diffusers.DiffusionPipeline`.

## License

MIT, as in the code repository.
