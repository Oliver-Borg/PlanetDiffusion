from PIL import Image
import numpy as np

from .inference_controller import quad_rivers
from ...interface.utils import SketchArgs


from planetAI.src.data.utils import PlanetConfig, np_rgb
from planetAI.src.data.atlas_loader import AtlasLoader, QuadAtlasLoader


if __name__ == "__main__":
    planet_cfg = PlanetConfig(size=3, downscale_offset=3)
    quad_atlas_loader = QuadAtlasLoader(planet_cfg)
    atlas_loader = AtlasLoader(planet_cfg)
    dem = quad_atlas_loader.quad_dem

    sketches = SketchArgs(
        atlas_loader.downsketch,
        atlas_loader.downland_sketch,
        atlas_loader.downtemp_sketch,
    )

    rivers = quad_rivers(
        sketches,
        planet_cfg,
        dem,
    )

    viridis_rivers = np_rgb(rivers, cmap="viridis")
    rgb_dem = np.dstack([dem] * 3)
    rivers = np.dstack([rivers] * 3)

    rgb_dem[rivers > 10] = viridis_rivers[rivers > 10]

    Image.fromarray(rgb_dem.astype(np.uint8)).save("quad_rivers_derived.png")
