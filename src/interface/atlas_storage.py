from typing import Callable

import numpy as np
import torch

from .interface_types import AtlasType, ATLAS_DISPLAY_FUNC_MAPPING, identity

from planetAI.src.data.utils import PlanetConfig, timing
from planetAI.src.data.atlas_loader import AtlasLoader


def atlas_display_func(atlas_type: AtlasType) -> Callable[[np.ndarray], np.ndarray]:
    if atlas_type == AtlasType.EARTH_RIVERS:
        return lambda x: x.astype(np.uint8) * 255
    if atlas_type == AtlasType.RIVERS:
        return lambda x: x.astype(np.uint8) * 25
    if atlas_type == AtlasType.SELECTION_MASK:
        return lambda x: x.astype(np.uint8) * 255
    return ATLAS_DISPLAY_FUNC_MAPPING.get(atlas_type, identity)


class AtlasStorage:
    def __init__(self, initial_map: dict[str, np.ndarray | torch.TensorType] = {}):
        self._atlas_map = initial_map

    def set(self, atlas_type: AtlasType, atlas: np.ndarray | torch.TensorType) -> None:
        self._atlas_map[atlas_type.value] = atlas

    def get(self, atlas_type: AtlasType) -> np.ndarray | torch.TensorType | None:
        return self._atlas_map.get(atlas_type.value)

    def _load_fast_earth_data(self, data_dir: str):
        planet_cfg = PlanetConfig(data_dir=data_dir)
        atlas_loader = AtlasLoader(planet_cfg)
        self.set(
            AtlasType.EARTH_LANDCOVER_SKETCH, np.dstack([atlas_loader.downland_sketch, atlas_loader.downtemp_sketch])
        )

    @timing
    def load_earth_data(
        self,
        data_dir: str,
    ) -> None:
        planet_cfg = PlanetConfig(data_dir=data_dir)
        atlas_loader = AtlasLoader(planet_cfg)
        # TODO Consider making this more lazy
        self.set(AtlasType.EARTH_DEM, atlas_loader.dem)
        self.set(AtlasType.EARTH_SAT, atlas_loader.sat)
        self.set(AtlasType.EARTH_RIVERS, atlas_loader.rivers)
        self.set(AtlasType.EARTH_DEM_SKETCH, atlas_loader.downsketch)
        self.set(
            AtlasType.EARTH_LANDCOVER_SKETCH, np.dstack([atlas_loader.downland_sketch, atlas_loader.downtemp_sketch])
        )
        self.set(AtlasType.EARTH_TEMPERATURE_SKETCH, atlas_loader.downtemp_sketch)
        self.set(
            AtlasType.EARTH_MODAL_STACK,
            np.dstack([atlas_loader.downmodal_sketch, atlas_loader.downland_sketch, atlas_loader.downtemp_sketch]),
        )
        self.set(AtlasType.EARTH_MODAL_RIVER_SKETCH, atlas_loader.modal_river_sketch)
        print("Earth data loaded successfully.")

    def available_types(self) -> list[AtlasType]:
        return [AtlasType(key) for key in self._atlas_map.keys()]


ATLAS_STORAGE = AtlasStorage()
