from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np

from planetAI.src.data.utils import hex_str_to_rgb, np_rgb
from planetAI.src.data.noise_settings import EdgeDistanceSettings, NoiseSettings, MeanFilterSettings
from planetAI.src.data.utils import gray_to_land
from src.interface.panels.Icons import (
    add_icons_to_modal_sketch,
    add_icons_to_temperature_sketch,
    get_temperature_icon_overlay,
    get_component_temperature_icon_overlay,
    get_temperature_line_overlay,
    get_circle_icon_overlay,
)


class AtlasType(Enum):
    """
    Enum for different types of atlas' to display
    """
    EARTH_DEM = "Earth DEM"
    EARTH_SAT = "Earth Satellite"
    EARTH_RIVERS = "Earth River Sketch"
    EARTH_DEM_SKETCH = "Earth DEM Sketch"
    EARTH_LANDCOVER_SKETCH = "Earth Landcover Sketch"
    EARTH_TEMPERATURE_SKETCH = "Earth Temperature Sketch"
    EARTH_MODAL_STACK = "Earth Preview"
    EARTH_MODAL_RIVER_SKETCH = "Earth Modal River Sketch"
    DEM = "Generated DEM"
    SAT = "Generated Satellite"
    RIVERS = "River Sketch"
    DEM_SKETCH = "DEM Sketch"
    SELECTION_MASK = "Selection Mask"
    UNCERTAINTY_MASK = "Uncertainty Mask"
    FILTERED_SELECTION_MASK = "Filtered Selection Mask"
    LANDCOVER_SKETCH = "Landcover Sketch"
    TEMPERATURE_SKETCH = "Temperature Sketch"
    MODAL_STACK = "Preview"
    NONE = "None"


def landcover_display_func(
    landcover_stack: np.ndarray,
    icon_size: int = 20,
    icon_spacing: int = 20,
    temperature_outlines: bool = False,
    landcover_shading_factor: float = 1.0,
    use_components: bool = False,
    use_lines: bool = False,
    icon_brightness: float = 0.2,
    use_circles: bool = False,
):
    assert landcover_stack.shape[2] == 2
    landcover_sketch = landcover_stack[:, :, 0]
    temperature_sketch = landcover_stack[:, :, 1]
    if use_lines:
        temperature_icons = get_temperature_line_overlay(
            temperature_sketch, icon_size, icon_spacing, temperature_outlines
        ) > 0
    elif use_circles:
        temperature_icons = get_circle_icon_overlay(
            temperature_sketch, icon_size, icon_spacing, temperature_outlines
        ) > 0
    elif use_components:
        temperature_icons = get_component_temperature_icon_overlay(
            temperature_sketch, landcover_sketch, icon_size, icon_spacing, temperature_outlines
        ) > 0
    else:
        temperature_icons = get_temperature_icon_overlay(
            temperature_sketch, icon_size, icon_spacing, temperature_outlines
        ) > 0
    coloured_landcover = gray_to_land(landcover_sketch)
    temperature_sketch = np.dstack([temperature_sketch] * 3)
    factor = landcover_shading_factor
    coloured_landcover = factor * coloured_landcover + (1 - factor) * temperature_sketch

    coloured_landcover[temperature_icons] = coloured_landcover[temperature_icons] * (1 + icon_brightness)
    return coloured_landcover.clip(0, 255).astype(np.uint8)


def dem_display_func() -> Callable[[np.ndarray], np.ndarray]:
    return lambda dem: np_rgb(dem, cmap="gist_earth", max_pixel=255, min_pixel=64)


def identity(noise: np.ndarray) -> np.ndarray:
    return noise


def modal_stack_display_func(modal_stack: np.ndarray):
    modal_sketch = modal_stack[:, :, :3]
    temp_sketch = modal_stack[:, :, 4]
    return add_icons_to_modal_sketch(modal_sketch, temp_sketch)


ATLAS_DISPLAY_FUNC_MAPPING = {
    AtlasType.EARTH_DEM: dem_display_func(),
    AtlasType.EARTH_SAT: identity,
    AtlasType.EARTH_RIVERS: identity,
    AtlasType.EARTH_DEM_SKETCH: identity,
    AtlasType.EARTH_LANDCOVER_SKETCH: landcover_display_func,
    AtlasType.EARTH_TEMPERATURE_SKETCH: add_icons_to_temperature_sketch,
    AtlasType.EARTH_MODAL_STACK: modal_stack_display_func,
    AtlasType.DEM: dem_display_func(),
    AtlasType.SAT: identity,
    AtlasType.RIVERS: identity,
    AtlasType.DEM_SKETCH: identity,
    AtlasType.SELECTION_MASK: identity,
    AtlasType.UNCERTAINTY_MASK: identity,
    AtlasType.FILTERED_SELECTION_MASK: identity,
    AtlasType.LANDCOVER_SKETCH: gray_to_land,
    AtlasType.TEMPERATURE_SKETCH: add_icons_to_temperature_sketch,
    AtlasType.MODAL_STACK: modal_stack_display_func,
    AtlasType.NONE: identity,
}


class EventTypeEnum(Enum):
    DEM = "dem"
    DEM_SKETCH = "dem_sketch"
    TEMP = "temperature"
    TEMP_PRESET = "temperature_preset"
    LANDCOVER = "landcover"
    RIVER = "river"
    RIVER_UPA = "river_upa"
    MODAL = "modal"
    REGEN = "regen"


class ToolEnum(Enum):
    """
    This is an enumeration class for the Drawing tools.
    """

    BRUSH = 0
    FILL = 1
    LASSO = 2


@dataclass
class EventMetadata:
    type: EventTypeEnum
    x: int
    y: int
    brush_size: int


@dataclass
class SketchEventMetadata(EventMetadata):
    tool: ToolEnum
    erase: bool
    roughness: float


@dataclass
class DEMEventMetadata(EventMetadata):
    brush: bool
    lasso: bool
    noise_settings: list[NoiseSettings]
    noise: np.ndarray
    mean_filter: MeanFilterSettings
    roughness: float
    lock_ocean: bool
    display_sketch: bool
    fill: bool
    preview_coords: tuple[float, float]
    view_coords: tuple[float, float]


class EditTypeEnum(Enum):
    CONFIRM = "confirm"
    CANCEL = "cancel"
    DRAW = "draw"
    OTHER_CHANGE = "other"


@dataclass
class DEMSketchEventMetadata(SketchEventMetadata):
    noise_settings: list[NoiseSettings]
    noise: np.ndarray
    edge_distance_settings: EdgeDistanceSettings
    lock_ocean: bool
    min_value: float
    max_value: float
    threshold: float
    edit_type: EditTypeEnum


@dataclass
class TempPresetEventMetadata(EventMetadata):
    lat_temp_list: list[tuple[float, float]]
    noise_weight: float
    pivot: int
    min_temp: float
    max_temp: float
    noise_settings: NoiseSettings
    noise: np.ndarray
    display_as_sketch: bool
    modal_view: bool
    steepness: float
    factor: float


class LandcoverClass:
    """
    This is a class for the landcover types.
    """
    def __init__(self, name: str, colour: int, displaycolour: str, subclasses: list["LandcoverClass"]):
        """
        Constructor for the LandcoverClass class.
        """
        self.name = name
        self.colour = colour
        self.displaycolour = displaycolour
        self.subclasses = subclasses
        self.textcolour = "white" if sum(hex_str_to_rgb(displaycolour)) < 382 else "black"

    def __eq__(self, value: object) -> bool:
        """
        Override the __eq__ method.
        """
        if not isinstance(value, LandcoverClass):
            return False
        return self.colour == value.colour


@dataclass
class LandCoverEventMetadata(SketchEventMetadata):
    primary_class: LandcoverClass
    primary_subclass: LandcoverClass | None
    secondary_class: LandcoverClass | None
    noise_settings: NoiseSettings
    noise: np.ndarray
    edge_distance_settings: EdgeDistanceSettings
    primary_ratio: float
    lock_ocean: bool
    modal_view: bool
    preview_coords: tuple[float, float]
    view_coords: tuple[float, float]
    edit_type: EditTypeEnum


class RiverDisplayType(Enum):
    MODAL = 0
    DEM = 1
    UPA = 2
    WEIGHTS = 3
    EFFICIENCY = 4


@dataclass
class RiverDerivationSettings:
    # brush_size, threshold_power, iterations, smoothing, full_mapping
    base_weight: float
    noise_factor: float
    smoothing: int
    full_mapping: dict[tuple[int, int, int], float]
    noise: np.ndarray | None = None
    use_sketch: bool = True


@dataclass
class RiverEventMetadata(EventMetadata):
    brush_size: int
    erase: bool
    display_type: RiverDisplayType
    weight_value: float
    efficiency_value: float
    full_mapping: dict[tuple[int, int, int], float]
    river_max: float
    river_min: float
    settings: RiverDerivationSettings


class OutputDisplayType(Enum):
    SATELLITE = 0
    DEM = 1


@dataclass
class RegenEventMetadata(EventMetadata):
    erase: bool
    display_type: OutputDisplayType
