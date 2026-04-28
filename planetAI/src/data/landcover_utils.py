from enum import Enum
from dataclasses import dataclass
import numpy as np
from random import shuffle


class LandcoverName(Enum):
    OPEN_WATER = "Ocean"
    ANTARCTICA = "Antarctica"
    TREE_COVER = "Tree cover"
    GRASSLAND = "Grassland"
    BARE = "Bare"
    SHRUBLAND = "Shrubland"
    CROPLAND = "Cropland"
    SNOW_AND_ICE = "Snow and ice"
    WATER = "Water"
    WETLAND = "Wetland"
    BUILT_UP = "Built-up"
    MARS = "Mars"


@dataclass
class Landcover:
    display_name: str
    gray_colour: int
    index: int
    display_colour: tuple[int, int, int]
    alternative_class: LandcoverName
    display_hex: str


class LandcoverClasses:
    OPEN_WATER = Landcover(
        display_name=LandcoverName.OPEN_WATER.value,
        gray_colour=0,
        index=0,
        display_colour=(0, 0, 0),
        alternative_class=LandcoverName.OPEN_WATER,
        display_hex="#000000",
    )
    ANTARCTICA = Landcover(
        display_name=LandcoverName.ANTARCTICA.value,
        gray_colour=25,
        index=1,
        display_colour=(192, 255, 255),
        alternative_class=LandcoverName.SNOW_AND_ICE,
        display_hex="#c0ffff",
    )
    TREE_COVER = Landcover(
        display_name=LandcoverName.TREE_COVER.value,
        gray_colour=50,
        index=2,
        display_colour=(51, 160, 44),
        alternative_class=LandcoverName.TREE_COVER,
        display_hex="#33a02c",
    )
    GRASSLAND = Landcover(
        display_name=LandcoverName.GRASSLAND.value,
        gray_colour=75,
        index=3,
        display_colour=(253, 191, 111),
        alternative_class=LandcoverName.GRASSLAND,
        display_hex="#fdbf6f",
    )
    BARE = Landcover(
        display_name=LandcoverName.BARE.value,
        gray_colour=100,
        index=4,
        display_colour=(251, 154, 153),
        alternative_class=LandcoverName.BARE,
        display_hex="#fb9a99",
    )
    SHRUBLAND = Landcover(
        display_name=LandcoverName.SHRUBLAND.value,
        gray_colour=125,
        index=5,
        display_colour=(178, 223, 138),
        alternative_class=LandcoverName.SHRUBLAND,
        display_hex="#b2df8a",
    )
    CROPLAND = Landcover(
        display_name=LandcoverName.CROPLAND.value,
        gray_colour=150,
        index=6,
        display_colour=(255, 127, 0),
        alternative_class=LandcoverName.CROPLAND,
        display_hex="#ff7f00",
    )
    SNOW_AND_ICE = Landcover(
        display_name=LandcoverName.SNOW_AND_ICE.value,
        gray_colour=175,
        index=7,
        display_colour=(166, 206, 227),
        alternative_class=LandcoverName.SNOW_AND_ICE,
        display_hex="#a6cee3",
    )
    WATER = Landcover(
        display_name=LandcoverName.WATER.value,
        gray_colour=200,
        index=8,
        display_colour=(31, 120, 180),
        alternative_class=LandcoverName.WATER,
        display_hex="#1f78b4",
    )
    WETLAND = Landcover(
        display_name=LandcoverName.WETLAND.value,
        gray_colour=225,
        index=9,
        display_colour=(0, 150, 160),
        alternative_class=LandcoverName.TREE_COVER,
        display_hex="#0096a0",
    )
    BUILT_UP = Landcover(
        display_name=LandcoverName.BUILT_UP.value,
        gray_colour=250,
        index=10,
        display_colour=(250, 0, 0),
        alternative_class=LandcoverName.CROPLAND,
        display_hex="#fa0000",
    )
    MARS = Landcover(
        display_name=LandcoverName.MARS.value,
        gray_colour=255,
        index=11,
        display_colour=(227, 26, 28),
        alternative_class=LandcoverName.MARS,
        display_hex="#e31a1c",
    )


landcover_mapping = {
    LandcoverName.OPEN_WATER: LandcoverClasses.OPEN_WATER,
    LandcoverName.ANTARCTICA: LandcoverClasses.ANTARCTICA,
    LandcoverName.TREE_COVER: LandcoverClasses.TREE_COVER,
    LandcoverName.GRASSLAND: LandcoverClasses.GRASSLAND,
    LandcoverName.BARE: LandcoverClasses.BARE,
    LandcoverName.SHRUBLAND: LandcoverClasses.SHRUBLAND,
    LandcoverName.CROPLAND: LandcoverClasses.CROPLAND,
    LandcoverName.SNOW_AND_ICE: LandcoverClasses.SNOW_AND_ICE,
    LandcoverName.WATER: LandcoverClasses.WATER,
    LandcoverName.WETLAND: LandcoverClasses.WETLAND,
    LandcoverName.BUILT_UP: LandcoverClasses.BUILT_UP,
    LandcoverName.MARS: LandcoverClasses.MARS,
}


def translate_land(landcover: np.ndarray, single_water_class: bool = False) -> np.ndarray:
    new_landcover = landcover.copy()
    for classname, landcover_class in landcover_mapping.items():
        alt_class = landcover_class.alternative_class
        if single_water_class and classname == LandcoverName.OPEN_WATER:
            alt_class = LandcoverName.WATER
        if classname == alt_class:
            continue
        old_colour = landcover_class.gray_colour
        new_colour = landcover_mapping[alt_class].gray_colour
        if old_colour in landcover:
            new_landcover[landcover == old_colour] = new_colour
    return new_landcover


def randomize_land(landcover: np.ndarray, single_water_class: bool = False) -> np.ndarray:
    landcover = translate_land(landcover, single_water_class)
    source_classes = [
        landcover_mapping[x] for x in used_classes
        if x not in [LandcoverName.OPEN_WATER, LandcoverName.WATER]
    ]
    dest_classes = source_classes.copy()
    shuffle(dest_classes)
    for src, dest in zip(source_classes, dest_classes):
        if src == LandcoverClasses.OPEN_WATER or dest == LandcoverClasses.OPEN_WATER:
            # Don't set to ocean as ocean can't have a DEM value > 0
            continue
        landcover[landcover == src.gray_colour] = dest.gray_colour
    # We have to translate because some of the random dest classes may be invalid
    return translate_land(landcover, single_water_class)


def landcover_index_map(index: int) -> Landcover:
    for name, landcover in landcover_mapping.items():
        if landcover.index == index:
            return landcover
    raise ValueError(f"Index {index} out of range")


used_classes = [
    LandcoverName.OPEN_WATER,
    LandcoverName.WATER,
    LandcoverName.TREE_COVER,
    LandcoverName.GRASSLAND,
    LandcoverName.BARE,
    LandcoverName.SHRUBLAND,
    LandcoverName.CROPLAND,
    LandcoverName.SNOW_AND_ICE,
    LandcoverName.MARS,
]


def landcover_class_list():
    return [landcover_mapping[c].display_name for c in used_classes]


def landcover_class_colours():
    return [landcover_mapping[c].display_colour for c in used_classes]


def land_to_gray(im: np.ndarray) -> np.ndarray:
    """
    Convert a three channel landcover image to a grayscale image
    Classes:

    Args:
        im (np.ndarray): The 3 channel image
    Returns:
        np.ndarray: The grayscale image
    """
    integer = np.zeros((im.shape[0], im.shape[1]), dtype=np.uint8)
    for i, c in enumerate(landcover_mapping.values()):
        integer[np.all(im == c.display_colour, axis=2)] = c.gray_colour
    return integer


class LandcoverEnum(Enum):
    OPEN_WATER = 0
    ANTARCTICA = 1
    TREE_COVER = 2
    GRASSLAND = 3
    BARE = 4
    SHRUBLAND = 5
    CROPLAND = 6
    SNOW_AND_ICE = 7
    WATER = 8
    WETLAND = 9
    BUILT_UP = 10
    MARS = 11


def gray_to_land(im: np.ndarray, old_classes: bool = False) -> np.ndarray:
    """
    Convert a grayscale landcover image to a 3 channel image
    Classes:

    Args:
        im (np.ndarray): The grayscale image
        old_classes (bool): Use the old classes
    Returns:
        np.ndarray: The 3 channel image
    """
    gray = im.copy()
    if len(im.shape) == 3:
        gray = gray[:, :, 0]
    mars_mask = gray == 255
    rgb = np.zeros((im.shape[0], im.shape[1], 3), dtype=np.uint8)
    for name, landcover in landcover_mapping.items():
        rgb[gray == landcover.gray_colour] = landcover_mapping[
            landcover.alternative_class
        ].display_colour
    rgb[mars_mask] = landcover_mapping[LandcoverName.MARS].display_colour
    return rgb
