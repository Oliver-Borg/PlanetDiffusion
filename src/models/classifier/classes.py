
from enum import IntEnum, auto

# Filter buildings using satellite imagery
# Filter artefacts/recording errors from derivative

class FilterClass(IntEnum):
    @classmethod
    def idx_to_class(cls):
        return {x.value: x.name for x in cls}

    @classmethod
    def class_to_idx(cls):
        return {x.name: x.value for x in cls}

    @classmethod
    def valid(cls):
        raise NotImplementedError

    @classmethod
    def invalid(cls):
        raise NotImplementedError

    @classmethod
    def class_split(cls):
        """Return tuple indicating which classes can be used for elevation data"""
        valid = cls.valid()
        invalid = cls.invalid()
        assert len(valid) + len(invalid) == len(cls)
        return valid, invalid


class DerivativeFilterClass(FilterClass):
    VALID = 0
    PIXELATED = auto()
    ARTEFACTS = auto()
    PATCHED = auto()

    @classmethod
    def valid(cls):
        return cls.VALID,

    @classmethod
    def invalid(cls):
        return cls.PIXELATED, cls.ARTEFACTS, cls.PATCHED


class SatelliteFilterClass(FilterClass):
    # Can still use elevation map:
    VALID = 0
    MIXED = auto()  # Mixed satellite/lighting images
    CLOUDS = auto()  # Visibility obscured (usability depends on heightmap)
    ONLY_WATER = auto()  # Tile only contains water

    # Can't use elevation map (not natural):
    ROADS = auto()  # Tile contains roads (note: small paths allowed)
    BUILDINGS = auto()  # Tile contains a significant number of buildings
    FARMLAND = auto()  # Altered to be flat

    @classmethod
    def valid(cls):  # Can use elevation map
        return cls.VALID, cls.MIXED, cls.CLOUDS, cls.ONLY_WATER

    @classmethod
    def invalid(cls):
        return cls.ROADS, cls.BUILDINGS, cls.FARMLAND

    @classmethod
    def invalid_satellite(cls):  # Can't use satellite image (clouds or mixed)
        return cls.CLOUDS, cls.MIXED


TYPE_MAPPINGS = {
    # name: (class, num_channels, ext)
    'derivative': (DerivativeFilterClass, 2, 'png'),
    'satellite': (SatelliteFilterClass, 3, 'jpg')
}
