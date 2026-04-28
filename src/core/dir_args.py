
# Directory arguments dataclasses used when parsing arguments

from dataclasses import dataclass, field

from .constants import MAX_ELEVATION_LEVEL


@dataclass
class ElevationArguments:
    elevation_dir: str = field(
        default='data/elevation',
        metadata={
            'help': 'Directory to store raw elevation data'
        },
    )
    valid_dir: str = field(
        default='valid',
        metadata={
            'help': 'Directory inside elevation_dir to store valid elevation data'
        },
    )
    to_delete: str = field(
        default='to_delete',
        metadata={
            'help': 'Directory inside elevation_dir to move invalid elevation data to'
        },
    )
    elevation_zoom: int = field(
        default=MAX_ELEVATION_LEVEL,
        metadata={
            'help': 'Scale of elevation data to download'
        },
    )


@dataclass
class SatelliteArguments:
    satellite_dir: str = field(
        default='data/satellite',
        metadata={
            'help': 'Directory to store raw satellite data'
        },
    )
    cache_dir: str = field(
        default='cache',
        metadata={
            'help': 'Directory inside satellite_dir to store images'
        },
    )


@dataclass
class DerivativeArguments:
    derivative_dir: str = field(
        default='data/derivative',
        metadata={
            'help': 'Directory to store derivative data'
        }
    )

    raw_dir: str = field(
        default='raw',
        metadata={
            'help': 'Directory inside derivative_dir to store raw derivative data'
        }
    )


@dataclass
class CoverageArguments:
    coverage_dir: str = field(
        default='data/coverage',
        metadata={
            'help': 'Directory to store raw coverage data'
        },
    )
