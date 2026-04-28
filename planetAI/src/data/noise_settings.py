import random
import numpy as np
from skimage.morphology import disk
from skimage.filters import rank
import cv2
from typing import Type, TypeVar, Callable

T = TypeVar("T")


class NoiseFilter:
    @staticmethod
    def filter_noise(noise_map: np.ndarray):
        raise NotImplementedError("Subclasses must implement this method")


class SimpleFilter(NoiseFilter):
    @staticmethod
    def filter_noise(noise_map: np.ndarray):
        return (noise_map + 1) / 2


class RidgeFilter(NoiseFilter):
    @staticmethod
    def filter_noise(noise_map: np.ndarray):
        return 1 - np.abs(noise_map)


filter_types = [SimpleFilter, RidgeFilter]


class DropDownOption:
    label: str
    value: int | float

    def __init__(self, label: str, value: int | float):
        self.label = label
        self.value = value


class DropDownConfig:
    label: str
    options: list[DropDownOption]
    setter: Callable[[int | float], None]
    type_: Type[int | float]

    def __init__(
        self,
        label: str = "",
        options: list[DropDownOption] = [],
        setter: Callable[[int | float], None] = None,
        type_: Type[int | float] = int,
    ):
        self.label = label
        self.options = options
        self.setter = setter
        self.type_ = type_


class SliderConfig:
    # tuple[str, float | int, callable, float | int, float | int, type, float | int]
    name: str
    value: float | int
    setter: callable
    min_val: float | int
    max_val: float | int
    type_: type
    step: float | int

    def __init__(
        self,
        name: str,
        value: float | int,
        setter: callable,
        min_val: float | int,
        max_val: float | int,
        type_: type,
        step: float | int,
    ):
        self.name = name
        self.value = value
        self.setter = setter
        self.min_val = min_val
        self.max_val = max_val
        self.type_ = type_
        self.step = step

    def get_tuple(self):
        return (
            self.name,
            self.value,
            self.setter,
            self.min_val,
            self.max_val,
            self.type_,
            self.step,
        )


class SettingsBase:
    def __init__(self):
        self.display_name = "Settings"
        ct = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.display_colour = f"#{ct[0]:02x}{ct[1]:02x}{ct[2]:02x}"
        self.simple = True

    def get_advanced_controls(self) -> list[DropDownConfig | SliderConfig]:
        raise NotImplementedError(
            "Subclasses must implement this method"
            "to return a list of controls for sliders and dropdowns"
        )

    def get_simple_controls(self) -> list[DropDownConfig | SliderConfig]:
        raise NotImplementedError(
            "Subclasses must implement this method"
            "to return a list of controls for sliders and dropdowns"
        )

    def get_controls(self) -> list[DropDownConfig | SliderConfig]:
        raise NotImplementedError(
            "Subclasses must implement this method"
            "to return a list of controls for sliders and dropdowns"
        )

    def apply_filter(self, sketch: np.ndarray):
        raise NotImplementedError(
            "Subclasses must implement this method" "to apply the filter to the sketch"
        )

    def set_display_name(self, display_name: str):
        self.display_name = display_name

    def set_display_colour(self, display_colour: str):
        self.display_colour = display_colour

    def get_display_name_tuple(self):
        # (name, value, setter, type)
        return ("Display Name", self.display_name, self.set_display_name, str)

    def get_display_colour_tuple(self):
        # (name, value, setter, type)
        return ("Display Colour", self.display_colour, self.set_display_colour, str)


class NoiseSettings(SettingsBase):
    frequency: float
    seed: int
    min_value: float
    persistence: float
    amplitude: float
    roughness: float
    octaves: int
    noise_filter: NoiseFilter
    mask: np.ndarray | int
    simple: bool
    negate: int
    power: float

    def __init__(
        self,
        frequency: float = 0.1,
        seed: int = 1,
        min_value: float = 0.0,
        persistence: float = 0.7,
        amplitude: float = 0.5,
        roughness: float = 2.5,
        octaves: int = 5,
        erode_iters: int = 3,
        noise_filter: NoiseFilter = SimpleFilter,
        mask: np.ndarray | int = 0,
        negate: int = 0,
        exclusive: int = 0,
        additive: int = 0,
        multiplier: float = 1.0,
        simple: bool = True,
        power: float = 1.0,
        max_value: float = 1.0,
        use_max: bool = False,
    ):

        super().__init__()
        self.frequency = frequency
        self.seed = seed
        self.min_value = min_value
        self.max_value = max_value
        self.use_max = use_max
        self.persistence = persistence
        self.amplitude = amplitude
        self.roughness = roughness
        self.octaves = octaves
        self.erode_iters = erode_iters
        self.noise_filter = noise_filter
        self.display_name = "Noise Filter"
        self.mask = mask
        self.negate = negate
        self.exclusive = exclusive
        self.additive = additive
        self.multiplier = multiplier
        self.simple = simple
        self.power = power

    def set_frequency(self, frequency: float):
        self.frequency = frequency

    def set_seed(self, seed: int):
        self.seed = seed

    def set_min_value(self, min_value: float):
        self.min_value = min_value

    def set_max_value(self, max_value: float):
        self.max_value = max_value

    def set_use_max(self, use_max: bool):
        self.use_max = bool(use_max)

    def set_persistence(self, persistence: float):
        self.persistence = persistence

    def set_amplitude(self, amplitude: float):
        self.amplitude = amplitude

    def set_roughness(self, roughness: float):
        self.roughness = roughness

    def set_octaves(self, octaves: int):
        self.octaves = octaves

    def set_erode_iters(self, erode_iters: int):
        self.erode_iters = erode_iters

    def set_noise_filter(self, noise_filter: int):
        self.noise_filter = filter_types[noise_filter]

    def set_negate(self, negate: int):
        self.negate = negate

    def set_exclusive(self, exclusive: int):
        self.exclusive = exclusive

    def set_additive(self, additive: int):
        self.additive = additive

    def set_multiplier(self, multiplier: float):
        self.multiplier = multiplier

    def set_power(self, power: float):
        self.power = power

    def apply_filter(self, sketch: np.ndarray):
        return self.noise_filter.filter_noise(sketch)

    def set_mask(self, mask: np.ndarray):
        self.mask = mask

    def get_mask(self):
        return self.mask

    def get_advanced_controls(self) -> list[DropDownConfig | SliderConfig]:
        # (name, value, setter, min, max, type, step)
        return [
            SliderConfig(
                "Frequency",
                self.frequency,
                self.set_frequency,
                0.001,
                0.1,
                float,
                0.001,
            ),
            SliderConfig("Seed", self.seed, self.set_seed, 1, 100, int, 1),
            SliderConfig(
                "Min Value",
                self.min_value,
                self.set_min_value,
                -1.0,
                2.0,
                float,
                0.01,
            ),
            SliderConfig(
                "Max Value",
                self.max_value,
                self.set_max_value,
                0,
                2.0,
                float,
                0.01,
            ),
            SliderConfig(
                "Persistence",
                self.persistence,
                self.set_persistence,
                0.1,
                1.0,
                float,
                0.05,
            ),
            SliderConfig(
                "Amplitude", self.amplitude, self.set_amplitude, 0.1, 2.0, float, 0.05
            ),
            SliderConfig(
                "Roughness", self.roughness, self.set_roughness, 0.1, 5, float, 0.05
            ),
            SliderConfig("Octaves", self.octaves, self.set_octaves, 3, 6, int, 1),
            SliderConfig("Steepness", self.power, self.set_power, 0.5, 5.0, float, 0.1),
            # ("Erode Iters", self.erode_iters, self.set_erode_iters, 2, 5, int, 1),
            DropDownConfig(
                "Noise Filter",
                [
                    DropDownOption(filter_type.__name__, i)
                    for i, filter_type in enumerate(filter_types)
                ],
                self.set_noise_filter,
                type_=int,
            ),
            DropDownConfig(
                "Negate",
                [DropDownOption("No", 0), DropDownOption("Yes", 1)],
                self.set_negate,
                type_=int,
            ),
            # ("Exclusive", self.exclusive, self.set_exclusive, 0, 1, int, 1),
            # ("Additive", self.additive, self.set_additive, 0, 1, int, 1)
            SliderConfig(
                "Multiplier",
                self.multiplier,
                self.set_multiplier,
                0.1,
                10.0,
                float,
                0.05,
            ),
            DropDownConfig(
                "Use Max Value",
                [DropDownOption("No", 0), DropDownOption("Yes", 1)],
                self.set_use_max,
                type_=int,
            ),
        ]

    def get_simple_controls(self) -> list[DropDownConfig | SliderConfig]:
        return [
            SliderConfig("Seed", self.seed, self.set_seed, 1, 100, int, 1),
            SliderConfig(
                "Min Value",
                self.min_value,
                self.set_min_value,
                -1.0,
                2.0,
                float,
                0.01,
            ),
            SliderConfig(
                "Max Value",
                self.max_value,
                self.set_max_value,
                0,
                2.0,
                float,
                0.01,
            ),
            # SliderConfig(
            #     "Multiplier",
            #     self.multiplier,
            #     self.set_multiplier,
            #     0.1,
            #     10.0,
            #     float,
            #     0.05,
            # ),
            # SliderConfig("Steepness", self.power, self.set_power, 0.5, 5.0, float, 0.1),
            DropDownConfig(
                "Noise Filter",
                [
                    DropDownOption(filter_type.__name__, i)
                    for i, filter_type in enumerate(filter_types)
                ],
                self.set_noise_filter,
                type_=int,
            ),
            DropDownConfig(
                "Feature size",
                [
                    DropDownOption("Tiny", 0.128),
                    DropDownOption("Small", 0.064),
                    DropDownOption("Medium", 0.032),
                    DropDownOption("Large", 0.016),
                    DropDownOption("Huge", 0.008),
                ],
                self.set_frequency,
                type_=float,
            ),
            # DropDownConfig(
            #     "Negate",
            #     [DropDownOption("No", 0), DropDownOption("Yes", 1)],
            #     self.set_negate,
            #     type_=int,
            # ),
        ]

    def get_controls(self):
        if self.simple:
            return self.get_simple_controls()
        else:
            return self.get_advanced_controls()

    def randomise_values(self):
        for control in self.get_advanced_controls():
            if isinstance(control, SliderConfig):
                control.setter(
                    control.type_(random.uniform(control.min_val, control.max_val))
                )

    def __str__(self):
        return (
            f"Frequency: {self.frequency}, Seed: {self.seed}, Min Value: {self.min_value}, " +
            f"Persistence: {self.persistence}, Amplitude: {self.amplitude}, Roughness: {self.roughness}, " +
            f"Octaves: {self.octaves}, Erode Iters: {self.erode_iters}, Max Value: {self.max_value}"
        )

    def get_config_dict(self):
        config_dict = {}
        for control in self.get_advanced_controls():
            if isinstance(control, SliderConfig):
                config_dict[control.name.lower().replace(" ", "_")] = control.value
        return config_dict

    def from_config_dict(self, config_dict: dict):
        for control in self.get_advanced_controls():
            if isinstance(control, SliderConfig):
                control.setter(
                    control.type_(
                        config_dict.get(control.name.lower().replace(" ", "_"), 0)
                    )
                )


class MeanFilterSettings(SettingsBase):
    radius: int
    mask: np.ndarray | int

    def __init__(
        self, radius: int = 2, mask: np.ndarray | int = 0, simple: bool = True
    ):
        super().__init__()
        self.display_name = "Mean Filter"
        self.radius = radius
        self.mask = mask
        self.simple = simple

    def set_radius(self, radius: int):
        self.radius = radius

    def get_simple_controls(self):
        return [SliderConfig("Radius", self.radius, self.set_radius, 0, 15, int, 1)]

    def get_advanced_controls(self):
        return self.get_simple_controls()

    def get_controls(self):
        return self.get_simple_controls()

    def apply_filter(self, sketch: np.ndarray):
        if self.radius == 0:
            return sketch
        return rank.mean((sketch * 255).astype(np.uint8), disk(self.radius)) / 255

    def set_mask(self, mask: np.ndarray):
        self.mask = mask

    def get_mask(self):
        return self.mask


class EdgeDistanceSettings(SettingsBase):
    def __init__(
        self, mask: np.ndarray | int = 0, simple: bool = True
    ):
        super().__init__()
        self.display_name = "Edge Distance Filter"
        self.mask = mask
        self.simple = simple
        self.min = 0.0
        self.max = 1.0
        self.offset = 0.0
        self.power = 1.0
        self.controls = [
            SliderConfig(
                "Min",
                self.min,
                lambda x: setattr(self, "min", x),
                0.0,
                1.0,
                float,
                0.05,
            ),
            SliderConfig(
                "Max",
                self.max,
                lambda x: setattr(self, "max", x),
                0.05,
                1.0,
                float,
                0.05,
            ),
            SliderConfig(
                "Power",
                self.power,
                lambda x: setattr(self, "power", x),
                0.0,
                3.0,
                float,
                0.25,
            ),
        ]

    def get_advanced_controls(self) -> list[DropDownConfig | SliderConfig]:
        return self.controls

    def get_simple_controls(self) -> list[DropDownConfig | SliderConfig]:
        return self.controls

    def get_controls(self) -> list[DropDownConfig | SliderConfig]:
        return self.controls

    def apply_filter(self, mask: np.ndarray):
        # This filter expects a boolean mask
        if mask.dtype != bool:
            return np.zeros(mask.shape, dtype=bool)
        if mask.all():
            return mask.astype(np.float32)
        int_mask = (mask > 0).astype(np.uint8)
        edge_mask = cv2.dilate(int_mask, np.ones((3, 3), dtype=np.uint8)) - int_mask
        edge_coords = np.array(np.where(edge_mask > 0)).transpose((1, 0))
        mask_coords = np.array(np.where(mask > 0)).transpose((1, 0))
        edge_coords = edge_coords.reshape(edge_coords.shape[0], 1, 2)
        mask_coords = mask_coords.reshape(1, mask_coords.shape[0], 2)
        distances: np.ndarray = np.sum((edge_coords - mask_coords)**2, axis=2)
        new_mask = np.zeros(mask.shape, dtype=np.float32)
        if len(distances) == 0:
            return new_mask
        min_distances: np.ndarray = distances.min(axis=0)
        min_distances = min_distances / min_distances.max()
        ys = mask_coords[0, :, 0]
        xs = mask_coords[0, :, 1]
        new_mask[ys, xs] = np.sqrt(min_distances)

        new_mask **= self.power
        new_mask += self.offset
        new_mask = np.maximum(new_mask - self.min, 0)
        new_mask = np.minimum(new_mask, self.max)
        new_mask[int_mask == 0] = 0.0
        if new_mask.max() > new_mask.min():
            new_mask = (new_mask - new_mask.min()) / (new_mask.max() - new_mask.min())

        return new_mask
