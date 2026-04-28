from .utils import PlanetConfig, timing
from .sketch_gen import temperature_paint
from .landcover_utils import LandcoverClasses, landcover_index_map, translate_land
from .sphere_mapping import normal_to_quad

import numpy as np
from PIL import Image
import os
from dataclasses import replace


class ModalSketch:

    @timing
    def __init__(
        self,
        planet_cfg: PlanetConfig,
        save_sketch: bool = False,
        save_counts: bool = False,
        log: bool = False,
        force: bool = False,
        temp_kernel: int = 7,
        temp: np.ndarray | None = None,
        land: np.ndarray | None = None,
        sat: np.ndarray | None = None,
        get_sat_loss: bool = False,
        min_count: int = 0,
        use_quad: bool = True,
    ):
        planet_cfg = replace(planet_cfg, size=5, image_mode="sat")
        use_summer = planet_cfg.use_summer
        self.mapping_file = (
            f"mapping{planet_cfg.size}_{planet_cfg.temp_classes}_"
            f"{planet_cfg.landcover_classes}_{use_summer}.txt"
        )
        if use_quad:
            self.mapping_file = self.mapping_file.replace(
                ".txt", "_quad.txt"
            )
        self.planet_cfg = planet_cfg
        combined_classes = (planet_cfg.landcover_classes + 1) * (
            planet_cfg.temp_classes + 1
        )
        self.max_class = combined_classes
        self.temp_kernel = temp_kernel
        self.load_mars_mapping(planet_cfg.data_dir)
        if self.load_mapping(planet_cfg.data_dir) and not force:
            return
        self.counts = np.zeros((combined_classes, 256**3), dtype=np.int32)
        if not force:
            if self.load_counts(planet_cfg.data_dir):
                return
        counts = self.counts

        land_step = 255 // planet_cfg.landcover_classes
        temp_step = 255 // planet_cfg.temp_classes

        H = 256 * 2**planet_cfg.size
        W = 512 * 2**planet_cfg.size

        def quad_func(x: np.ndarray, discrete: bool = False) -> np.ndarray:
            if use_quad:
                return normal_to_quad(x, discrete=discrete)
            else:
                return x

        from planetAI.src.data.atlas_loader import AtlasLoader, QuadAtlasLoader

        self.atlas_loader = AtlasLoader(planet_cfg)
        self.quad_atlas_loader = QuadAtlasLoader(planet_cfg)

        if temp is None:
            temp = self.quad_atlas_loader.quad_temp if use_quad else self.atlas_loader.temp
            if use_summer:
                temp2 = self.quad_atlas_loader.quad_temp_summer if use_quad else self.atlas_loader.temp_summer
                temp = np.concatenate((temp, temp2))
        # Min/max normalise the temperature
        # temp = (temp - temp.min()) / (temp.max() - temp.min())
        # temp = (temp * 255).astype(np.uint8)

        if land is None:
            land = self.quad_atlas_loader.quad_land if use_quad else self.atlas_loader.land
            if use_summer:
                land = np.concatenate(
                    (land, land)
                )  # TODO Check if landcover is the same for summer
            land = translate_land(land)
        downland_sketch = quad_func(self.atlas_loader.downland_sketch)
        missing = (temp == 0) & (temp < 255)
        temp[missing] += 1
        if sat is None:
            sat = self.quad_atlas_loader.quad_sat if use_quad else self.atlas_loader.sat
            if use_summer:
                sat2 = self.quad_atlas_loader.quad_sat_summer if use_quad else self.atlas_loader.sat_summer
                sat = np.concatenate((sat, sat2))

        temp_sketch = temperature_paint(temp, planet_cfg.downscale_cfg).astype(
            np.uint16
        )
        temp_sketch = ((temp_sketch + 1) // temp_step).astype(np.uint8)
        downtemp_sketch = np.concatenate(
            (quad_func(self.atlas_loader.downtemp_sketch), quad_func(self.atlas_loader.downtemp_sketch_summer))
        )
        downtemp_sketch = ((downtemp_sketch + 1) // temp_step).astype(np.uint8)
        land_sketch = land // land_step
        downland_sketch = np.concatenate(
            (quad_func(self.atlas_loader.downland_sketch), quad_func(self.atlas_loader.downland_sketch))
        ) // land_step

        # TODO Maybe do this better
        # temp_sketch[land_sketch == 0] = 0
        # land_sketch[temp_sketch == 0] = 0

        sat_hex = np.zeros_like(sat[:, :, 0]).astype(np.int32)
        sat_hex = sat[:, :, 0] * 256**2 + sat[:, :, 1] * 256 + sat[:, :, 2]

        ocean_mask = sat_hex == (2 * 256**2 + 5 * 256 + 20)
        land_sketch[(ocean_mask & (land_sketch != 0))] = LandcoverClasses.WATER.gray_colour // land_step

        # rgb_mapping_values = self.mapping.values()
        # hex_mapping_values = [r*256**2 + g*256 + b for r, g, b in rgb_mapping_values]
        # down_sat_hex = modal_resize(sat_hex, 2**planet_cfg.size, included_values=np.array(hex_mapping_values))
        # down_sat = np.zeros((h, w, 3), dtype=np.uint8)
        # down_sat[:, :, 2] = down_sat_hex % 256
        # down_sat_hex //= 256
        # down_sat[:, :, 1] = down_sat_hex % 256
        # down_sat_hex //= 256
        # down_sat[:, :, 0] = down_sat_hex
        # Image.fromarray(down_sat).save(
        #   os.path.join(planet_cfg.data_dir, f'test/down_sat_{W}x{H}_{planet_cfg.temp_classes}.png')
        # )
        combined_sketch = (
            land_sketch * (planet_cfg.temp_classes + 1) + temp_sketch
        ).astype(np.int32)
        downcombined_sketch = (
            downland_sketch * (planet_cfg.temp_classes + 1) + downtemp_sketch
        ).astype(np.int32)
        # mapping = np.dstack((combined_sketch, sat_hex))

        combined_values = np.unique(combined_sketch)
        downcombined_values = np.unique(downcombined_sketch)

        for c in range(combined_classes):
            if not (combined_values == c).any():
                continue
            colours, num = np.unique(sat_hex[combined_sketch == c], return_counts=True)
            num[num < min_count] = 0
            if (downcombined_values == c).any():
                counts[c, colours] += num

        combined_rgb_sketch = np.zeros_like(sat)

        min_temp = 257  # K
        max_temp = 307  # K

        # Convert to Celsius
        min_temp -= 273
        max_temp -= 273

        tc = planet_cfg.temp_classes

        temp_list = ["Water"]
        for t in range(tc):
            temp_list.append(
                f"{min_temp + t*(max_temp-min_temp)/tc:.2f} C to {min_temp + (t+1)*(max_temp-min_temp)/tc:.2f} C"
            )

        os.makedirs(os.path.join(planet_cfg.data_dir, "test"), exist_ok=True)

        used = {}

        mapping: dict[int, tuple[int, int, int]] = {}

        # For debugging
        Image.fromarray(land_sketch * land_step).save(
            os.path.join(
                planet_cfg.data_dir,
                f"test/land_sketch_{W}x{H}_{planet_cfg.temp_classes}.png",
            )
        )
        Image.fromarray(temp_sketch * temp_step).save(
            os.path.join(
                planet_cfg.data_dir,
                f"test/temp_sketch_{W}x{H}_{planet_cfg.temp_classes}.png",
            )
        )
        Image.fromarray(sat).save(
            os.path.join(
                planet_cfg.data_dir, f"test/sat_{W}x{H}_{planet_cfg.temp_classes}.png"
            )
        )
        Image.fromarray(
            (combined_sketch * 255 // combined_classes).astype(np.uint8)
        ).save(
            os.path.join(
                planet_cfg.data_dir,
                f"test/combined_sketch_{W}x{H}_{planet_cfg.temp_classes}.png",
            )
        )

        for c in range(combined_classes):
            modal = np.argmax(counts[c])
            count = counts[c, modal]
            r = modal // 256**2
            g = (modal % 256**2) // 256
            b = modal % 256
            default = f" count: {count}"
            lc = c // (planet_cfg.temp_classes + 1)
            tc = c % (planet_cfg.temp_classes + 1)
            full_lc = landcover_index_map(lc)
            if lc == 0:
                r, g, b = 2, 5, 20  # Always set water landcover to be dark blue
            elif tc == 0:
                # Set water temperature (missing data) to be the default class colour
                r, g, b = full_lc.display_colour
                # default = " (default)"
            if log:
                print(
                    f"Class {c:02}\t[{full_lc.display_name}]\t[{temp_list[tc]}]\t({r}, {g}, {b}){default}",
                    end="",
                )
                if used.get((r, g, b)) is None:
                    print()
                    used[(r, g, b)] = c
                else:
                    print(" same as", used[(r, g, b)])
            if save_sketch:
                combined_rgb_sketch[combined_sketch == c] = [r, g, b]
            mapping[int(c)] = (int(r), int(g), int(b))
        if get_sat_loss:
            # Get mse between sat and combined_rgb_sketch
            sat_loss = np.mean((combined_rgb_sketch - sat) ** 2)
            print(f"Satellite loss: {sat_loss}")
        if save_sketch:
            Image.fromarray(combined_rgb_sketch).save(
                os.path.join(
                    planet_cfg.data_dir,
                    f"test/combined_rgb_sketch_{W}x{H}_{planet_cfg.temp_classes}.png",
                )
            )
        self.mapping = mapping
        self.fill_zeros()
        self.save_mapping(planet_cfg.data_dir)
        colour_matrix = self.create_colour_matrix()
        if save_sketch:
            Image.fromarray(colour_matrix).save(
                os.path.join(
                    planet_cfg.data_dir,
                    f"test/colour_matrix_{W}x{H}_{planet_cfg.temp_classes}.png",
                )
            )
            down_rgb_sketch = self.get_sketch(
                np.concatenate([self.atlas_loader.downland_sketch, self.atlas_loader.downland_sketch]),
                np.concatenate([self.atlas_loader.downtemp_sketch, self.atlas_loader.downtemp_sketch_summer]),
            )
            Image.fromarray(down_rgb_sketch).save(
                os.path.join(
                    planet_cfg.data_dir,
                    f"test/down_rgb_sketch_{W}x{H}_{planet_cfg.temp_classes}.png",
                )
            )

        self.counts = counts
        if save_counts:
            self.save_counts(planet_cfg.data_dir)

    def save_mapping(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, self.mapping_file), "w") as f:
            for c in range(self.max_class):
                r, g, b = self.mapping[c]
                f.write(f"{c} {r} {g} {b}\n")

    def load_mapping(self, path: str) -> bool:
        if not os.path.exists(os.path.join(path, self.mapping_file)):
            return False
        with open(os.path.join(path, self.mapping_file), "r") as f:
            self.mapping = {}
            for line in f:
                c, r, g, b = line.split()
                self.mapping[int(c)] = (int(r), int(g), int(b))
        return True

    def load_mars_mapping(self, path: str) -> bool:
        if not os.path.exists(os.path.join(path, "mars_mapping.txt")):
            return False
        with open(os.path.join(path, "mars_mapping.txt"), "r") as f:
            self.mars_mapping = {}
            for line in f:
                c, r, g, b = line.split()
                self.mars_mapping[int(c)] = (int(r), int(g), int(b))
        return True

    def get_tuple_mapping(
        self, use_sketch_colours: bool = False
    ) -> dict[tuple[int, int], tuple[int, int, int]]:
        """
        Get the modal sketch mapping with the keys as tuples of (landcover, temperature)
        """
        mapping = self.mapping
        temp_classes = self.planet_cfg.temp_classes
        land_classes = self.planet_cfg.landcover_classes
        temp_step = 255 // temp_classes
        land_step = 255 // land_classes
        tuple_mapping = {}
        for c in range(self.max_class):
            temp_class = c % (temp_classes + 1)
            land_class = c // (temp_classes + 1)
            if use_sketch_colours:
                temp_class = max(temp_class * temp_step - 1, 0)
                land_class = land_class * land_step
            tuple_mapping[(land_class, temp_class)] = mapping[c]
        return tuple_mapping

    def save_counts(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "counts.txt"), "w") as f:
            for c in range(self.max_class):
                for i in range(256**3):
                    if self.counts[c, i] > 0:
                        f.write(f"{c} {i} {self.counts[c, i]}\n")

    def load_counts(self, path: str) -> bool:
        if not os.path.exists(os.path.join(path, "counts.txt")):
            return False
        with open(os.path.join(path, "counts.txt"), "r") as f:
            self.counts = np.zeros((self.max_class, 256**3), dtype=np.int32)
            for line in f:
                c, i, count = line.split()
                self.counts[int(c), int(i)] = int(count)
        return True

    def fill_zeros(self):
        """
        Fill the zeros in the mapping by using the nearest temperature value for each class
        """
        mapping = self.mapping
        temp_mapping = {}
        temp_classes = self.planet_cfg.temp_classes
        for c in range(self.max_class):
            temp_class = c % (temp_classes + 1)
            land_class = c // (temp_classes + 1)
            temp_mapping[c] = mapping[c]
            if mapping.get(c) == (0, 0, 0):
                # Find the nearest temperature class with a colour
                for i in range(1, temp_classes + 1):
                    if temp_class + i <= temp_classes and mapping.get(
                        land_class * (temp_classes + 1) + temp_class + i
                    ) != (0, 0, 0):
                        temp_mapping[c] = mapping[
                            land_class * (temp_classes + 1) + temp_class + i
                        ]
                        break
                    if temp_class - i >= 1 and mapping.get(
                        land_class * (temp_classes + 1) + temp_class - i
                    ) != (0, 0, 0):
                        temp_mapping[c] = mapping[
                            land_class * (temp_classes + 1) + temp_class - i
                        ]
                        break
        self.mapping = temp_mapping

    def create_colour_matrix(self, tile_size: int = 100) -> np.ndarray:
        mapping = self.mapping
        temp_colours = self.planet_cfg.temp_classes
        land_colours = self.planet_cfg.landcover_classes
        H = tile_size * (land_colours + 1)
        W = tile_size * (temp_colours + 1)
        colour_matrix = np.zeros((H, W, 3), dtype=np.uint8)
        for c in range(self.max_class):
            r, g, b = mapping[c]
            y = c // (temp_colours + 1) * tile_size
            x = c % (temp_colours + 1) * tile_size
            colour_matrix[y: y + tile_size, x: x + tile_size] = [r, g, b]

        return colour_matrix

    def get_sketch(
        self,
        landcover_sketch: np.ndarray,
        temperature_sketch: np.ndarray,
        is_mars: bool = False,
        mars_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Get the modal satellite image colour of a combination of landcover and temperature sketch
        """
        if is_mars:
            mars_mask = np.ones_like(landcover_sketch)
        elif mars_mask is None:
            mars_mask = np.zeros_like(landcover_sketch)
        has_mars = np.any(mars_mask)
        has_earth = np.any(~mars_mask)
        land_step = 255 // self.planet_cfg.landcover_classes
        temp_step = 255 // self.planet_cfg.temp_classes
        landcover_sketch = translate_land(landcover_sketch) // land_step
        temperature_sketch = (temperature_sketch) // (temp_step - 1)
        if not is_mars:
            temperature_sketch[landcover_sketch == 0] = 0
        temperature_sketch[(mars_mask > 0) & (temperature_sketch == 0)] = 3
        landcover_sketch[mars_mask > 0] = 0
        vals, num = np.unique(temperature_sketch, return_counts=True)
        nonzero = vals != 0
        vals = vals[nonzero]
        num = num[nonzero]
        if len(vals) == 0:
            mode_temp = 0
        else:
            mode_temp = vals[np.argmax(num)]
        # Create a circle kernel to dilate and erode the temperature sketch
        # kernel = np.zeros((self.temp_kernel, self.temp_kernel), dtype=np.uint8)
        # r = self.temp_kernel // 2
        # kernel = circle(kernel, (r, r), r, 1, -1)
        # TODO Figure out how to do this well
        # temperature_sketch = dilate(temperature_sketch, kernel, iterations=1)
        # temperature_sketch = erode(temperature_sketch, kernel, iterations=1)
        temperature_sketch[(landcover_sketch != 0) & (temperature_sketch == 0)] = (
            mode_temp
        )
        combined_sketch = (
            landcover_sketch * (self.planet_cfg.temp_classes + 1) + temperature_sketch
        ).astype(np.int32)
        H, W = combined_sketch.shape
        combined_rgb_sketch = np.zeros((H, W, 3), dtype=np.uint8)
        if has_mars:
            for c in range(5):
                r, g, b = self.mars_mapping[c + 1]
                combined_rgb_sketch[(mars_mask > 0) & (combined_sketch == c + 1)] = [
                    r,
                    g,
                    b,
                ]
        if has_earth:
            for c in range(self.max_class):
                r, g, b = self.mapping[c]
                combined_rgb_sketch[(mars_mask == 0) & (combined_sketch == c)] = [
                    r,
                    g,
                    b,
                ]

        return combined_rgb_sketch.astype(np.uint8)

    def get_colour(
        self, landcover: int, temperature: int, dem: int | None = None
    ) -> tuple[int, int, int]:
        temperature = (temperature + 1) // (255 // self.planet_cfg.temp_classes)
        if landcover == 255:
            normal_colour = self.mars_mapping[max(temperature, 1)]
        else:
            landcover = landcover // (255 // self.planet_cfg.landcover_classes)
            normal_colour = self.mapping[
                landcover * (self.planet_cfg.temp_classes + 1) + temperature
            ]
        if dem is None:
            return normal_colour
        else:
            # Do some brightness changes depending on dem value
            # Colour should be slightly duller for lower dem values 25% duller for 0 dem
            # Colour should be slightly brighter for higher dem values 25% brighter for 255 dem
            normal_colour = np.array(normal_colour)
            dem_scale = (dem / 512) - 0.25
            normal_colour = normal_colour + normal_colour * dem_scale
            normal_colour = np.clip(normal_colour, 0, 255).astype(np.uint8)
            return tuple(normal_colour)


if __name__ == "__main__":
    # We want to find the modal satellite image colour of a combination of landcover and temperature sketch
    planet_cfg = PlanetConfig(downscale_offset=3)
    modal_sketch = ModalSketch(
        planet_cfg,
        save_sketch=True,
        log=True,
        force=True,
        get_sat_loss=True,
        save_counts=False,
    )
