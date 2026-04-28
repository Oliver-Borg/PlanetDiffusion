import numpy as np
import cv2
from typing import Optional, Callable
import os

from planetAI.src.data.sphere_mapping import QuadSphere, SampleMode
from planetAI.src.data.utils import PlanetConfig, get_data_image, modal_resize, timing
from planetAI.src.data.sketch_gen import (
    landcover_paint,
    dilate_paint,
    temperature_paint,
    get_buckets,
)
from planetAI.src.data.modal_sketch import ModalSketch
from planetAI.src.data.river_processing import apply_filters, get_stacked_rivers
from planetAI.src.data.river_modal_sketch import RiverModalSketch
from .line_dataset import line_resizer


class AtlasLoader:
    def __init__(self, planet_cfg: PlanetConfig):
        self.planet_cfg = planet_cfg
        self._dem: Optional[np.ndarray] = None
        self._float_dem: Optional[np.ndarray] = None
        self._down_dem: Optional[np.ndarray] = None
        self._sat: Optional[np.ndarray] = None
        self._rivers: Optional[np.ndarray] = None
        self._landmask: Optional[np.ndarray] = None
        self._downsketch: Optional[np.ndarray] = None
        self._land: Optional[np.ndarray] = None
        self._downland: Optional[np.ndarray] = None
        self._downland_sketch: Optional[np.ndarray] = None
        self._temp: Optional[np.ndarray] = None
        self._temp_sketch: Optional[np.ndarray] = None
        self._downtemp: Optional[np.ndarray] = None
        self._downtemp_summer: Optional[np.ndarray] = None
        self._downtemp_sketch: Optional[np.ndarray] = None
        self._downtemp_sketch_summer: Optional[np.ndarray] = None
        self._modal_sketch: Optional[np.ndarray] = None
        self._downmodal_sketch: Optional[np.ndarray] = None
        self._full_river_sketch: Optional[np.ndarray] = None
        self._modal_river_sketch: Optional[np.ndarray] = None
        self._bathy: Optional[np.ndarray] = None
        self._mars_dem: Optional[np.ndarray] = None
        self._mars_sat: Optional[np.ndarray] = None
        self._mars_temp: Optional[np.ndarray] = None
        self._down_mars_dem: Optional[np.ndarray] = None
        self._down_mars_sat: Optional[np.ndarray] = None
        self._down_mars_temp: Optional[np.ndarray] = None
        self._sat_summer: Optional[np.ndarray] = None
        self._temp_summer: Optional[np.ndarray] = None
        self._precipitation: Optional[np.ndarray] = None
        self._precipitation_summer: Optional[np.ndarray] = None
        self._quad_boundary_sketch: Optional[np.ndarray] = None
        self._river_upa: Optional[np.ndarray] = None

        self.H, self.W = self.planet_cfg.H, self.planet_cfg.W
        self.h, self.w = self.planet_cfg.h, self.planet_cfg.w

        self.mars_H = int(4362 // (2 ** (5 - self.planet_cfg.size)))
        self.mars_W = 2 * self.mars_H

        self.mars_h = self.mars_H // self.planet_cfg.delta
        self.mars_w = 2 * self.mars_h

    @property
    def dem(self) -> np.ndarray:
        if self._dem is None:
            self._dem = np.ceil(self.float_dem.copy()).clip(0, 255).astype(np.uint8)
        return self._dem

    @property
    def float_dem(self) -> np.ndarray:
        if self._float_dem is None:
            self._float_dem = get_data_image(
                self.planet_cfg.data_dir,
                (self.H, self.W),
                "World_DEM_WxH.png",
                default_shape=(8192, 16384),
                interpolation=cv2.INTER_LANCZOS4,
                custom_resizer=lambda x, shape: cv2.resize(
                    x.astype(np.float32), tuple(reversed(shape)), interpolation=cv2.INTER_LANCZOS4
                )
            )
            # This causes some artifacts so I am going to remove it
            # self._float_dem[self.landmask > 0] += 1
            eroded_landmask = cv2.erode(self.landmask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
            self._float_dem[(self._float_dem < 0.1) & (eroded_landmask > 0)] = 0.1
        return self._float_dem

    @property
    def down_dem(self) -> np.ndarray:
        if self._down_dem is None:
            self._down_dem = get_data_image(
                self.planet_cfg.data_dir,
                (self.h, self.w),
                "World_DEM_WxH.png",
                default_shape=(8192, 16384),
                interpolation=cv2.INTER_AREA,
            )
        return self._down_dem

    @property
    def sat(self) -> np.ndarray:
        if self._sat is None:
            self._sat = get_data_image(
                self.planet_cfg.data_dir,
                (self.H, self.W),
                "world.satellite.WxH.png",
                default_shape=(8192, 16384),
                interpolation=cv2.INTER_LANCZOS4,
            )
        return self._sat

    @property
    def rivers(self) -> np.ndarray:
        if self._rivers is None:
            self._rivers = get_stacked_rivers(self.planet_cfg.data_dir, self.H, self.W)
            self._rivers = apply_filters(
                self._rivers[:, :, 0],
                self._rivers[:, :, 1],
                self._rivers[:, :, 2],
            )
        return self._rivers

    @property
    def landmask(self) -> np.ndarray:
        if self._landmask is None:
            self._landmask = get_data_image(
                self.planet_cfg.data_dir,
                (self.H, self.W),
                "world.oceanmask.WxH.png",  # This is called oceanmask but the ocean is 0 and the land is 255
                default_shape=(8192, 16384),
                interpolation=cv2.INTER_NEAREST,
            )
        return self._landmask

    @property
    def downsketch(self) -> np.ndarray:
        if self._downsketch is None:
            buckets = (
                get_buckets(self.down_dem, self.planet_cfg)
                if self.planet_cfg.bucketing_mode == "uniform"
                else None
            )
            self._downsketch = dilate_paint(
                self.down_dem, self.planet_cfg.downscale_cfg, buckets=buckets
            )
        return self._downsketch

    @property
    def landcover_sketch(self) -> np.ndarray:
        if self._landcover_sketch is None:
            downland = get_data_image(
                self.planet_cfg.data_dir,
                (self.h, self.w),
                "World_LandCover_WxH.png",
                custom_resizer=lambda x, shape: modal_resize(
                    x, dx=x.shape[0] // shape[0]
                ),
                default_shape=(8192, 16384),
            )
            self._landcover_sketch = landcover_paint(
                downland, self.planet_cfg.downscale_cfg
            )
            self._landcover_sketch[self.downsketch == 0] = 0
        return self._landcover_sketch

    @property
    def land(self) -> np.ndarray:
        if self._land is None:
            self._land = get_data_image(
                self.planet_cfg.data_dir,
                (self.H, self.W),
                "World_LandCover_WxH.png",
                custom_resizer=lambda x, shape: modal_resize(
                    x, dx=x.shape[0] // shape[0]
                ),
                default_shape=(8192, 16384),
            )
        return self._land

    @property
    def downland(self) -> np.ndarray:
        if self._downland is None:
            self._downland = get_data_image(
                self.planet_cfg.data_dir,
                (self.h, self.w),
                "World_LandCover_WxH.png",
                custom_resizer=lambda x, shape: modal_resize(
                    x, dx=x.shape[0] // shape[0]
                ),
                default_shape=(8192, 16384),
            )
        return self._downland

    @property
    def temp(self) -> np.ndarray:
        if self._temp is None:
            self._temp = get_data_image(
                self.planet_cfg.data_dir,
                (self.H, self.W),
                "World_Temp_WxH.png",
                interpolation=cv2.INTER_AREA,
                default_shape=(8192, 16384),
            )
            self._temp[self._temp == 0] += 1
        return self._temp

    @property
    def downtemp(self) -> np.ndarray:
        if self._downtemp is None:
            self._downtemp = get_data_image(
                self.planet_cfg.data_dir,
                (self.h, self.w),
                "World_Temp_WxH.png",
                interpolation=cv2.INTER_AREA,
                default_shape=(8192, 16384),
            )
            self._downtemp[(self._downtemp < 255)] += 1
        return self._downtemp

    @property
    def downtemp_summer(self) -> np.ndarray:
        if self._downtemp_summer is None:
            self._downtemp_summer = get_data_image(
                self.planet_cfg.data_dir,
                (self.h, self.w),
                "World_Temp_WxH_summer.png",
                interpolation=cv2.INTER_AREA,
                default_shape=(8192, 16384),
            )
            self._downtemp_summer[(self._downtemp_summer < 255)] += 1
        return self._downtemp_summer

    @property
    def temp_sketch(self) -> np.ndarray:
        if self._temp_sketch is None:
            self._temp_sketch = temperature_paint(self.temp, self.planet_cfg)
            self._temp_sketch[self.land == 0] = 0
        return self._temp_sketch

    @property
    def modal_sketch(self) -> np.ndarray:
        if self._modal_sketch is None:
            self._modal_sketch = ModalSketch(
                self.planet_cfg, temp=self.temp, land=self.land, sat=self.sat
            ).get_sketch(
                self.landcover_sketch, self.temp_sketch
            )
        return self._modal_sketch

    @property
    def downmodal_sketch(self) -> np.ndarray:
        if self._downmodal_sketch is None:
            self._downmodal_sketch = ModalSketch(
                self.planet_cfg, temp=self.temp, land=self.land, sat=self.sat
            ).get_sketch(self.downland, self.downtemp_sketch)
        return self._downmodal_sketch

    @property
    def downland_sketch(self) -> np.ndarray:
        if self._downland_sketch is None:
            downland = self.downland.copy()
            downsketch = self.downsketch.copy()
            self._downland_sketch = landcover_paint(
                downland, self.planet_cfg.downscale_cfg
            )
            self._downland_sketch[downsketch == 0] = 0
        return self._downland_sketch

    @property
    def downtemp_sketch(self) -> np.ndarray:
        if self._downtemp_sketch is None:
            self._downtemp_sketch = temperature_paint(self.downtemp, self.planet_cfg)
        return self._downtemp_sketch

    @property
    def downtemp_sketch_summer(self) -> np.ndarray:
        if self._downtemp_sketch_summer is None:
            self._downtemp_sketch_summer = temperature_paint(self.downtemp_summer, self.planet_cfg)
        return self._downtemp_sketch_summer

    @property
    def full_river_sketch(self) -> np.ndarray:
        if self._full_river_sketch is None:
            self._full_river_sketch = get_data_image(
                self.planet_cfg.data_dir,
                (self.H, self.W),
                "World_Modal_Rivers_WxH.png",
                default_shape=(8192, 16384),
                interpolation=cv2.INTER_NEAREST,
                generator_func=RiverModalSketch(self.planet_cfg).get_sketch,
                generator_args=[
                    cv2.resize(self.downland_sketch, (self.W, self.H), interpolation=cv2.INTER_NEAREST),
                    cv2.resize(self.downtemp_sketch, (self.W, self.H), interpolation=cv2.INTER_NEAREST),
                    self.rivers
                ],
            )
        return self._full_river_sketch

    @property
    def modal_river_sketch(self) -> np.ndarray:
        if self._modal_river_sketch is None:
            self._modal_river_sketch = cv2.resize(
                self.downmodal_sketch,
                (self.W, self.H),
                interpolation=cv2.INTER_NEAREST,
            )
            self._modal_river_sketch[self.rivers > 0] = self.full_river_sketch[self.rivers > 0]
        return self._modal_river_sketch

    @property
    def bathy(self) -> np.ndarray:
        if self._bathy is None:
            self._bathy = get_data_image(
                self.planet_cfg.data_dir,
                (self.H, self.W),
                "gebco_bathy.WxH.jpg",
                default_shape=(10801, 21601),
                interpolation=cv2.INTER_AREA,
            )
        return self._bathy

    @property
    def mars_dem(self) -> np.ndarray:
        if self._mars_dem is None:
            self._mars_dem = get_data_image(
                self.planet_cfg.data_dir,
                (self.mars_H, self.mars_W),
                "Mars_DEM_WxH.png",
                default_shape=(4362, 8724),
                interpolation=cv2.INTER_AREA,
            )
        return self._mars_dem

    @property
    def mars_sat(self) -> np.ndarray:
        if self._mars_sat is None:
            self._mars_sat = get_data_image(
                self.planet_cfg.data_dir,
                (self.mars_H, self.mars_W),
                "Mars_Sat_WxH.png",
                default_shape=(4362, 8724),
                interpolation=cv2.INTER_LANCZOS4,
            )
        return self._mars_sat

    @property
    def mars_temp(self) -> np.ndarray:
        if self._mars_temp is None:
            self._mars_temp = get_data_image(
                self.planet_cfg.data_dir,
                (self.mars_H, self.mars_W),
                "Mars_Temp_WxH.png",
                default_shape=(512, 1024),
                interpolation=cv2.INTER_NEAREST,
            )
        return self._mars_temp

    @property
    def down_mars_dem(self) -> np.ndarray:
        if self._down_mars_dem is None:
            self._down_mars_dem = get_data_image(
                self.planet_cfg.data_dir,
                (self.mars_h, self.mars_w),
                "Mars_DEM_WxH.png",
                default_shape=(4362, 8724),
                interpolation=cv2.INTER_AREA,
            )
        return self._down_mars_dem

    @property
    def down_mars_sat(self) -> np.ndarray:
        if self._down_mars_sat is None:
            self._down_mars_sat = get_data_image(
                self.planet_cfg.data_dir,
                (self.mars_h, self.mars_w),
                "Mars_Sat_WxH.png",
                default_shape=(4362, 8724),
                interpolation=cv2.INTER_LANCZOS4,
            )
        return self._down_mars_sat

    @property
    def down_mars_temp(self) -> np.ndarray:
        if self._down_mars_temp is None:
            self._down_mars_temp = get_data_image(
                self.planet_cfg.data_dir,
                (self.mars_h, self.mars_w),
                "Mars_Temp_WxH.png",
                default_shape=(512, 1024),
                interpolation=cv2.INTER_NEAREST,
            )
        return self._down_mars_temp

    @property
    def sat_summer(self) -> np.ndarray:
        """
        Summer in the Northern Hemisphere
        """
        if self._sat_summer is None:
            self._sat_summer = get_data_image(
                self.planet_cfg.data_dir,
                (self.H, self.W),
                "world.satellite.WxH.summer.png",
                default_shape=(8192, 16384),
                interpolation=cv2.INTER_LANCZOS4,
            )
        return self._sat_summer

    @property
    def temp_summer(self) -> np.ndarray:
        """
        Summer in the Northern Hemisphere
        """
        if self._temp_summer is None:
            self._temp_summer = get_data_image(
                self.planet_cfg.data_dir,
                (self.H, self.W),
                "World_Temp_WxH_summer.png",
                default_shape=(8192, 16384),
                interpolation=cv2.INTER_AREA,
            )
        return self._temp_summer

    @property
    def precipitation(self) -> np.ndarray:
        if self._precipitation is None:
            self._precipitation = get_data_image(
                self.planet_cfg.data_dir,
                (self.H, self.W),
                "Precipitation_2004_01_WxH.png",
                default_shape=(8192, 16384),
                interpolation=cv2.INTER_LANCZOS4,
            )
        return self._precipitation

    @property
    def precipitation_summer(self) -> np.ndarray:
        """
        Summer in the Northern Hemisphere
        """
        if self._precipitation_summer is None:
            self._precipitation_summer = get_data_image(
                self.planet_cfg.data_dir,
                (self.H, self.W),
                "Precipitation_2004_06_WxH.png",
                default_shape=(8192, 16384),
                interpolation=cv2.INTER_LANCZOS4,
            )
        return self._precipitation_summer

    @property
    def river_upa(self) -> np.ndarray:
        if self._river_upa is None:
            self._river_upa = np.nan_to_num(get_data_image(
                self.planet_cfg.data_dir,
                (self.H, self.W),
                "river_upa_WxH.tif",
                default_shape=(8192, 16384),
                custom_resizer=lambda x, shape: line_resizer(
                    x, shape, thickness=1, min_val=5000 * self.planet_cfg.full_delta
                ),
                # expiration_date="01/09/2026"
            ))
        return self._river_upa


def quad_data_loader(
    data_dir: str,
    shape: tuple[int, int],
    quadsphere: QuadSphere,
) -> Callable[[str, np.ndarray, bool, bool], np.ndarray]:
    @timing
    def loader(
        name: str, normal_atlas: np.ndarray, discrete: bool = False,
        use_cached: bool = True, sampling_mode: SampleMode = "mean"
    ) -> np.ndarray:
        H, W = shape
        quad_path = os.path.join(data_dir, ".cache", f"quad_{name}_{W}x{H}.npy")
        os.makedirs(os.path.join(data_dir, ".cache"), exist_ok=True)
        if os.path.exists(quad_path) and use_cached:
            quad_data = np.load(quad_path)
        else:
            quadsphere.discrete = discrete
            quadsphere.sampling_mode = sampling_mode
            quad_data = quadsphere.get_quad_sphere_atlas(normal_atlas)
            np.save(quad_path, quad_data)
        return quad_data
    return loader


class QuadAtlasLoader:
    def __init__(self, planet_cfg: PlanetConfig, atlas_loader: Optional[AtlasLoader] = None):
        self.planet_cfg = planet_cfg
        self._quad_boundary_sketch: Optional[np.ndarray] = None
        self._quad_river_upa: Optional[np.ndarray] = None
        self.quad_sphere_mapping = QuadSphere(shape=(planet_cfg.H, planet_cfg.W))
        self.quad_loader = quad_data_loader(
            planet_cfg.data_dir,
            (planet_cfg.H, planet_cfg.W),
            self.quad_sphere_mapping
        )
        self.atlas_loader = atlas_loader or AtlasLoader(planet_cfg)

    @property
    def quad_boundary_sketch(self) -> np.ndarray:
        if self._quad_boundary_sketch is None:
            self._quad_boundary_sketch = get_data_image(
                self.planet_cfg.data_dir,
                self.quad_sphere_mapping.quad_shape,
                "quad_boundary_line_sketch_WxH.png",
                default_shape=(652, 3912),
                interpolation=cv2.INTER_NEAREST,
                custom_resizer=line_resizer,
            )
        return self._quad_boundary_sketch

    @property
    def quad_river_upa(self) -> np.ndarray:
        if self._quad_river_upa is None:
            self._quad_river_upa = self.quad_loader(
                "river_upa",
                self.atlas_loader.river_upa,
                discrete=False,
                use_cached=False,
                sampling_mode="mode"
            )
        return self._quad_river_upa

    @property
    def quad_dem(self) -> np.ndarray:
        return self.quad_loader(
            "dem",
            self.atlas_loader.float_dem,
            discrete=False,
            use_cached=True,
            sampling_mode="mean"
        )

    @property
    def quad_landmask(self) -> np.ndarray:
        return self.quad_loader(
            "landmask",
            self.atlas_loader.landmask,
            discrete=True,
            use_cached=True,
            sampling_mode="mode"
        )

    @property
    def quad_sat(self) -> np.ndarray:
        return self.quad_loader(
            "sat",
            self.atlas_loader.sat,
            discrete=False,
            use_cached=True,
            sampling_mode="mean"
        )

    @property
    def quad_rivers(self) -> np.ndarray:
        return self.quad_loader(
            "rivers",
            self.atlas_loader.rivers,
            discrete=True,
            use_cached=True,
            sampling_mode="mode"
        )

    @property
    def quad_temp(self) -> np.ndarray:
        return self.quad_loader(
            "temp",
            self.atlas_loader.temp,
            discrete=False,
            use_cached=True,
            sampling_mode="mean"
        )

    @property
    def quad_land(self) -> np.ndarray:
        return self.quad_loader(
            "land",
            self.atlas_loader.land,
            discrete=True,
            use_cached=True,
            sampling_mode="mode"
        )

    @property
    def quad_modal_sketch(self) -> np.ndarray:
        return self.quad_loader(
            "modal_sketch",
            self.atlas_loader.modal_sketch,
            discrete=True,
            use_cached=True,
            sampling_mode="mode"
        )

    @property
    def quad_modal_river_sketch(self) -> np.ndarray:
        return self.quad_loader(
            "modal_river_sketch",
            self.atlas_loader.modal_river_sketch,
            discrete=True,
            use_cached=True,
            sampling_mode="mode"
        )

    @property
    def quad_bathy(self) -> np.ndarray:
        return self.quad_loader(
            "bathy",
            self.atlas_loader.bathy,
            discrete=False,
            use_cached=True,
            sampling_mode="mean"
        )

    @property
    def quad_sat_summer(self) -> np.ndarray:
        return self.quad_loader(
            "sat_summer",
            self.atlas_loader.sat_summer,
            discrete=False,
            use_cached=True,
            sampling_mode="mean"
        )

    @property
    def quad_temp_summer(self) -> np.ndarray:
        return self.quad_loader(
            "temp_summer",
            self.atlas_loader.temp_summer,
            discrete=False,
            use_cached=True,
            sampling_mode="mean"
        )

    @property
    def quad_precipitation(self) -> np.ndarray:
        return self.quad_loader(
            "precipitation",
            self.atlas_loader.precipitation,
            discrete=False,
            use_cached=True,
            sampling_mode="mean"
        )

    @property
    def quad_precipitation_summer(self) -> np.ndarray:
        return self.quad_loader(
            "precipitation_summer",
            self.atlas_loader.precipitation_summer,
            discrete=False,
            use_cached=True,
            sampling_mode="mean"
        )
