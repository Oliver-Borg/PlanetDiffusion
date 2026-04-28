# Script to move files from the format planet_cfg.dem_dir()/f"{mask}_{ref}_{ang}_{y}_{x}.{self.planet_cfg.image_extension}"
# to the format planet_cfg.dem_dir()/f"{mask}_{ref}_{ang}/{y}_{x}.{self.planet_cfg.image_extension}"

from utils import PlanetConfig
import os
import shutil
from dataclass_argparser import CustomArgumentParser

def move_files(planet_cfg: PlanetConfig):
    directories = [planet_cfg.dem_dir(), planet_cfg.sat_dir(), planet_cfg.land_dir()]
    downscale_cfg = planet_cfg.downscale_cfg
    directories += [downscale_cfg.dem_dir(), downscale_cfg.sat_dir(), downscale_cfg.land_dir()]
    for d in directories:
        tiles = os.listdir(d)
        for tile in tiles:
            if not '.' in tile:
                continue
            
            splt = tile.split('.')[0].split('_')
            if not len(splt) == 5:
                continue
            mask, ref, ang, y, x = splt
            new_dir = os.path.join(d, f"{mask}_{ref}_{ang}")
            if not os.path.exists(new_dir):
                os.mkdir(new_dir)
            shutil.move(os.path.join(d, tile), os.path.join(new_dir, f"{y}_{x}.{planet_cfg.image_extension}"))

if __name__ == "__main__":
    parser = CustomArgumentParser(
        (
            PlanetConfig
        ),
        description="Benchmark dataset loading speed" 
    )
    test_cfg = parser.parse_args_into_dataclasses()
    move_files(test_cfg)

    