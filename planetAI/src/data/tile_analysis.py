from map_paster import setup
from utils import PlanetConfig
from PIL import Image as img
import numpy as np
import cv2

cfg = PlanetConfig(size=0)

for size in range(6):
    total_coastal_tiles = 0
    total_tiles = 0
    w = 512*2**size
    h = 256*2**size
    dem = np.array(img.open(f"data/World_DEM_{w}x{h}.png"))
    for i in range(7):
        coastal_tiles = 0
        tiles = 0
        mask = img.open(f"data/masks/{i}.png")
        mask = mask.resize((w, h), img.NEAREST)
        mask = mask.convert("1")
        mask = np.array(mask)
        landmass = dem * mask
        ops = cfg.operations
        num_ops = 1
        for k in ops:
            if k == 'rem':
                if not False in ops[k][i]:
                    num_ops = 0
                continue
            num_ops *= len(ops[k][i])
        num_ops -= 1
        if num_ops <= 0:
            continue
        best_tile = None
        max_val = 0
        for y in range(0, h, 256):
            for x in range(0, w, 256):
                for sea_level in [0]: # list(set(cfg.values['sea_level'])):
                    tile = landmass[y:y+256, x:x+256]
                    tile = np.where(tile < sea_level, 0, tile)
                    sketch = cv2.resize(tile, (8, 8), interpolation=cv2.INTER_NEAREST)
                    if np.count_nonzero(sketch) > 3+size:
                        tiles += 1
                        if np.max(tile) > max_val:
                            best_tile = tile
                            max_val = np.max(tile)
                        # If there is any water on the edge of the tile, it is coastal
                        # if not (np.all(tile[0]) and np.all(tile[-1]) and np.all(tile[:, 0]) and np.all(tile[:, -1])):
                        #     coastal_tiles += 1
                        # Check if there is more than 30% water
                        if np.count_nonzero(tile) < 256*256*0.7:
                            coastal_tiles += 1
        if best_tile is not None:
            img.fromarray(best_tile).save(f"analysis/{size}_{i}.png")
        total_coastal_tiles += coastal_tiles*num_ops
        total_tiles += tiles*num_ops
        # print(f"Size {size}, mask {i}: {coastal_tiles*num_ops}/{tiles*num_ops} coastal tiles")
    print(f"Size {size}: {total_coastal_tiles}/{total_tiles} coastal tiles")





