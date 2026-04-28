import numpy as np
import os
from PIL import Image

final_size = (16384, 8192)

# Create rbga array of zeros
output_array = np.zeros((final_size[1], final_size[0], 4), dtype=np.uint8)

data_dir = os.path.join(os.getcwd(), 'data/outputs')

dem = True
output_name = 'all_tiled_image.png'
if dem:
    output_name =  output_name.replace('.png', '_dem.png')


zoom_sizes = {
    0: "512x256",
    1: "1024x512",
    2: "2048x1024",
    3: "4096x2048",
    4: "8192x4096",
    5: "16384x8192",
    # 6: "32768x16384",
    # 7: "65536x32768",
}

needed_sizes = {
    0: (8192, 8192),
    1: (4096, 4096),
    2: (2048, 2048),
    3: (1024, 1024),
    4: (512, 512),
    5: (256, 256),
}    

def get_tile_directory(zoom, x, y, dem=False):
    if dem:
        return os.path.join(data_dir, f'tiles{zoom_sizes[zoom]}_dem/tile{y}_{x}.png')
    return os.path.join(data_dir, f'tiles{zoom_sizes[zoom]}/tile{y}_{x}.png')

def get_parents(zoom, x, y):
    for z in range(zoom-1, -1, -1):
        x = x // 2
        y = y // 2
        yield z, x, y

# tiles = [
#     (0, 0, 0),
#     (4, 5, 24),
#     (3, 4, 9),
# ]
tiles = [
   
]
zoom_level = 4
size = zoom_sizes[zoom_level].split('x')
for x in range(0, int(size[0])//256):
    for y in range(0, int(size[1])//256):
        tiles.append((zoom_level, x, y))


processed = set()

for zoom, x, y in tiles:
    parents = list(get_parents(zoom, x, y))
    parents.reverse()
    parents += [(zoom, x, y)]
    for z, x, y in parents:
        if (z, x, y) in processed:
            continue
        processed.add((z, x, y))
        tile_path = get_tile_directory(z, x, y, dem)
        print(tile_path)
        tile = Image.open(tile_path)
        tile_array = np.array(tile.resize(needed_sizes[z], Image.Resampling.BOX))
        border_size = 8
        tile_array[0:border_size, :, 0] = 255
        tile_array[0:border_size, :, 3] = 255
        tile_array[-border_size:, :, 0] = 255
        tile_array[-border_size:, :, 3] = 255
        tile_array[:, 0:border_size, 0] = 255
        tile_array[:, 0:border_size, 3] = 255
        tile_array[:, -border_size:, 0] = 255
        tile_array[:, -border_size:, 3] = 255
        output_array[
            y*needed_sizes[z][1]:(y+1)*needed_sizes[z][1],
            x*needed_sizes[z][0]:(x+1)*needed_sizes[z][0],
        ] = tile_array

output_image = Image.fromarray(output_array)
output_image.save(os.path.join(data_dir, output_name))