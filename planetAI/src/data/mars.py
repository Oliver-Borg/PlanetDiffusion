from PIL import Image
import numpy as np
from cv2 import dilate, resize, INTER_LANCZOS4, INTER_NEAREST

from .utils import np_rgb_to_hex, np_hex_to_rgb, hex_to_rgb
Image.MAX_IMAGE_PIXELS = 1000000000


im = np.array(Image.open('data/mars-temp-1024x512.jpg')).max(axis=2).astype(np.uint8)

invalid_mask = (im % 50) != 0

im[invalid_mask] = 0

im = dilate(im, np.ones((5, 5)))

Image.fromarray(im).save('data/Mars_Temp_1024x512.png')

sat = np.array(Image.open('data/mars-sat.jpg'))

dem = np.array(Image.open('data/MarsHRSCMOLA_MAP2_SIMP.tif'))

mars_shape = (4362, 8724)
h, w = mars_shape
# Min-Max Normalization
dem = dem[1:, :-1] # Last pixel is a border pixel
dem = np.concatenate([dem[:, w//2:], dem[:, :w//2]], axis=1)
dem[dem < -16000] = 0
dem = ((dem - dem.min()) / (dem.max() - dem.min()) * 255).astype(np.uint8)

Image.fromarray(dem).resize((w, h)).save(f'data/Mars_DEM_{w}x{h}.png')
Image.fromarray(sat).resize((w, h)).save(f'data/Mars_Sat_{w}x{h}.png')

# im = resize(im, (sat.shape[1], sat.shape[0]))
sat = resize(sat, (im.shape[1], im.shape[0]), interpolation=INTER_LANCZOS4)



hex_sat = np_rgb_to_hex(sat)
hex_modal_sketch = np.zeros_like(hex_sat)
hex_palette = np.zeros((5, 100), dtype=np.uint32)

mapping = {}

for i in range(5):
    temp = (i + 1) * 50
    temp_mask = im == temp
    colours, counts = np.unique(hex_sat[temp_mask], return_counts=True)
    colours = colours[np.argsort(counts)][::-1]
    hex_modal_sketch[temp_mask] = colours[7] # Chosen by hand
    hex_palette[i] = colours[:100]
    mapping[i+1] = hex_to_rgb(int(colours[7]))

palette = np_hex_to_rgb(hex_palette)
palette = resize(palette, (1000, 50), interpolation=INTER_NEAREST)

with open('data/mars_mapping.txt', 'w') as f:
    for i in range(5):
        f.write(f'{i+1} {" ".join(list(map(str, mapping[i+1])))}\n')



modal_sketch = np_hex_to_rgb(hex_modal_sketch)
Image.fromarray(modal_sketch).save('data/Mars_Modal_Sketch.png')

