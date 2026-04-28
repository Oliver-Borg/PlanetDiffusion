from PIL import Image
import numpy as np
Image.MAX_IMAGE_PIXELS = 933120000
import os
import cv2
from .utils import brush_mask, modal_resize, PlanetConfig
from .sketch_gen import dilate_paint, landcover_paint



# Colours
#006400 - Tree cover
#ffbb22 - Shrubland
#ffff4c - Grassland
#f096ff - Cropland
#fa0000 - Built-up
#b4b4b4 - Bare / sparse vegetation
#f0f0f0 - Snow and ice
#c0ffff - Antarctica
#0064c8 - Permanent water bodies
#0096a0 - Herbaceous wetland
#00cf75 - Mangroves
#fae6a0 - Moss and lichen
#000000 - Open water

colours = [
    [0, 100, 0],
    [255, 187, 34],
    [255, 255, 76],
    [240, 150, 255],
    [250, 0, 0],
    [180, 180, 180],
    [240, 240, 240],
    [192, 255, 255],
    [0, 100, 200],
    [0, 150, 160],
    [0, 207, 117],
    [250, 230, 160],
    [0, 0, 0]
]

int_colours = [
    0x006400,
    0xffbb22,
    0xffff4c,
    0xf096ff,
    0xfa0000,
    0xb4b4b4,
    0xf0f0f0,
    0xc0ffff,
    0x0064c8,
    0x0096a0,
    0x00cf75,
    0xfae6a0,
    0x000000
]



im = Image.open("data/World_LandCover_21600x10800.tif")
dem = np.array(Image.open("data/World_DEM_16384x8192.png"))
sat = np.array(Image.open("data/world.satellite.16384x8192.png"))
watermask = Image.open("data/world.watermask.21600x10800.png")
oceanmask = Image.open("data/world.oceanmask.21600x10800.png")
# Resize landcover and others to match DEM
im = im.resize((dem.shape[1], dem.shape[0]), Image.NEAREST)
watermask = np.array(watermask.resize((dem.shape[1], dem.shape[0]), Image.NEAREST)) < 128
oceanmask = np.array(oceanmask.resize((dem.shape[1], dem.shape[0]), Image.NEAREST)) < 128

rivermask = np.logical_and(watermask, ~oceanmask)
# Preprocess image
im = np.array(im, dtype=np.uint32)
ocean = np.logical_and(im[:, :, 0] == 0, im[:, :, 1] == 0, im[:, :, 2] == 0)
dem_ocean = dem == 0
im[dem_ocean] = [0, 0, 0]
missing = np.logical_and(ocean, dem > 0)
top_mask = np.zeros(shape=(im.shape[0], im.shape[1]), dtype=bool)
top_mask[:im.shape[0]//2, :] = True
# Change pixels below half the height to Antarctica
im[missing & ~top_mask] = [192, 255, 255]

# Change pixels above half the height to snow and ice
im[missing & top_mask] = [240, 240, 240]

# Ocean: 2, 5, 20

sat_ocean = np.logical_and(sat[:, :, 0] == 2, sat[:, :, 1] == 5, sat[:, :, 2] == 20)
# Change bottom 1288 pixels that have no sat ocean, but have landcover ocean to Antarctica
bottom_mask = np.zeros(shape=(im.shape[0], im.shape[1]), dtype=bool)
bottom_mask[im.shape[0]-1288:, :] = True
im[ocean & ~sat_ocean & bottom_mask] = [192, 255, 255]
im[dem_ocean] = [0, 0, 0]



landcover_missing = np.logical_and(im[:, :, 0] == 0, im[:, :, 1] == 0, im[:, :, 2] == 0)
missing = np.logical_and(landcover_missing, ~oceanmask)

# Change extra water to permanent water
# im[extra_water] = [0, 100, 200]





integer = np.zeros(shape=(im.shape[0], im.shape[1]), dtype=np.int32)
integer[:, :] = im[:, :, 0] * 256 * 256 + im[:, :, 1] * 256 + im[:, :, 2]

small_integer = modal_resize(integer, 32, exclude_zeros=True, use_fast=True)
values = np.unique(small_integer)
for i, value in enumerate(values):
    small_integer[small_integer == value] = i
small_integer = small_integer.astype(np.uint8)
dilated_integer = cv2.dilate(small_integer, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                            iterations=3)
small_integer[small_integer == 0] = dilated_integer[small_integer == 0]
small_integer = cv2.resize(small_integer, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
small_integer = small_integer.astype(np.uint64)
for i, value in enumerate(values):
    small_integer[small_integer == i] = value


# Change river pixels to permanent water
integer[missing] = small_integer[missing]
integer[rivermask] = 0x0064c8


# Check all colours are present
# for i in range(len(int_colours)):
#     assert np.any(integer == int_colours[i])

# Change permanent water in antarctica region to antarctica
integer[(integer == 0x0064c8) & bottom_mask] = 0xc0ffff
# Change snow and ice in antarctica region to antarctica
integer[(integer == 0xf0f0f0) & bottom_mask] = 0xc0ffff
# Change moss and lichen to snow and ice
integer[integer == 0xfae6a0] = 0xf0f0f0
# Change herbaceous wetland to wetland
integer[integer == 0x0096a0] = 0x0096a0
# Change mangroves to wetland
integer[integer == 0x00cf75] = 0x0096a0

im[:, :, 0] = integer // (256 * 256)
im[:, :, 1] = (integer // 256) % 256
im[:, :, 2] = integer % 256
im = np.array(im, dtype=np.uint8)
im = Image.fromarray(im)

maxsize = 5
im.save(f"data/World_LandCover_RGB_16384x8192.png")

planet_cfg = PlanetConfig(dilate_iters=0, erode_iters=0)

H = 256 * 2 ** maxsize
W = 2 * H
im = Image.open(f"data/World_LandCover_RGB_{W}x{H}.png")
im = np.array(im, dtype=np.uint32)
integer = np.zeros(shape=(im.shape[0], im.shape[1]), dtype=np.uint64)
integer[:, :] = im[:, :, 0] * 256 * 256 + im[:, :, 1] * 256 + im[:, :, 2]
dem = np.array(Image.open(f"data/World_DEM_{W}x{H}.png"))
stacked = np.dstack([integer > 0, dem > 0, np.zeros(shape=dem.shape, dtype=bool)])
stacked = Image.fromarray(stacked.astype(np.uint8) * 255)
vals, counts = np.unique(integer, return_counts=True)
order_array = np.argsort(counts)[::-1]
vals = vals[order_array]
counts = counts[order_array]
ordered = np.zeros(shape=(integer.shape[:2]), dtype=np.uint8)
num_cols = vals.shape[0] - 1
rgb_vals = np.zeros(shape=(vals.shape[0], 3), dtype=np.uint8)
for i in range(vals.shape[0]):
    col = vals[i]
    rgb_vals[i, 0] = col // (256 * 256)
    rgb_vals[i, 1] = (col // 256) % 256
    rgb_vals[i, 2] = col % 256
print(rgb_vals)
for i in range(vals.shape[0]):
    ordered[integer == vals[i]] = i
for size in range(maxsize+1):
    h = 256 * 2 ** size
    w = 2 * h
    # gray = cv2.resize(ordered, (w, h), interpolation=cv2.INTER_AREA)
    dx = 2**(maxsize - size)
    if dx < 32:
        gray = modal_resize(ordered, dx, exclude_zeros=False, use_fast=True)
    else:
        gray = modal_resize(ordered, dx, exclude_zeros=True, use_fast=True)
    # gray = np.array(Image.open(f"data/World_LandCover_{w}x{h}.png")) // (256 // num_cols)
    dem = Image.open(f"data/World_DEM_{w}x{h}.png")
    dem = np.array(dem, dtype=np.uint8)
    _oceanmask = cv2.resize(oceanmask.astype(np.uint8)*255, (w, h), interpolation=cv2.INTER_NEAREST) < 128
    gray[_oceanmask == 0] = 0
    sketch = dilate_paint(dem, planet_cfg)
    landcover = landcover_paint(gray*(255//num_cols), planet_cfg)
    land_extra = np.logical_and(gray == 0, landcover > 0)
    sketch_extra = np.logical_and(dem == 0, sketch > 0)
    missing = np.dstack([dem > 0, sketch_extra, land_extra]).astype(np.uint8)*255
    missing = Image.fromarray(missing)
    Image.fromarray(gray*(255//num_cols), mode="L").save(f"data/World_LandCover_{w}x{h}.png")
    rgb_im = np.zeros(shape=(gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
    for i in range(num_cols+1):
        col = vals[i]
        r = col // (256 * 256)
        g = (col // 256) % 256
        b = col % 256
        rgb_im[gray == i] = [r, g, b]
    Image.fromarray(rgb_im).save(f"data/World_LandCover_RGB_{w}x{h}.png")
