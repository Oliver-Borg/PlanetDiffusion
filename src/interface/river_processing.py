import os

import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import cv2
from skimage import filters
from skimage.morphology import square
from skimage.morphology import disk
from skimage.segmentation import watershed
from skimage import data
from skimage.filters import rank
from skimage.util import img_as_ubyte
from scipy import ndimage as ndi

from .panels import GlobePanel

from planetAI.src.data.river_processing import (
    RIVER_LEN_BOUNDS,
    RIVER_WIDTH_BOUNDS,
    RIVER_ORDER_BOUNDS,
    RIVER_FILTERS,
    apply_filters
)
    

class RiverGlobePanel(GlobePanel):
    init_complete = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_river_length_1 = tk.IntVar(value=6532)
        self.min_river_length_2 = tk.IntVar(value=3421)
        self.min_river_length_3 = tk.IntVar(value=2271)
        self.min_river_width_1 = tk.IntVar(value=2000)
        self.min_river_width_2 = tk.IntVar(value=2194)
        self.min_river_width_3 = tk.IntVar(value=3100)
        self.min_river_order = tk.IntVar(value=0)
        self.max_river_order = tk.IntVar(value=10)
        self.outline_dist = tk.IntVar(value=5)
        self.outline_opacity = tk.DoubleVar(value=0.5)
        self.init_complete = True

    def var_changed(self, *args):
        self.display_preview()

    def post_process(self, preview):
        # Do things here for preview
        sat = preview[:, :, :3]


        # Attempted implementation from River Detection in Remotely Sensed Imagery Using Gabor Filtering and Path Opening
        # sat = cv2.cvtColor(sat, cv2.COLOR_RGB2GRAY)

        # # 1. Apply 3x3 mean filter to reduce salt and pepper noise
        # sat = filters.rank.mean(sat, square(3))

        # # 2. Apply contrast limited adaptive histogram equalization (CLAHE)
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # sat = clahe.apply(sat)

        # # 3. Next, shade correction was performed by subtracting an image approximating the background. This approximation was obtained by applying a mean filter with a 50 x 50 pixel size kernel.
        # background = filters.rank.mean(sat, square(50))
        # sat = sat.astype(np.int16) - background
        # sat = cv2.normalize(sat, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # # 4. The gray values of the filtered image were then inversed to represent rivers as bright features
        # sat = 255 - sat
        # sat = (sat / 100).round()
        # sat = cv2.normalize(sat, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        # return sat
        # # 5. Apply Gabor filter to enhance river features
        # gabor_real = cv2.filter2D(sat, cv2.CV_32F, cv2.getGaborKernel((21, 21), 10.0, 0.0, 10.0, 0.5, 0, ktype=cv2.CV_32F), borderType=cv2.BORDER_CONSTANT)
        
        # # 6. Normalize the image
        # gabor_real = cv2.normalize(gabor_real, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        # return gabor_real
        
        if preview.shape[2] > 3 and self.init_complete:
            river_length = preview[:, :, 3]
            river_width = preview[:, :, 4]
            river_order = preview[:, :, 5]
            RIVER_FILTERS[0].set_river_length(self.min_river_length_1.get())
            RIVER_FILTERS[0].set_river_width(self.min_river_width_1.get())
            RIVER_FILTERS[1].set_river_length(self.min_river_length_2.get())
            RIVER_FILTERS[1].set_river_width(self.min_river_width_2.get())
            RIVER_FILTERS[2].set_river_length(self.min_river_length_3.get())
            RIVER_FILTERS[2].set_river_width(self.min_river_width_3.get())
            min_ro_val = self.min_river_order.get()
            max_ro_val = self.max_river_order.get()
            # TODO Add river dilate controls properly
            is_river = apply_filters(river_length, river_width, river_order, RIVER_FILTERS, 0, 10, min_ro_val, max_ro_val)
            d = self.outline_dist.get()
            if d > 1:
                is_river = cv2.dilate(is_river.astype(np.uint8), np.ones((d, d), np.uint8), iterations=1)
                is_river -= cv2.dilate(is_river.astype(np.uint8), np.ones((d-1, d-1), np.uint8), iterations=1)
            # river_outline = cv2.Canny(is_river.astype(np.uint8), 0, 1)
            river_outline = is_river
            sat[river_outline > 0] = (sat[river_outline > 0] * self.outline_opacity.get()).astype(np.uint8)
        return sat
    


if __name__ == "__main__":
    root = tk.Tk()
    shape = (16384, 8192)
    force = True
    if not force and os.path.exists('stacked_atlas.npy'):
        stacked_atlas = np.load('stacked_atlas.npy')
    else:
        sat_atlas = np.array(Image.open(f'./planetAI/data/world.satellite.16384x8192.png').resize(shape))
        river_length = np.array(Image.open(f'./planetAI/data/River_Length_16384.tif').resize(shape))
        river_width = np.array(Image.open(f'./planetAI/data/River_Width_16384.tif').resize(shape))
        river_orders = np.array(Image.open(f'./planetAI/data/global_river_order_16384.tif').resize(shape))
        # river_length = cv2.dilate(river_length, np.ones((5, 5), np.uint8), iterations=1)
        # river_width = cv2.dilate(river_width, np.ones((5, 5), np.uint8), iterations=1)
        # river_orders = cv2.dilate(river_orders, np.ones((5, 5), np.uint8), iterations=1)
        stacked_atlas = np.dstack([sat_atlas, river_length, river_width, river_orders])
        # Cache the npy file
        np.save('stacked_atlas.npy', stacked_atlas)
    shape = tuple(stacked_atlas.shape[:2])
    globe_panel = RiverGlobePanel(master=root, shape=shape, preview_size=512)
    globe_panel.set_normal_atlas(stacked_atlas)
    globe_panel.grid(row=0, column=0, columnspan=3)

    # Add controls for river processing
    min_river_length_slider_1 = tk.Scale(
        root, from_=RIVER_LEN_BOUNDS[0], to=RIVER_LEN_BOUNDS[1], orient=tk.HORIZONTAL,
        label='River Len 1', variable=globe_panel.min_river_length_1, command=globe_panel.var_changed
    )
    min_river_length_slider_1.grid(row=1, column=0)
    min_river_width_slider_1 = tk.Scale(
        root, from_=RIVER_WIDTH_BOUNDS[0], to=RIVER_WIDTH_BOUNDS[1], orient=tk.HORIZONTAL,
        label='River Width 1', variable=globe_panel.min_river_width_1, command=globe_panel.var_changed
    )
    min_river_width_slider_1.grid(row=2, column=0)

    min_river_length_slider_2 = tk.Scale(
        root, from_=RIVER_LEN_BOUNDS[0], to=RIVER_LEN_BOUNDS[1], orient=tk.HORIZONTAL,
        label='River Len 2', variable=globe_panel.min_river_length_2, command=globe_panel.var_changed
    )
    min_river_length_slider_2.grid(row=1, column=1)
    min_river_width_slider_2 = tk.Scale(
        root, from_=RIVER_WIDTH_BOUNDS[0], to=RIVER_WIDTH_BOUNDS[1], orient=tk.HORIZONTAL,
        label='River Width 2', variable=globe_panel.min_river_width_2, command=globe_panel.var_changed
    )
    min_river_width_slider_2.grid(row=2, column=1)

    min_river_length_slider_3 = tk.Scale(
        root, from_=RIVER_LEN_BOUNDS[0], to=RIVER_LEN_BOUNDS[1], orient=tk.HORIZONTAL,
        label='River Len 3', variable=globe_panel.min_river_length_3, command=globe_panel.var_changed
    )
    min_river_length_slider_3.grid(row=1, column=2)
    min_river_width_slider_3 = tk.Scale(
        root, from_=RIVER_WIDTH_BOUNDS[0], to=RIVER_WIDTH_BOUNDS[1], orient=tk.HORIZONTAL,
        label='River Width 3', variable=globe_panel.min_river_width_3, command=globe_panel.var_changed
    )
    min_river_width_slider_3.grid(row=2, column=2)


    min_river_order_slider = tk.Scale(
        root, from_=RIVER_ORDER_BOUNDS[0], to=RIVER_ORDER_BOUNDS[1], orient=tk.HORIZONTAL,
        label='River Order', variable=globe_panel.min_river_order, command=globe_panel.var_changed
    )
    min_river_order_slider.grid(row=3, column=0)
    max_river_order_slider = tk.Scale(
        root, from_=RIVER_ORDER_BOUNDS[0], to=RIVER_ORDER_BOUNDS[1], orient=tk.HORIZONTAL,
        label='River Order', variable=globe_panel.max_river_order, command=globe_panel.var_changed
    )
    max_river_order_slider.grid(row=3, column=1)
    outline_dist_slider = tk.Scale(
        root, from_=1, to=20, orient=tk.HORIZONTAL, label='Outline Distance',
        variable=globe_panel.outline_dist, command=globe_panel.var_changed
    )
    outline_dist_slider.grid(row=4, column=0)
    outline_opacity_slider = tk.Scale(
        root, from_=0, to=1, orient=tk.HORIZONTAL, label='Outline Opacity',
        variable=globe_panel.outline_opacity, resolution=0.05, command=globe_panel.var_changed
    )
    outline_opacity_slider.grid(row=4, column=1)
    root.mainloop()