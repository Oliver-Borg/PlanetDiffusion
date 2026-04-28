import tkinter as tk
from PIL import ImageTk, Image
from PIL import Image as img
import numpy as np
import math


from src.data.sphere_mapping import SphereMapping

tile_size = 512
size = 5
W = 512*2**size
H = 256*2**size
atlas = img.open(f"./data/world.satellite.{W}x{H}.png")
atlas = np.array(atlas)
sphere = SphereMapping(atlas)
tile = sphere.get_tile((-90, 0))


def update_image(lat=None, lon=None, rotation=None, zoom=None):
    if lat is None:
        lat = lat_slider.get()
    if lon is None:
        lon = lon_slider.get()
    if rotation is None:
        rotation = rotation_slider.get()
    if zoom is None:
        zoom = zoom_slider.get()
    # Get the tile image
    tile = sphere.get_tile((lat, lon), rotation, tile_size, zoom=zoom**2)
    # Convert the image to a format that can be used in Tkinter
    image = ImageTk.PhotoImage(Image.fromarray(tile))
    # Update the image_label
    image_label.config(image=image)
    image_label.image = image

# Create the main window
root = tk.Tk()

# Create the latitude and longitude sliders
lat_slider = tk.Scale(root, orient=tk.HORIZONTAL, from_=-90, to=90, command=lambda x: update_image(lat=float(x)), resolution=0.1)
lon_slider = tk.Scale(root, orient=tk.HORIZONTAL, from_=-180, to=180, command=lambda x: update_image(lon=float(x)), resolution=0.1)
rotation_slider = tk.Scale(root, orient=tk.HORIZONTAL, from_=-15, to=15, command=lambda x: update_image(rotation=float(x)), resolution=1)
zoom_slider = tk.Scale(root, orient=tk.HORIZONTAL, from_=0.1, to=3, command=lambda x: update_image(zoom=float(x)), resolution=0.1)
zoom_slider.set(1)
# Create a label to display the image
image_label = tk.Label(root)
image_label.pack()

# Pack the sliders
lat_slider.pack()
lon_slider.pack()
rotation_slider.pack()
zoom_slider.pack()

# Start the main loop
update_image(0, 0, 0)
root.mainloop()