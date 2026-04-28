import tkinter as tk
from PIL import Image, ImageTk
import numpy as np

from .rough_edge_mask import jigsaw_grid_mask


class RoughEdgeMaskTester(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.mask = None
        self.image = None
        self.canvas = tk.Canvas(root, width=768, height=768)
        self.canvas.pack()

        self.control_frame = tk.Frame(root)
        self.control_frame.pack()

        self.seed_slider = tk.Scale(
            self.control_frame, from_=0, to=100, label="Seed", command=self.onchange
        )
        self.seed_slider.pack(side="left")

        self.frequency_slider = tk.Scale(
            self.control_frame,
            from_=0,
            to=0.1,
            resolution=0.001,
            label="Frequency",
            command=self.onchange,
        )
        self.frequency_slider.pack(side="left")
        self.frequency_slider.set(0.1)

        self.amplitude_slider = tk.Scale(
            self.control_frame,
            from_=1,
            to=50,
            resolution=1,
            label="Amplitude",
            command=self.onchange,
        )
        self.amplitude_slider.pack(side="left")
        self.amplitude_slider.set(10)

        self.offset_x_slider = tk.Scale(
            self.control_frame, from_=0, to=100, label="Offset X", command=self.onchange
        )
        self.offset_x_slider.pack(side="left")

        self.offset_y_slider = tk.Scale(
            self.control_frame, from_=0, to=100, label="Offset Y", command=self.onchange
        )
        self.offset_y_slider.pack(side="left")

        self.grid_spacing_slider = tk.Scale(
            self.control_frame,
            from_=1,
            to=3,
            resolution=1,
            label="Grid Spacing",
            command=self.onchange,
        )
        self.grid_spacing_slider.pack(side="left")
        self.grid_spacing_slider.set(2)

    def onchange(self, event):
        seed = self.seed_slider.get()
        frequency = self.frequency_slider.get()
        amplitude = self.amplitude_slider.get()
        offset_x = self.offset_x_slider.get()
        offset_y = self.offset_y_slider.get()
        grid_spacing = self.grid_spacing_slider.get() / 4
        mask = jigsaw_grid_mask(
            (768, 768),
            256,
            grid_spacing=grid_spacing,
            grid_offset=(offset_y, offset_x),
            seed=seed,
            frequency=frequency,
            amplitude=amplitude,
        )
        self.mask = mask
        self.image = Image.fromarray(self.mask.astype(np.uint8) * 255)
        self.photo = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)


if __name__ == "__main__":

    jigsaw_grid = jigsaw_grid_mask((1024, 1024), seed=0)
    # Image.fromarray(jigsaw_grid).show()

    root = tk.Tk()
    root.title("Rough Edge Tile Mask")
    app = RoughEdgeMaskTester(master=root)
    app.mainloop()
