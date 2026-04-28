from threading import Thread
import time
import json
from pathlib import Path
import argparse

import numpy as np
from PIL import Image, ImageTk
import tkinter as tk

from planetAI.src.data.utils import (
    tensor_to_np,
)

from .inference_tests import TileGenerator, generate_configs, merge_config


class TestDisplay(tk.Frame):
    def __init__(
        self,
        master: tk.Widget,
        tile_generator: TileGenerator,
        shape=(768, 768),
    ):
        super().__init__(master)
        self.shape = shape
        self.tile_generator = tile_generator
        self.grid()
        self.create_widgets()
        self.update_display_thread = Thread(target=self.update_display)
        self.thread_running = True
        self.update_display_thread.start()
        self.generation_thread = None
        self.start_generation()

    def create_widgets(self):
        c_h, c_w = self.shape
        self.canvas = tk.Canvas(self, width=c_w, height=c_h)
        self.canvas.grid(row=0, column=0)
        self.image = None
        self.start_button = tk.Button(
            self, text="Generate Samples", command=self.start_generation
        )
        self.start_button.grid(row=1, column=0)

    def update_display(self):
        while self.thread_running:
            sat_tensor = self.tile_generator.inference_controller.current_sat_output
            if sat_tensor is not None:
                sat_h, sat_w = sat_tensor.shape[:2]
                h, w = self.shape
                ys = np.linspace(0, sat_h, h).astype(int).clip(0, sat_h - 1)
                xs = np.linspace(0, sat_w, w).astype(int).clip(0, sat_w - 1)
                xs, ys = np.meshgrid(xs, ys)
                display_tile = sat_tensor[ys, xs]
                display_tile = tensor_to_np(display_tile)
                image = Image.fromarray(display_tile)
                image = image.resize(self.shape, resample=Image.NEAREST)
                try:
                    self.image = ImageTk.PhotoImage(image)
                    self.canvas.create_image(0, 0, image=self.image, anchor=tk.NW)
                except Exception:
                    pass
            time.sleep(1)

    def start_generation(self):
        if self.generation_thread is not None:
            self.generation_thread.join()
        self.generation_thread = Thread(target=self.generate_samples)
        self.generation_thread.start()

    def generate_samples(self):
        self.tile_generator.generate_grid()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Terrain Generation Interface")
    parser.add_argument(
        "--config",
        type=str,
        default="inference_config.json",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--num_images", type=int, default=100, help="Number of images to generate"
    )
    parser.add_argument(
        "--experiment_name", type=str, default="Planet", help="Name of the experiment"
    )

    args = parser.parse_args()

    # Load config and get experiment parameters
    config_path = Path(args.config)
    with open(config_path) as f:
        config = json.load(f)

    all_experiment_configs = generate_configs(config["experiments"][args.experiment_name])

    final_config = merge_config(config["default"], all_experiment_configs[0])
    experiment_params = final_config["experiment"]

    generator = TileGenerator(
        experiment_name=args.experiment_name,
        config=final_config,
        output_size=experiment_params["output_size"],
        seed=experiment_params["seed"],
        mars=experiment_params["mars"],
        shuffle_landcover=experiment_params["shuffle_landcover"],
        temp_variance=experiment_params["temp_variance"],
        do_upscaling=experiment_params["do_upscaling"],
        num_images=args.num_images,
    )
    root = tk.Tk()
    inference_display = TestDisplay(root, generator, shape=(768, 768))
    inference_display.pack()
    root.mainloop()
    inference_display.thread_running = False
    inference_display.update_display_thread.join()
