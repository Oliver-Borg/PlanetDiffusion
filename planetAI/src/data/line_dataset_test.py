
from PIL import Image, ImageTk
import tkinter as tk
import cv2

from .utils import get_data_image, PlanetConfig
from .sphere_mapping import QuadSphere
from .line_dataset import line_resizer


def show_tile(
    canvas: tk.Canvas,
    sphere: QuadSphere,
    coord: tuple[float, float],
    tile_size: int = 256
) -> None:
    ys, xs = sphere.get_quad_tile_mapping(coord, tile_size=tile_size)
    sketch = _quad_boundary_line_sketch_data[ys, xs]
    bathy = sphere.get_quad_tile(coord=coord, atlas_indices=(ys, xs))
    bathy[sketch > 0] = sketch[sketch > 0]
    image = Image.fromarray(bathy)
    image = ImageTk.PhotoImage(image=image)
    canvas.create_image(0, 0, anchor=tk.NW, image=image)
    canvas.image = image  # Keep a reference to avoid garbage collection


def create_preview_frame(parent: tk.Tk, tile_size: int = 256) -> tk.Frame:
    """
    Returns a frame with a canvas for displaying the image and controls to change the latitude and longitude.
    """
    frame = tk.Frame(parent)
    frame.pack()

    canvas = tk.Canvas(frame, width=tile_size, height=tile_size)
    canvas.pack()

    # Store current coordinates
    coords = {"lat": -37.685074, "lon": 49.381831}
    coord_label = tk.Label(frame, text=f"Lat: {coords['lat']:.2f}, Lon: {coords['lon']:.2f}")
    coord_label.pack()

    # Variables for tracking mouse movement
    pan_data = {"prev_x": 0, "prev_y": 0, "panning": False}

    def update_image():
        coord_label.config(text=f"Lat: {coords['lat']:.2f}, Lon: {coords['lon']:.2f}")
        show_tile(canvas, sphere, (coords['lat'], coords['lon']), tile_size)

    def start_pan(event):
        pan_data["panning"] = True
        pan_data["prev_x"] = event.x
        pan_data["prev_y"] = event.y
        canvas.config(cursor="fleur")  # Change cursor to indicate panning

    def stop_pan(event):
        pan_data["panning"] = False
        canvas.config(cursor="")  # Restore default cursor

    def pan(event):
        if pan_data["panning"]:
            # Calculate delta movement
            dx = event.x - pan_data["prev_x"]
            dy = event.y - pan_data["prev_y"]

            # Update longitude (x movement) - invert for natural feel
            coords["lon"] -= dx * 0.25
            coords["lon"] = ((coords["lon"] + 180) % 360) - 180  # Wrap around

            # Update latitude (y movement) - invert for natural feel
            coords["lat"] += dy * 0.25
            coords["lat"] = max(-90, min(90, coords["lat"]))  # Clamp to valid range

            # Save current position
            pan_data["prev_x"] = event.x
            pan_data["prev_y"] = event.y

            # Update the display
            update_image()

    # Bind mouse events
    canvas.bind("<Button-2>", start_pan)  # Middle mouse button press
    canvas.bind("<ButtonRelease-2>", stop_pan)  # Middle mouse button release
    canvas.bind("<B2-Motion>", pan)  # Middle mouse button drag

    # Initial display
    update_image()

    return frame


if __name__ == "__main__":
    planet_cfg = PlanetConfig(size=4)
    H, W = planet_cfg.H, planet_cfg.W
    _bathy_data = get_data_image(
        planet_cfg.data_dir,
        (H, W),
        "gebco_bathy.WxH.jpg",
        default_shape=(10801, 21601),
        interpolation=cv2.INTER_AREA,
    )
    sphere = QuadSphere(atlas=_bathy_data)
    _quad_boundary_line_sketch_data = get_data_image(
        planet_cfg.data_dir,
        sphere.quad_shape,
        "quad_boundary_line_sketch_WxH.png",
        default_shape=(652, 3912),
        interpolation=cv2.INTER_NEAREST,
        custom_resizer=line_resizer,
    )
    # sketch_sphere.add_line((0, -170), (0, 170), 1, 255)

    app = tk.Tk()
    app.title("Line Dataset Preview")
    app.geometry("1200x800")
    preview_frame = create_preview_frame(app, tile_size=768)
    app.mainloop()
