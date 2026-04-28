import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
from utils import get_brush_deltas

# Define the initial dot size and image dimensions
INITIAL_DOT_SIZE = 10
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

class DotSizeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dot Size App")

        # Create the slider
        self.slider = tk.Scale(
            self.root,
            from_=0,
            to=10,
            resolution=1,
            length=200,
            orient=tk.HORIZONTAL,
            command=self.update_dot_size
        )
        self.slider.set(INITIAL_DOT_SIZE)
        self.slider.pack()

        # Create the canvas to display the image
        self.canvas = tk.Canvas(self.root, width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
        self.canvas.pack()

        # Create the initial image with a dot
        self.image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.draw_dot()

        # Display the image on the canvas
        self.photo = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def draw_dot(self):
        dot_size = int(self.slider.get())
        x = IMAGE_WIDTH // 2
        y = IMAGE_HEIGHT // 2
        for dx, dy in get_brush_deltas(dot_size):
            self.draw.point((x + dx, y + dy), fill="red")
        # self.draw.ellipse((x - dot_size, y - dot_size, x + dot_size, y + dot_size), fill="red")

    def update_dot_size(self, value):
        self.draw.rectangle((0, 0, IMAGE_WIDTH, IMAGE_HEIGHT), fill="white")
        self.draw_dot()
        self.photo = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

# Create the main window
root = tk.Tk()

# Create an instance of the app
app = DotSizeApp(root)

# Start the Tkinter event loop
root.mainloop()
