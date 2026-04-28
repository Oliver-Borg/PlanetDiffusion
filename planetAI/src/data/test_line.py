import tkinter as tk
import math
from sketch_gen import get_line

class PointRotationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Point Rotation App")

        # Define the initial angle and line length
        self.angle = 0
        self.line_length = 100

        # Create the slider
        self.slider = tk.Scale(
            self.root,
            from_=0,
            to=360,
            resolution=1,
            length=200,
            orient=tk.HORIZONTAL,
            command=self.update_angle
        )
        self.slider.set(self.angle)
        self.slider.pack()

        # Create the canvas to display the line
        self.canvas = tk.Canvas(self.root, width=400, height=400)
        self.canvas.pack()

        # Draw the initial line
        self.draw_line()

    def update_angle(self, angle):
        self.angle = int(angle)
        self.draw_line()

    def draw_line(self):
        # Clear the canvas
        self.canvas.delete("all")

        # Calculate the coordinates of the rotated point
        x = self.line_length * math.cos(math.radians(self.angle))
        y = self.line_length * math.sin(math.radians(self.angle))
        self.canvas.create_rectangle(200 + int(x), 200 + int(y), 200 + int(x) + 1, 200 + int(y) + 1, fill="green")
        self.canvas.create_rectangle(200, 200, 201, 201, fill="green")
        # Draw the line
        points = get_line((200, 200), (200 + int(y), 200 + int(x)))
        for y, x in points:
            self.canvas.create_rectangle(x, y, x + 1, y + 1, fill="black")

if __name__ == '__main__':
    root = tk.Tk()
    app = PointRotationApp(root)
    root.mainloop()