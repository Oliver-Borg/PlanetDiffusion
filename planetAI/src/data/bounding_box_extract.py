from typing import Generator

import numpy as np
import cv2
from scipy.ndimage import label
from PIL import Image


class Bounds:
    x0: int
    y0: int
    x1: int
    y1: int
    component: int

    def __init__(self, x0: int, y0: int, x1: int, y1: int, component: int):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.component = component

    def __call__(self):
        return slice(self.y0, self.y1), slice(self.x0, self.x1)
    
    @property
    def tuple(self):
        return self.x0, self.y0, self.x1, self.y1
    
    @property
    def width(self):
        return self.x1 - self.x0
    
    @property
    def height(self):
        return self.y1 - self.y0
    
    @property
    def area(self):
        return self.width * self.height
    
    def contains(self, other: "Bounds") -> bool:
        return self.x0 <= other.x0 and self.y0 <= other.y0 and self.x1 >= other.x1 and self.y1 >= other.y1
    

class BoundList:
    def __init__(self, bounds: list[Bounds], labeled: np.ndarray, ncomponents: int):
        self.bounds = bounds
        self.labeled = labeled
        self.ncomponents = ncomponents

    def collapse(self):
        processed = []
        for i in range(len(self.bounds)):
            for j in range(len(self.bounds)):
                if i == j or j in processed:
                    continue
                bounds_i = self.bounds[i]
                bounds_j = self.bounds[j]
                if bounds_i.contains(bounds_j):
                    self.labeled[self.labeled == bounds_j.component] = bounds_i.component
                    processed.append(j)
                    self.ncomponents -= 1
        self.bounds = [b for i, b in enumerate(self.bounds) if i not in processed]

                

    @property
    def tuple(self):
        return self.bounds, self.labeled, self.ncomponents


def extract_boxes(atlas: np.ndarray) -> BoundList:
    """ Extract bounding boxes from an atlas image"""
    land = (atlas > 0).astype(np.uint8) * 255
    land = cv2.dilate(land, np.ones((5, 5), np.uint8), iterations=1)

    labeled, ncomponents = label(land)
    boxes = []
    for i in range(1, ncomponents + 1):
        component = (labeled == i).astype(np.uint8)
        y, x = np.where(component)
        box = Bounds(x.min(), y.min(), x.max()+1, y.max()+1, i)
        boxes.append(box)
    bound_list = BoundList(boxes, labeled, ncomponents)
    bound_list.collapse()
    return bound_list


def sketch_iterator(atlas: np.ndarray) -> Generator[tuple[np.ndarray, Bounds], None, None]:
    """ Iterate over the sketches in an atlas image"""
    boxes, labeled, ncomponents = extract_boxes(atlas).tuple
    for box in boxes:
        component = labeled[box()].copy()
        component = (component == box.component).astype(np.uint8)
        atlas_piece = atlas[box()].copy()
        atlas_piece[component == 0] = 0
        if atlas_piece.sum() == 0:
            continue
        yield atlas_piece, box


if __name__ == "__main__":
    atlas = np.array(Image.open("./planetAI/data/downsketch.png"))
    # atlas = (atlas > 0).astype(np.uint8) * 255
    boxes, labeled, ncomponents = extract_boxes(atlas).tuple
    coloured = np.zeros((atlas.shape[0], atlas.shape[1], 3), np.uint8)
    for box in boxes:
        coloured[labeled == box.component] = np.random.randint(0, 255, 3)
    coloured[atlas == 0] = (0, 0, 0)
    for box in boxes:
        x, y, X, Y = box.tuple
        coloured = cv2.rectangle(coloured, (x, y), (X, Y), (255, 255, 255), 1)
    Image.fromarray(coloured).show()

    for sketch, box in sketch_iterator(atlas):
        Image.fromarray(sketch).show()
        print(box.tuple)