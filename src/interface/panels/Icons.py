import cv2
import numpy as np


CHEVRON = [
    [
        (0.65, 0.2),
        (0.35, 0.5),
        (0.65, 0.8),
    ],
]

DOUBLE_CHEVRON = [
    [
        (0.55, 0.2),
        (0.25, 0.5),
        (0.55, 0.8),
    ],
    [
        (0.75, 0.2),
        (0.45, 0.5),
        (0.75, 0.8),
    ],
]

LINE = [
    [
        (0.5, 0.2),
        (0.5, 0.8),
    ]
]


def rasterize_icon(icon: list[list[tuple[float, float]]], size: int, thickness: float = 0.03, flipped: bool = False):
    icon_array = np.zeros((size, size), dtype=np.uint8)
    pixel_thickness = max(int(round(size * thickness)), 1)
    for line_segment in icon:
        assert len(line_segment) >= 2
        for p1, p2 in zip(line_segment[:-1], line_segment[1:]):
            y1, x1 = p1
            y2, x2 = p2
            if flipped:
                y1 = 1 - y1
                y2 = 1 - y2
            y1 = int(round(y1 * size))
            x1 = int(round(x1 * size))
            y2 = int(round(y2 * size))
            x2 = int(round(x2 * size))
            icon_array = cv2.line(icon_array, (x1, y1), (x2, y2), 255, pixel_thickness)
    return icon_array


def get_temperature_icon_overlay(
    temp_sketch: np.ndarray, size: int = 20, spacing: int = 20, add_outlines: bool = False
):
    h, w = temp_sketch.shape
    icon_overlay = np.zeros_like(temp_sketch)
    if size < 18:
        return icon_overlay
    interval = size + spacing
    icon_map = {
        0: None,
        1: rasterize_icon(DOUBLE_CHEVRON, size, flipped=True),
        2: rasterize_icon(CHEVRON, size, flipped=True),
        3: rasterize_icon(LINE, size),
        4: rasterize_icon(CHEVRON, size),
        5: rasterize_icon(DOUBLE_CHEVRON, size),
    }
    component_outlines = (cv2.dilate(temp_sketch, np.ones((3, 3))) - temp_sketch) > 0
    for y in range(0, h, interval):
        even = (y // interval) % 2 == 0
        for x in range(interval if even else 0, w, interval * 2):
            y_end = y + size
            x_end = x + size
            if y_end > h or x_end > w:
                continue
            tile = temp_sketch[y:y_end, x:x_end]
            colour = tile[size//2, size//2]
            if colour == 0:
                continue
            icon = icon_map[colour // 50 or 1]
            # colours = np.unique(tile[np.where(icon)])
            # if len(colours) != 1:
            #     continue
            icon_overlay[y:y_end, x:x_end] = icon

    if add_outlines:
        icon_overlay[component_outlines] = 255
    return icon_overlay


def get_circle_icon_overlay(temp_sketch: np.ndarray, size: int = 20, spacing: int = 20, add_outlines: bool = False):
    h, w = temp_sketch.shape
    icon_overlay = np.zeros_like(temp_sketch)
    normal_size = size
    for t in range(1, 6):
        size = int(round(t / 4 * normal_size))
        interval = spacing
        icon = cv2.circle(np.zeros((size, size), dtype=np.uint8), (size // 2, size // 2), size // 2, 255, cv2.FILLED)
        for y in range(0, h, interval):
            even = (y // interval) % 2 == 0
            for x in range(interval if even else 0, w, interval * 2):
                y_end = y + size
                x_end = x + size
                if y_end > h or x_end > w:
                    continue
                tile = temp_sketch[y:y_end, x:x_end]
                colour_mask = tile[size//2, size//2] // 50 == t
                icon_overlay[y:y_end, x:x_end][colour_mask] = icon[colour_mask]

    component_outlines = (cv2.dilate(temp_sketch, np.ones((3, 3))) - temp_sketch) > 0
    if add_outlines:
        icon_overlay[component_outlines] = 255
    return icon_overlay


def line_mask(shape: tuple[int, int], line_width: int, line_spacing: int, rotations: int):
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    rotations = rotations % 8
    if rotations % 2 == 1:
        line_width = int(round(np.sqrt(2) * line_width))
        line_spacing = int(round(np.sqrt(2) * line_spacing))
    for i in range(line_width):
        if rotations % 2 == 0:
            columns = np.arange(line_spacing + line_width // 2 - i, w, line_spacing)
            if rotations in [0, 4]:
                mask[:, columns] = 255
            else:
                mask[columns, :] = 255
        else:
            if rotations in [1, 5]:
                m = 1
            else:
                m = -1
            ys = np.arange(0, h)
            xs = np.arange(0, w)
            xs, ys = np.meshgrid(xs, ys)
            diagonals = (ys + m * xs) % line_spacing == i
            mask[diagonals] = 255

    return mask


def get_temperature_line_overlay(
    temp_sketch: np.ndarray, size: int = 1, spacing: int = 20, add_outlines: bool = False
):
    h, w = temp_sketch.shape
    icon_overlay = np.zeros_like(temp_sketch)
    component_outlines = (cv2.dilate(temp_sketch, np.ones((3, 3))) - temp_sketch) > 0
    rot_0 = line_mask((h, w), size, spacing + size, 0)
    rot_45 = line_mask((h, w), size, spacing + size, 1)
    rot_90 = line_mask((h, w), size, spacing + size, 2)
    rot_135 = line_mask((h, w), size, spacing + size, 3)
    for temp in range(1, 6):
        if temp == 1:
            lines = np.maximum(rot_0, rot_90)
        elif temp == 2:
            lines = rot_0
        elif temp == 3:
            lines = rot_45
        elif temp == 4:
            lines = rot_90
        elif temp == 5:
            lines = np.maximum(rot_45, rot_135)
        else:
            lines = rot_135
        temp_mask = temp_sketch // 50 == temp
        icon_overlay[temp_mask] = lines[temp_mask]

    if add_outlines:
        icon_overlay[component_outlines] = 255
    return icon_overlay


def full_connected_components(component_map: np.ndarray) -> tuple[int, np.ndarray]:
    unique_vals = np.unique(component_map)

    output = np.zeros_like(component_map, dtype=np.int32)
    current_label = 1

    for val in unique_vals:
        if val == 0:
            continue
        mask = (component_map == val).astype(np.uint8)
        num, labels = cv2.connectedComponents(mask)
        labels = np.where(labels > 0, labels + current_label - 1, 0)
        output += labels
        current_label += num - 1
    return current_label, output


def get_component_temperature_icon_overlay(
    temp_sketch: np.ndarray,
    landcover_sketch: np.ndarray,
    size: int = 20,
    spacing: int = 20,
    add_outlines: bool = False,
):
    h, w = temp_sketch.shape
    icon_overlay = np.zeros_like(temp_sketch)
    if size < 18:
        return icon_overlay
    component_map = temp_sketch // 50 + (landcover_sketch // 25) * 6
    icon_map = {
        0: None,
        1: rasterize_icon(DOUBLE_CHEVRON, size, flipped=True),
        2: rasterize_icon(CHEVRON, size, flipped=True),
        3: rasterize_icon(LINE, size),
        4: rasterize_icon(CHEVRON, size),
        5: rasterize_icon(DOUBLE_CHEVRON, size),
    }
    component_outlines = (cv2.dilate(temp_sketch, np.ones((3, 3))) - temp_sketch) > 0
    num_components, components = full_connected_components(component_map)
    for i in range(num_components):
        ys, xs = np.where(components == i)
        if len(ys) == 0 or len(xs) == 0:
            continue
        # y = (ys.max() + ys.min() - size) // 2
        # x = (xs.max() + xs.min() - size) // 2
        y = ys[len(ys) // 2] - size // 2
        x = xs[len(xs) // 2] - size // 2
        y_end = y + size
        x_end = x + size
        if y_end > h or x_end > w or x < 0 or y < 0:
            continue
        tile = temp_sketch[y:y_end, x:x_end]
        colour = tile[size//2, size//2]
        if colour == 0:
            continue
        icon = icon_map[colour // 50 or 1]
        # colours = np.unique(tile[np.where(icon)])
        # if len(colours) != 1:
        #     continue
        if icon_overlay[y:y_end, x:x_end].any():
            continue
        icon_overlay[y:y_end, x:x_end] = icon

    if add_outlines:
        icon_overlay[component_outlines] = 255
    return icon_overlay


def add_icons_to_temperature_sketch(temperature: np.ndarray):
    icon_overlay = get_temperature_icon_overlay(temperature) > 0
    to_return = temperature.copy()
    to_return[icon_overlay] = (to_return[icon_overlay] < 128).astype(np.uint8) * 255
    return to_return


def add_icons_to_modal_sketch(modal_sketch: np.ndarray, temperature: np.ndarray):
    icon_overlay = get_temperature_icon_overlay(temperature) > 0
    to_return = modal_sketch.copy()
    brightness = cv2.cvtColor(modal_sketch, cv2.COLOR_RGB2GRAY)
    to_return[icon_overlay] = np.dstack([(brightness[icon_overlay] < 128).astype(np.uint8) * 255] * 3)
    return to_return


if __name__ == "__main__":
    from PIL import Image

    all_icons = [CHEVRON, DOUBLE_CHEVRON, LINE]
    num_icons = len(all_icons)

    size = 200

    res_array = np.zeros((size * 2, num_icons * size), dtype=np.uint8)

    for i, icon in enumerate(all_icons):
        res_array[:size, i*size:(i + 1) * size] = rasterize_icon(icon, size)
        res_array[size:, i*size:(i + 1) * size] = rasterize_icon(icon, size, flipped=True)

    Image.fromarray(res_array).show()

    for rotations in range(0, 8):
        rotated_line_mask = line_mask((512, 512), 1, 20, rotations)
        Image.fromarray(rotated_line_mask).show()
        continue
