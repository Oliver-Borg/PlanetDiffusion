from enum import Enum
import numpy as np
from typing import TYPE_CHECKING, Callable
import cv2
from line_profiler import profile
from torch import Tensor
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QComboBox
from PyQt5.QtCore import Qt, QPoint, QEvent
from PyQt5.QtGui import QImage, QPixmap, QCursor, QMouseEvent, QWheelEvent

from planetAI.src.data.utils import (
    tensor_to_np,
    np_rgb,
)
from planetAI.src.data.sphere_mapping import SphereMapping, QuadSphere
from planetAI.src.data.noise_settings import NoiseSettings
from planetAI.src.data.noise_funcs import stacked_multi_noise
from ..interface_types import AtlasType
from ..atlas_storage import ATLAS_STORAGE, atlas_display_func

if TYPE_CHECKING:
    from ..view import View


class GlobePanelMode(Enum):
    """
    Enum for the GlobePanelMode class.
    """
    SPHERE = "sphere"
    QUAD_SPHERE = "quad_sphere"
    NOISE = "noise"


class GlobePanel(QWidget):
    def __init__(
        self,
        parent=None,
        shape: tuple[int, int] = (4096, 8192),
        preview_size: tuple[int, int] = 512,
        view_api: "View" = None,
        background_colour: tuple[int, int, int] = [135, 206, 235],
        globe_mode: GlobePanelMode = GlobePanelMode.SPHERE,
        cmap: str | None = None,
        initial_zoom: int = 0,
        on_motion_callback: Callable[
            ["GlobePanel", bool], None
        ] = lambda globe_panel, final=False: None,
        post_process_function: Callable[[np.ndarray], np.ndarray] = lambda x: x,
        noise_format_func: Callable[[np.ndarray], np.ndarray] = lambda x: x,
        display_colour_hint: bool = False,
        initial_atlas_type: AtlasType = AtlasType.DEM_SKETCH,
        enable_atlas_type_selector: bool = False,
    ):
        super().__init__(parent)
        self.quad_sphere = QuadSphere(shape=shape)
        self.cursor_sphere = None
        fw = self.quad_sphere.face_width
        self.post_process_function = post_process_function
        self.noise_format_func = noise_format_func
        quad_atlas_shape = (fw, fw * 6)
        self.quad_sphere.quad_sphere_atlas = np.random.randint(
            0, 256, (quad_atlas_shape[0], quad_atlas_shape[1], 3), dtype=np.uint8
        )

        self.sphere_mapping = SphereMapping(
            atlas=np.zeros((shape[0], shape[1], 3), dtype=np.uint8),
            method="surface-straight",
            discrete=True,
        )
        self.shape = shape
        self.preview_size = preview_size
        self.preview = np.zeros((self.preview_size, self.preview_size), dtype=np.uint8)
        self.cursor_tile: np.ndarray = np.zeros((self.preview_size, self.preview_size), dtype=bool)
        self.zoom = initial_zoom
        self.sample_factor = 1
        self.zoom_offset = 0
        self.initial_zoom = initial_zoom
        self.zoom_to_mouse = False
        self.min_zoom = -5
        self.max_zoom = 5
        self.latitude = 0
        self.longitude = 0
        self.last_x = 0
        self.last_y = 0
        self.moving = False
        self.has_moved = False
        self.uv = None
        self.atlas_indices = None
        self.globe_mode = globe_mode
        self.surface_coords = None
        self.view_api = view_api
        self.background_colour = background_colour
        self.cmap = cmap
        self.max_pixel_value = 175  # This seems to be the max value we can generate so use this for scaling
        self.on_motion_callback = on_motion_callback
        self.noise_settings = []
        self.display_colour_hint = display_colour_hint
        self.atlas_type = initial_atlas_type
        self.enable_atlas_type_selector = enable_atlas_type_selector
        self.mouse_cursor = ""
        self.create_widgets()

    def create_widgets(self):
        """
        Create the widgets for the GlobePanel class using PyQt5.
        """
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.preview_panel = QLabel()
        self.preview_panel.setFixedSize(self.preview_size, self.preview_size)
        layout.addWidget(self.preview_panel)

        if self.enable_atlas_type_selector:
            self.atlas_selector = QComboBox()
            self.atlas_selector.addItems([t.value for t in AtlasType])
            self.atlas_selector.setCurrentText(self.atlas_type.value)
            self.atlas_selector.currentTextChanged.connect(self.set_atlas_type)
            layout.addWidget(self.atlas_selector)

        if self.display_colour_hint:
            self.colour_hint = QLabel("")
            layout.addWidget(self.colour_hint)

        # Set up event handling
        self.preview_panel.setMouseTracking(True)
        self.preview_panel.wheelEvent = self.scroll
        self.preview_panel.mousePressEvent = self.mouse_press_event
        self.preview_panel.mouseMoveEvent = self.mouse_move_event
        self.preview_panel.mouseReleaseEvent = self.mouse_release_event
        self.preview_panel.leaveEvent = self.leave_event

        self.display_preview()

    def set_atlas_type(self, atlas_type: AtlasType | str) -> None:
        new_atlas_type = (
            atlas_type if isinstance(atlas_type, AtlasType) else AtlasType(atlas_type)
        )
        if self.atlas_type == new_atlas_type:
            return
        new_atlas = ATLAS_STORAGE.get(new_atlas_type)
        if new_atlas is None:
            # TODO Do something better here
            print(new_atlas_type, "not found")
            return
        self.atlas_type = new_atlas_type
        self.set_atlas(new_atlas)

    def display_pixel_color(self, event):
        """
        Display the pixel colour for the GlobePanel class.
        """
        x, y = event.pos().x(), event.pos().y()
        try:
            image = self.preview_panel.pixmap().toImage()
            pixel_colour = image.pixelColor(x, y).getRgb()[:3]
            self.colour_hint.setText(f"Pixel colour: {pixel_colour}")
        except Exception:
            self.colour_hint.setText("")

    def leave_event(self, event: QEvent):
        """Handle leave events"""
        if self.view_api is not None:
            self.view_api.leave_event()
        self.display_preview_with_cursor()

    def mouse_press_event(self, event: QMouseEvent):
        """Handle mouse press events"""
        if event.button() == Qt.MiddleButton:
            self.middle_click_down(event)
        elif event.button() == Qt.LeftButton and self.view_api is not None:
            self.draw_event(event)

    @profile
    def mouse_move_event(self, event: QMouseEvent):
        """Handle mouse move events"""
        if event.buttons() & Qt.MiddleButton:
            self.motion(event)
        elif event.buttons() & Qt.LeftButton and self.view_api is not None:
            self.draw_event(event)
        elif self.view_api is not None:
            self.motion_event(event)
        if self.display_colour_hint:
            self.display_pixel_color(event)

    def mouse_release_event(self, event: QMouseEvent):
        """Handle mouse release events"""
        if event.button() == Qt.MiddleButton:
            self.middle_click_up(event)
        elif event.button() == Qt.LeftButton and self.view_api is not None:
            self.finish_draw(event)

    @profile
    def motion_event(self, event: QMouseEvent):
        """
        Draw the motion for the GlobePanel class.
        """
        if self.view_api is not None:
            self.view_api.motion_event(self.convert_sphere_event(event))
        self.display_preview_with_cursor()

    def draw_event(self, event: QMouseEvent):
        """
        Draw the event for the GlobePanel class.
        """
        if self.view_api is not None:
            self.view_api.draw_event(self.convert_sphere_event(event))
        self.display_preview_with_cursor()

    def finish_draw(self, event: QMouseEvent):
        """
        Finish the draw for the GlobePanel class.
        """
        if self.view_api is not None:
            self.view_api.finish_draw(self.convert_sphere_event(event))

    def scroll(self, event: QWheelEvent):
        """Convert Qt wheel event to zoom"""
        delta = event.angleDelta().y()
        if delta > 0:
            if self.zoom_to_mouse:
                pos = event.pos()
                zoom_lat, zoom_long = self.sphere_mapping.click_coords(
                    (pos.x(), pos.y()),
                    (self.latitude, self.longitude),
                    self.preview_size,
                    2**self.zoom,
                    self.surface_coords,
                )
                self.set_coords(zoom_lat, zoom_long)
            self.set_zoom(self.zoom + 1)
        else:
            self.set_zoom(self.zoom - 1)
        if self.view_api is not None:
            self.view_api.motion_event(self.convert_sphere_event(event))
        self.on_motion_callback(self, True)

    def set_zoom(self, zoom: int):
        """
        Set the zoom for the GlobePanel class.
        """
        self.zoom = min(self.max_zoom, max(self.min_zoom, zoom))
        self.has_moved = True
        self.display_preview()

    def middle_click_down(self, event: QMouseEvent):
        """
        Middle click down for the GlobePanel class.
        """
        self.last_x = event.pos().x()
        self.last_y = event.pos().y()
        self.moving = True
        self.has_moved = True
        self.sample_factor = 4
        self.zoom_offset = -2
        self.has_moved = True
        self.display_preview()
        self.on_motion_callback(self)

    @profile
    def motion(self, event: QMouseEvent):
        """
        Motion for the GlobePanel class.
        """
        lat_sensitivity = 0.25 * 1.1 ** (-self.zoom)
        long_sensitivity = 0.25 * 1.1 ** (-self.zoom)
        if self.moving:
            self.longitude -= (event.pos().x() - self.last_x) * long_sensitivity
            self.latitude += (event.pos().y() - self.last_y) * lat_sensitivity
            self.latitude = min(90, max(-90, self.latitude))
            self.longitude = self.longitude % 360
            self.last_x = event.pos().x()
            self.last_y = event.pos().y()
            self.has_moved = True
            self.display_preview()
            self.on_motion_callback(self)

    def set_coords(self, latitude: float, longitude: float):
        """
        Set the coordinates for the GlobePanel class.
        """
        self.latitude = latitude
        self.longitude = longitude
        self.has_moved = True
        self.display_preview()

    def middle_click_up(self, event: QMouseEvent):
        """
        Middle click up for the GlobePanel class.
        """
        self.last_x = 0
        self.last_y = 0
        self.moving = False
        self.has_moved = True
        self.sample_factor = 1
        self.zoom_offset = 0
        self.has_moved = True
        if self.view_api is not None:
            self.view_api.motion_event(self.convert_sphere_event(event))
        self.display_preview()
        self.on_motion_callback(self, True)

    def convert_sphere_event(self, event: QMouseEvent) -> QPoint:
        """
        Convert the click event for the GlobePanel class.
        """
        click_coords = (event.pos().x(), event.pos().y())
        click_lat, click_long = self.sphere_mapping.click_coords(
            click_coords,
            (self.latitude, self.longitude),
            self.preview_size,
            2**self.zoom,
            self.surface_coords,
        )
        if bool(np.isnan(click_lat)) or bool(np.isnan(click_long)):
            return QPoint(-1, -1)
        if self.globe_mode == GlobePanelMode.QUAD_SPHERE:
            atlas_y, atlas_x = self.quad_sphere.get_quad_tile_mapping(
                (click_lat, click_long), 1
            )
        else:  # For noise just use sphere mode too
            atlas_y, atlas_x = self.sphere_mapping.get_tile_mapping(
                (click_lat, click_long), 1, 2**self.zoom
            )
        return QPoint(int(atlas_x), int(atlas_y))

    def atlas_changed(self, atlas_types: list[AtlasType]) -> None:
        if self.atlas_type in atlas_types:
            self.set_atlas(ATLAS_STORAGE.get(self.atlas_type))

    def set_cursor_tile(self, new_cursor_tile: np.ndarray):
        self.cursor_tile = new_cursor_tile
        self.display_preview_with_cursor()

    def set_preview_mouse_cursor(self, cursor: str):
        cursor_map = {
            "": Qt.ArrowCursor,
            "fleur": Qt.SizeAllCursor,
        }
        if cursor != self.mouse_cursor:
            self.preview_panel.setCursor(QCursor(cursor_map.get(cursor, Qt.ArrowCursor)))
            self.mouse_cursor = cursor

    def convert_np_to_qpixmap(self, img: np.ndarray) -> QPixmap:
        """Convert numpy array to QPixmap"""
        height, width = img.shape[:2]
        if len(img.shape) == 2:
            c = 1
        else:
            c = img.shape[2]
        bytes_per_line = c * width
        qimg = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)

    @profile
    def display_preview_with_cursor(self):
        preview = self.preview.copy()
        if not self.moving:
            self.set_preview_mouse_cursor("")
            ys, xs = np.where(self.cursor_tile)
            if self.cursor_tile.shape == preview.shape[:2] and len(ys) > 0:
                intensity: np.ndarray = preview[ys, xs].mean(axis=1)
                cursor_color = (intensity < 128).astype(np.uint8) * 255
                preview[ys, xs, 0] = cursor_color
                preview[ys, xs, 1] = cursor_color
                preview[ys, xs, 2] = cursor_color
        else:
            self.set_preview_mouse_cursor("fleur")

        pixmap = self.convert_np_to_qpixmap(preview.astype(np.uint8))
        self.preview_panel.setPixmap(pixmap)

    @profile
    def display_preview(self):
        """
        Display the preview for the GlobePanel class.
        """
        sample_size = self.preview_size // self.sample_factor
        preview = np.zeros((sample_size, sample_size, 3), dtype=np.uint8)
        nan_mask = np.ones((sample_size, sample_size), dtype=bool)

        recalculation_needed = (
            self.has_moved
            or (self.globe_mode == GlobePanelMode.QUAD_SPHERE and self.uv is None)
            or (self.globe_mode == GlobePanelMode.SPHERE and self.atlas_indices is None)
            or self.surface_coords is None
        )

        if recalculation_needed:
            self.surface_coords = self.sphere_mapping.get_surface_coords(
                (self.latitude, self.longitude),
                sample_size,
                2 ** (self.zoom + self.zoom_offset),
            )
            self.has_moved = False
        if recalculation_needed:
            if self.globe_mode == GlobePanelMode.QUAD_SPHERE:
                self.uv = self.quad_sphere.surface_coords_to_quad_coords(
                    *self.surface_coords, preserve_nan=False, round=True
                )
            elif self.globe_mode == GlobePanelMode.SPHERE:
                self.atlas_indices = self.sphere_mapping.get_tile_mapping(
                    (self.latitude, self.longitude),
                    tile_size=sample_size,
                    zoom=2 ** (self.zoom + self.zoom_offset),
                    round_result=False,
                    surface_coords=self.surface_coords,
                )

        nan_mask = np.isnan(self.surface_coords[0])

        if self.globe_mode == GlobePanelMode.QUAD_SPHERE:
            preview = self.quad_sphere.get_quad_tile(
                (self.latitude, self.longitude),
                tile_size=sample_size,
                zoom=2 ** (self.zoom + self.zoom_offset),
                atlas_indices=self.uv,
                discrete=True,
            )
        elif self.globe_mode == GlobePanelMode.SPHERE:
            preview = self.sphere_mapping.get_tile(
                (self.latitude, self.longitude),
                tile_size=sample_size,
                zoom=2 ** (self.zoom + self.zoom_offset),
                atlas_indices=self.atlas_indices,
            )
        elif self.globe_mode == GlobePanelMode.NOISE:
            total_noise = (
                (stacked_multi_noise(self.surface_coords, self.noise_settings) * 255)
                .clip(0, 255)
                .astype(np.uint8)
            )
            total_noise = self.noise_format_func(total_noise)
            if len(total_noise.shape) == 2:
                total_noise = np.dstack([total_noise] * 3)
            preview = total_noise
        if isinstance(preview, Tensor):
            preview = tensor_to_np(preview)
        if self.cmap is not None:
            if len(preview.shape) == 3:
                preview = preview[:, :, 0]
            max_pixel = (
                self.max_pixel_value
                if self.max_pixel_value > preview.max()
                else preview.max()
            )
            preview = np_rgb(preview, self.cmap, max_pixel=max_pixel)
        preview = atlas_display_func(self.atlas_type)(preview)
        preview = self.post_process(preview)
        if len(preview.shape) == 2:
            preview = np.repeat(preview[:, :, np.newaxis], 3, axis=2)

        if nan_mask.shape == preview.shape[:2]:
            preview[:, :, 0][nan_mask] = self.background_colour[0]
            preview[:, :, 1][nan_mask] = self.background_colour[1]
            preview[:, :, 2][nan_mask] = self.background_colour[2]
        if self.sample_factor > 1:
            preview = cv2.resize(
                preview, (self.preview_size, self.preview_size), interpolation=cv2.INTER_NEAREST
            )
        self.preview = preview
        self.display_preview_with_cursor()

    def post_process(self, preview: np.ndarray):
        """
        Postprocessing for the GlobePanel class.
        You can either override this method or pass a function to the constructor.
        """
        preview = self.post_process_function(preview)
        return preview

    def set_quad_atlas(self, atlas: np.ndarray):
        """
        Set the quad atlas for the GlobePanel class.
        """
        self.quad_sphere.set_quad_atlas(atlas)
        self.globe_mode = GlobePanelMode.QUAD_SPHERE
        self.display_preview()

    def set_normal_atlas(self, atlas: np.ndarray):
        """
        Set the normal atlas for the GlobePanel class.
        """
        self.sphere_mapping.set_atlas(atlas)
        self.globe_mode = GlobePanelMode.SPHERE
        self.display_preview()

    def set_atlas(self, atlas: np.ndarray):
        h, w = atlas.shape[:2]
        if atlas.shape[:2] != self.shape:
            self.has_moved = True
            self.shape = (h, w)
        if 2 * h == w:
            self.set_normal_atlas(atlas)
        else:
            self.set_quad_atlas(atlas)

    def set_noise_settings(self, noise_settings: list[NoiseSettings]):
        """
        Set the noise settings for the GlobePanel class.
        """
        self.noise_settings = noise_settings
        self.globe_mode = GlobePanelMode.NOISE
        self.display_preview()


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication

    app = QApplication([])
    globe_panel = GlobePanel()
    globe_panel.show()
    app.exec_()
