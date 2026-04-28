from dataclasses import replace

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QRadioButton,
    QPushButton,
    QCheckBox,
    QProgressBar,
    QButtonGroup,
    QTabWidget,
)
from PyQt5.QtCore import QPoint
import numpy as np
from typing import Callable
import sys

from planetAI.src.data.modal_sketch import ModalSketch
from planetAI.src.data.river_modal_sketch import RiverModalSketch
from planetAI.src.data.utils import (
    PlanetConfig,
    profile,
    rgb_to_hex_str,
)
from planetAI.src.data.landcover_utils import (
    LandcoverName,
    landcover_class_list,
    landcover_class_colours,
    landcover_mapping,
)
from planetAI.src.data.dataset import EncoderOverride
from planetAI.src.data.noise_settings import NoiseSettings
from planetAI.src.data.sphere_mapping import SphereMapping

from .interface_types import LandcoverClass, EventMetadata, AtlasType
from .panels.DEMPanel import DEMPanel
from .panels.DEMSketchPanel import DEMSketchPanel
from .panels.TempPanel import TempPanel
from .panels.LandcoverPanel import LandcoverPanel
from .panels.GlobePanel import GlobePanel, GlobePanelMode
from .panels.RiverPanel import RiverPanel
from .panels.DataClassPanel import DataClassPanel
from .panels.EncoderOverridePanel import EncoderOverridePanel
from .panels.RegenPanel import RegenPanel

from .controller import Controller
from ..models.diffusion.inpaint_inference import InferenceArguments
from ..core.dataclass_argparser import CustomArgumentParser


class View(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.preview_size = 512
        self.encoder_override = EncoderOverride()
        parser = CustomArgumentParser(
            (InferenceArguments, PlanetConfig),
            description="Run inference on a diffusion model",
        )

        args: tuple[InferenceArguments, PlanetConfig] = (
            parser.parse_args_into_dataclasses()
        )
        self.inference_args, self.planet_cfg = args
        self.output_shape = (self.planet_cfg.H, self.planet_cfg.W)
        self.w = self.planet_cfg.w
        self.h = self.planet_cfg.h
        self.modal_sketch = ModalSketch(self.planet_cfg)
        self.river_modal_sketch = RiverModalSketch(replace(self.planet_cfg, size=5))
        self.controller = Controller(
            self,
            (self.h, self.w),
            self.output_shape,
            self.modal_sketch,
            self.river_modal_sketch,
            self.inference_args,
            self.planet_cfg,
            self.encoder_override,
        )
        self.landcover_labels = landcover_class_list()
        self.subclass_labels = self.planet_cfg.temp_labels()
        self.landcover_display_colours = landcover_class_colours()
        self.landcover_colours = self.planet_cfg.landcover_colour_list()
        self.subclass_colours = self.planet_cfg.temp_colour_list()

        self.landcover_classes = []

        for label in self.landcover_labels:
            subclasses = []
            lc = landcover_mapping[LandcoverName(label)]
            for j, subclass_label in enumerate(self.subclass_labels):
                subclass_display = self.modal_sketch.get_colour(
                    self.landcover_colours[lc.index], self.subclass_colours[j]
                )
                subclass = LandcoverClass(
                    subclass_label,
                    self.subclass_colours[j],
                    rgb_to_hex_str(subclass_display),
                    [],
                )
                subclasses.append(subclass)

            landcover_class = LandcoverClass(
                label,
                lc.gray_colour,
                rgb_to_hex_str(lc.display_colour),
                subclasses[1:],
            )  # Ignore the first one that is the same as the parent class
            self.landcover_classes.append(landcover_class)

        self.create_widgets()
        self.controller.startup()

    def display_controls(self):
        """Display the correct control panel based on the image type selected"""
        image_type = self.image_type_group.checkedButton().property("type")

        # Switch to appropriate tab based on image type
        if image_type == "dem":
            self.tab_widget.setCurrentWidget(self.dem_panel)
        elif image_type == "dem_sketch":
            self.tab_widget.setCurrentWidget(self.dem_sketch_panel)
        elif image_type == "landcover":
            self.tab_widget.setCurrentWidget(self.landcover_panel)
        elif image_type == "temperature":
            self.tab_widget.setCurrentWidget(self.temp_panel)
        elif image_type == "river":
            self.tab_widget.setCurrentWidget(self.river_panel)
        elif image_type == "regen":
            self.tab_widget.setCurrentWidget(self.regen_panel)
        else:
            raise ValueError(f"Unknown image type {image_type}")

    def get_noise_format_func(self):
        image_type = self.image_type_group.checkedButton().property("type")
        if image_type == "dem":
            return self.dem_panel.noise_format_func()
        elif image_type == "dem_sketch":
            return self.dem_sketch_panel.noise_format_func()
        elif image_type == "landcover":
            return self.landcover_panel.noise_format_func()
        elif image_type == "temperature":
            return self.temp_panel.noise_format_func()
        elif image_type == "river":
            return self.river_panel.noise_format_func()
        elif image_type == "regen":
            return self.regen_panel.noise_format_func()
        else:
            raise ValueError(f"Unknown image type {image_type}")

    def select_type(self):
        self.controller.select_image(
            self.get_event_metadata(EventMetadata("_", None, None, 0)),
            self.display_controls,
            self.get_noise_format_func()
        )

    def set_progress(self, progress_pcnt: float | None):
        # progress is a float between 0 and 100 or None if progress is indeterminate
        if progress_pcnt is None:
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setValue(0)
        else:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(int(progress_pcnt))
        self.progress_bar.setFormat(f"{progress_pcnt / 100:.0%}" if progress_pcnt is not None else "Loading...")

    def create_widgets(self):
        main_layout = QVBoxLayout()  # Changed to QVBoxLayout for vertical stacking
        self.setLayout(main_layout)

        # Top section with controls and displays
        top_section = QHBoxLayout()

        # Control frame (left side)
        control_layout = QVBoxLayout()

        # Radio button frame
        rb_layout = QVBoxLayout()
        self.image_type_group = QButtonGroup(self)

        radio_buttons = [
            ("DEM Sketch", "dem_sketch"),
            ("DEM Generation", "dem"),
            ("Landcover", "landcover"),
            ("Temperature", "temperature"),
            ("Rivers", "river"),
            ("Regenerate", "regen"),
        ]

        for text, value in radio_buttons:
            radio = QRadioButton(text)
            radio.setProperty("type", value)
            self.image_type_group.addButton(radio)
            rb_layout.addWidget(radio)
            if value == "dem_sketch":
                radio.setChecked(True)

        self.image_type_group.buttonClicked.connect(self.select_type)

        # Buttons
        self.save_button = QPushButton("Save")
        self.load_button = QPushButton("Load")
        self.generate_button = QPushButton("Generate")
        self.stop_generate_button = QPushButton("Stop")

        self.save_button.clicked.connect(self.controller.save_images)
        self.load_button.clicked.connect(self.controller.load_images)
        self.generate_button.clicked.connect(
            lambda: self.controller.start_generation(
                self.two_phase_var.isChecked(),
                self.use_rivers_var.isChecked(),
                self.do_upscaling_var.isChecked(),
            )
        )
        self.stop_generate_button.clicked.connect(self.controller.stop_generation)

        # Checkboxes
        self.use_rivers_var = QCheckBox("Use Rivers")
        self.use_rivers_var.setChecked(True)
        self.two_phase_var = QCheckBox("Two Phase")
        self.two_phase_var.setChecked(False)
        self.do_upscaling_var = QCheckBox("Do Upscaling")
        self.do_upscaling_var.setChecked(False)
        self.sync_globe_var = QCheckBox("Sync Globes")
        self.sync_globe_var.setChecked(True)
        self.noise_align_var = QCheckBox("Align Noise")
        self.noise_align_var.setChecked(self.controller.align_noise)
        self.noise_align_var.stateChanged.connect(
            lambda: self.controller.set_align_noise(self.noise_align_var.isChecked())
        )

        # Add widgets to control layout
        control_layout.addLayout(rb_layout)
        for widget in [
            self.save_button,
            self.load_button,
            self.generate_button,
            self.stop_generate_button,
            self.use_rivers_var,
            self.two_phase_var,
            self.do_upscaling_var,
            self.sync_globe_var,
            self.noise_align_var,
        ]:
            control_layout.addWidget(widget)

        # Display area layout
        display_layout = QVBoxLayout()
        preview_layout = QHBoxLayout()

        # Globe panels
        self.input_display = GlobePanel(
            self,
            preview_size=self.preview_size,
            shape=(self.h, self.w),
            view_api=self,
            globe_mode=GlobePanelMode.SPHERE,
            on_motion_callback=self.globe_panel_motion_event,
        )

        self.output_sat_display = GlobePanel(
            self,
            preview_size=self.preview_size,
            shape=self.output_shape,
            globe_mode=GlobePanelMode.QUAD_SPHERE,
            on_motion_callback=self.globe_panel_motion_event,
            initial_atlas_type=AtlasType.SAT,
            enable_atlas_type_selector=True,
        )

        self.output_dem_display = GlobePanel(
            self,
            preview_size=self.preview_size,
            shape=self.output_shape,
            globe_mode=GlobePanelMode.QUAD_SPHERE,
            on_motion_callback=self.globe_panel_motion_event,
            initial_atlas_type=AtlasType.DEM,
            enable_atlas_type_selector=True,
        )
        self.globe_panel_list = [
            self.input_display,
            self.output_sat_display,
            self.output_dem_display,
        ]

        preview_layout.addWidget(self.input_display)
        preview_layout.addWidget(self.output_sat_display)
        preview_layout.addWidget(self.output_dem_display)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)

        display_layout.addLayout(preview_layout)
        display_layout.addWidget(self.progress_bar)

        # Create tab widget for panels
        self.tab_widget = QTabWidget()

        # Create panels
        self.inference_panel = DataClassPanel(
            self, self.inference_args, "Inference Arguments"
        )
        self.encoder_override_panel = EncoderOverridePanel(self, self.encoder_override)

        # Create a new tab widget for these panels
        self.inference_tab_widget = QTabWidget()
        self.inference_tab_widget.addTab(self.inference_panel, "Inference")
        self.inference_tab_widget.addTab(self.encoder_override_panel, "Encoder")

        self.dem_panel = DEMPanel(
            self,
            import_callback=self.controller.import_dem,
            shape=(self.h, self.w),
            view_api=self,
            on_control_change=self.controller.process_event,
            reset_callback=self.controller.reset_sketch,
            derive_callback=self.controller.derive_dem,
        )
        self.dem_sketch_panel = DEMSketchPanel(
            self,
            shape=(self.h, self.w),
            view_api=self,
            on_control_change=self.controller.process_event,
            reset_callback=self.controller.reset_sketch,
        )
        self.temp_panel = TempPanel(
            self,
            shape=(self.h, self.w),
            on_control_change=self.controller.process_event,
            reset_callback=self.controller.reset_sketch,
        )
        self.landcover_panel = LandcoverPanel(
            self,
            landcover_classes=self.landcover_classes,
            shape=(self.h, self.w),
            view_api=self,
            on_control_change=self.controller.process_event,
            reset_callback=self.controller.reset_sketch,
            import_callback=self.controller.import_landcover,
            derive_callback=self.controller.derive_landcover,
        )
        self.river_panel = RiverPanel(
            self,
            derive_callback=self.controller.derive_rivers,
            shape=self.output_shape,
            view_api=self,
            planet_config=self.planet_cfg,
            on_control_change=self.controller.process_event,
            reset_callback=self.controller.reset_sketch,
            on_precip_change=self.controller.sketch_precip,
        )
        self.regen_panel = RegenPanel(
            self,
            shape=self.output_shape,
            view_api=self,
            planet_config=self.planet_cfg,
            on_control_change=self.controller.process_event,
            reset_callback=self.controller.reset_sketch,
        )

        # Add panels to tabs
        self.tab_widget.addTab(self.dem_sketch_panel, "DEM Sketch")
        self.tab_widget.addTab(self.dem_panel, "DEM Generation")
        self.tab_widget.addTab(self.landcover_panel, "Landcover")
        self.tab_widget.addTab(self.temp_panel, "Temperature")
        self.tab_widget.addTab(self.river_panel, "Rivers")
        self.tab_widget.addTab(self.regen_panel, "Regenerate")

        # Main layout assembly
        top_section.addLayout(control_layout)
        top_section.addLayout(display_layout)

        right_panel_layout = QVBoxLayout()
        right_panel_layout.addWidget(self.inference_tab_widget)
        top_section.addLayout(right_panel_layout)

        main_layout.addLayout(top_section)
        main_layout.addWidget(self.tab_widget)

        # Connect tab changes to radio buttons
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

        self.controller.create_temperature_preset(
            self.temp_panel.get_event_metadata(None, None)
        )
        self.display_controls()
        self.select_type()

    def _on_tab_changed(self, index):
        """Sync radio button selection with tab changes"""
        tab_to_type = {
            0: "dem_sketch",
            1: "dem",
            2: "landcover",
            3: "temperature",
            4: "river",
            5: "regen",
        }
        for button in self.image_type_group.buttons():
            if button.property("type") == tab_to_type[index]:
                button.setChecked(True)
                self.select_type()
                break

    def get_event_metadata(self, event: QPoint) -> EventMetadata:
        try:
            x = event.x()
            y = event.y()
        except Exception:
            x = event.x
            y = event.y
        image_type = self.image_type_group.checkedButton().property("type")
        if image_type == "dem":
            metadata = self.dem_panel.get_event_metadata(x, y)
        elif image_type == "dem_sketch":
            metadata = self.dem_sketch_panel.get_event_metadata(x, y)
        elif image_type == "landcover":
            metadata = self.landcover_panel.get_event_metadata(x, y)
        elif image_type == "temperature":
            metadata = self.temp_panel.get_event_metadata(x, y)
        elif image_type == "river":
            metadata = self.river_panel.get_event_metadata(x, y)
        elif image_type == "regen":
            metadata = self.regen_panel.get_event_metadata(x, y)
        else:
            return EventMetadata(image_type, x, y, 0)
        return metadata

    @profile
    def update_globes(self, atlas_types: list[AtlasType]):
        self.output_dem_display.atlas_changed(atlas_types)
        self.output_sat_display.atlas_changed(atlas_types)

    def motion_event(self, event: QPoint):
        metadata = self.get_event_metadata(event)
        self.controller.process_cursor_event(metadata)

    def leave_event(self):
        self.controller.process_cursor_event(EventMetadata("_", None, None, 0))

    @profile
    def cursor_mask_update(self, new_cursor_mask: np.ndarray):
        self.cursor_sphere = SphereMapping(
            new_cursor_mask, method="surface-straight", discrete=True
        )
        sample_size = self.preview_size // self.input_display.sample_factor
        cursor_tile = np.zeros((sample_size, sample_size), dtype=bool)

        surface_coords = self.cursor_sphere.get_surface_coords(
            (self.input_display.latitude, self.input_display.longitude),
            tile_size=sample_size,
            zoom=2 ** (self.input_display.zoom + self.input_display.zoom_offset),
        )

        atlas_indices = self.cursor_sphere.get_tile_mapping(
            (self.input_display.latitude, self.input_display.longitude),
            tile_size=sample_size,
            zoom=2 ** (self.input_display.zoom + self.input_display.zoom_offset),
            round_result=True,
            surface_coords=surface_coords,
        )
        cursor_tile: np.ndarray = self.cursor_sphere.get_tile(
            (self.input_display.latitude, self.input_display.longitude),
            tile_size=sample_size,
            zoom=2 ** (self.input_display.zoom + self.input_display.zoom_offset),
            discrete=True,
            atlas_indices=atlas_indices,
        )
        nan_mask = np.isnan(surface_coords[0])
        cursor_tile = (cursor_tile > 0) & (~nan_mask)
        if self.sync_globe_var.isChecked():
            for panel in self.globe_panel_list:
                panel.set_cursor_tile(cursor_tile)
        elif self.input_display is not None:
            self.input_display.set_cursor_tile(cursor_tile)

    def draw_event(self, event: QPoint):
        metadata = self.get_event_metadata(event)
        self.controller.process_event(metadata)
        self.controller.process_cursor_event(metadata)

    def finish_draw(self, event: QPoint):
        metadata = self.get_event_metadata(event)
        self.controller.finish_draw(metadata)
        self.controller.process_cursor_event(metadata)

    @profile
    def set_image(self, image: np.ndarray):
        self.input_display.set_atlas(image)

    def set_input_post_process(
        self, post_process_func: Callable[[np.ndarray], np.ndarray]
    ):
        self.input_display.post_process_function = post_process_func

    def set_output_sat_image(self, image: np.ndarray):
        self.output_sat_display.set_atlas(image)

    def set_output_dem_image(self, image: np.ndarray):
        self.output_dem_display.set_atlas(image)

    def set_river_image(self, image: np.ndarray):
        self.river_panel.set_atlas(image)

    @profile
    def set_noise_settings(self, noise_settings: NoiseSettings):
        self.controller.set_noise_settings(noise_settings)

    def toggle_noise_overlay(self):
        self.controller.toggle_noise_overlay()

    def globe_panel_motion_event(self, source: GlobePanel, final: bool = False):
        if final:
            self.controller.set_view_coords((source.latitude, source.longitude))
        if self.sync_globe_var.isChecked():
            desired_zoom = source.zoom - source.initial_zoom
            for panel in self.globe_panel_list:
                if panel != source:
                    panel.sample_factor = source.sample_factor
                    panel.zoom_offset = source.zoom_offset
                    panel.moving = source.moving
                    panel.set_coords(source.latitude, source.longitude)
                    panel.set_zoom(desired_zoom + panel.initial_zoom)
        return None

    def set_noise_format_func(self, func: Callable[[np.ndarray], np.ndarray]) -> None:
        self.controller.set_noise_format_func(func)

    def get_input_coords(self):
        return self.input_display.latitude, self.input_display.longitude

    def on_close(self):
        self.dem_panel.save_config_dict()
        self.dem_sketch_panel.save_config_dict()
        self.temp_panel.save_config_dict()
        self.landcover_panel.save_config_dict()
        self.controller.is_exiting = True
        self.controller.save_on_close()
        self.close()

    def closeEvent(self, event):
        self.on_close()
        super().closeEvent(event)


def main():
    from PyQt5.QtWidgets import QApplication
    import qdarktheme

    app = QApplication(sys.argv)
    app.setApplicationDisplayName("Planet Painter")
    # TODO Consider changing this to "auto"
    qdarktheme.setup_theme()
    view = View()
    view.show()
    view.showMaximized()
    app.exec_()


if __name__ == "__main__":
    main()
