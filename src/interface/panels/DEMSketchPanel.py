import numpy as np
from typing import TYPE_CHECKING, Callable
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QCheckBox,
    QGroupBox,
)

from ..widgets.LabeledSlider import FloatLabeledSlider
from planetAI.src.data.noise_settings import EdgeDistanceSettings
from planetAI.src.data.utils import PlanetConfig, profile
from planetAI.src.data.noise_settings import NoiseSettings
from planetAI.src.data.sketch_gen import dilate_paint
from ..interface_types import (
    DEMEventMetadata,
    EventTypeEnum,
    DEMSketchEventMetadata,
    ToolEnum,
    EditTypeEnum,
)
from .SavablePanel import SaveablePanel
from .GlobePanel import GlobePanel, GlobePanelMode
from .SettingsPanel import SettingsPanel
from .ToolPanel import ToolPanel

if TYPE_CHECKING:
    from ..view import View


class DEMSketchPanel(QWidget, SaveablePanel):
    NAME = "dem_sketch_panel"

    def __init__(
        self,
        parent=None,
        shape: tuple[int, int] = (256, 512),
        view_api: "View" = None,
        on_control_change: Callable[[DEMEventMetadata], None] = lambda metadata: None,
        reset_callback: Callable[[DEMEventMetadata], None] = lambda metadata: None,
    ):
        """
        Constructor for the DEMPanel class.
        """
        QWidget.__init__(self, parent)

        # Initialize variables
        self.brush_size = 0
        self.roughness = 0.0
        self.tool = ToolEnum.BRUSH.value
        self.lock_ocean = False
        self.erase = False
        self.threshold = 0.75
        self.min_val = 0.0
        self.max_val = 1.0
        self.noise_settings = NoiseSettings(use_max=True)
        self.noise_settings.set_display_name("Noise")
        self.edge_distance_settings = EdgeDistanceSettings()
        self.change_callback = on_control_change
        self.planet_cfg = PlanetConfig()
        self.shape = shape
        self.preview_shape = (256, 256)
        self.noise = np.ones(self.shape, dtype=np.float32)
        self.view_api = view_api
        self.reset_callback = reset_callback
        self.create_widgets()
        self.load_config_dict()

    def create_widgets(self):
        """
        Create the widgets for the DEMPanel class using PyQt5.
        """
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(2, 2, 2, 2)
        main_layout.setSpacing(2)

        # Settings frame
        settings_frame = QWidget(self)
        settings_layout = QHBoxLayout(settings_frame)
        settings_layout.setContentsMargins(2, 2, 2, 2)
        settings_layout.setSpacing(2)

        left_panel = QVBoxLayout()
        left_panel.setContentsMargins(2, 2, 2, 2)
        left_panel.setSpacing(2)

        # Checkboxes
        self.lock_ocean_button = QCheckBox("Lock Ocean")
        self.lock_ocean_button.toggled.connect(self.callback)

        # Value sliders
        self.min_val_slider = FloatLabeledSlider("Min Value", 0.0, 1.0, 0.0, 0.01)
        self.min_val_slider.valueChanged.connect(lambda _: self.display_preview())

        self.max_val_slider = FloatLabeledSlider("Max Value", 0.0, 1.0, 1.0, 0.01)
        self.max_val_slider.valueChanged.connect(lambda _: self.display_preview())

        self.threshold_slider = FloatLabeledSlider("Threshold", 0.0, 1.0, 0.75, 0.01)
        self.threshold_slider.valueChanged.connect(lambda _: self.display_preview())

        sketch_control_group_box = QGroupBox("Sketch Controls")
        sketch_control_layout = QVBoxLayout()
        sketch_control_group_box.setLayout(sketch_control_layout)
        sketch_control_layout.setContentsMargins(2, 2, 2, 2)
        sketch_control_layout.setSpacing(0)
        sketch_control_layout.addWidget(self.lock_ocean_button)
        # TODO Remove properly
        # sketch_control_layout.addWidget(self.min_val_slider)
        # sketch_control_layout.addWidget(self.max_val_slider)
        # sketch_control_layout.addWidget(self.threshold_slider)
        left_panel.addWidget(sketch_control_group_box)

        # Create tool and settings layout
        tools_settings_layout = QHBoxLayout()

        # Tool panel
        self.tool_panel = ToolPanel(
            self,
            callback=self.callback,
            confirm_callback=self.confirm_edit,
            cancel_callback=self.cancel_edit,
            reset_callback=lambda: self.reset_callback(self.get_event_metadata(None, None)),
        )
        tools_settings_layout.addWidget(self.tool_panel)

        left_panel.addLayout(tools_settings_layout)
        settings_layout.addLayout(left_panel)

        main_layout.addWidget(settings_frame)

        # Initialize the SettingsPanel for noise and edge distance
        self.noise_settings_panel = SettingsPanel(
            self,
            self.noise_settings,
            lambda: None,
            onchange=lambda: self.display_preview(),
            onfinish=lambda: self.update_view_noise(),
            columns=5,
        )
        self.edge_filter_panel = SettingsPanel(
            self,
            self.edge_distance_settings,
            lambda: None,
            onchange=lambda: self.edge_distance_change(),
            columns=3,
        )

        # Initialize GlobePanel for noise preview
        self.noise_preview = GlobePanel(
            self,
            preview_size=self.preview_shape[0],
            shape=self.shape,
            view_api=self.view_api,
            globe_mode=GlobePanelMode.NOISE,
            initial_zoom=-1,
        )

        # Right side layout
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(2, 2, 2, 2)
        right_layout.setSpacing(2)
        right_layout.addWidget(self.edge_filter_panel)
        right_layout.addWidget(self.noise_settings_panel)
        settings_layout.addLayout(right_layout)

        preview_layout = QHBoxLayout()
        preview_layout.setContentsMargins(2, 2, 2, 2)
        preview_layout.setSpacing(2)
        preview_layout.addStretch()
        preview_layout.addWidget(self.noise_preview)
        preview_layout.addStretch()
        main_layout.addLayout(preview_layout)

        # Initialize noise preview
        self.noise_preview.set_noise_settings([self.noise_settings])

    def get_event_metadata(
        self, x, y, edit_type: EditTypeEnum | None = None
    ) -> DEMSketchEventMetadata:
        if (x is None or y is None) and edit_type is None:
            edit_type = EditTypeEnum.OTHER_CHANGE
        elif edit_type is None:
            edit_type = EditTypeEnum.DRAW

        metadata = DEMSketchEventMetadata(
            type=EventTypeEnum.DEM_SKETCH,
            x=x,
            y=y,
            brush_size=self.tool_panel.brush_size,
            tool=ToolEnum(self.tool_panel.tool_var),
            erase=self.tool_panel.erase,
            noise_settings=[self.noise_settings],
            noise=self.noise,
            edge_distance_settings=self.edge_distance_settings,
            roughness=self.tool_panel.roughness,
            lock_ocean=self.lock_ocean_button.isChecked(),
            min_value=self.min_val_slider.value,
            max_value=self.max_val_slider.value,
            threshold=self.threshold_slider.value,
            edit_type=edit_type,
        )
        return metadata

    @profile
    def callback(self, event=None, final=False):
        self.change_callback(
            self.get_event_metadata(None, None, edit_type=EditTypeEnum.OTHER_CHANGE)
        )

    def confirm_edit(self):
        self.change_callback(
            self.get_event_metadata(None, None, edit_type=EditTypeEnum.CONFIRM)
        )

    def cancel_edit(self):
        self.change_callback(
            self.get_event_metadata(None, None, edit_type=EditTypeEnum.CANCEL)
        )

    def edge_distance_change(self):
        self.change_callback(
            self.get_event_metadata(None, None, edit_type=EditTypeEnum.OTHER_CHANGE)
        )

    def get_config_dict(self):
        return {
            "lock_ocean": self.lock_ocean_button.isChecked(),
            "noise_settings": self.noise_settings.get_config_dict(),
        }

    def from_config_dict(self, config_dict: dict):
        self.lock_ocean_button.setChecked(config_dict.get("lock_ocean", False))
        self.noise_settings.from_config_dict(config_dict["noise_settings"])

    def noise_format_func(self) -> Callable[[np.ndarray], np.ndarray]:
        def format_noise(noise: np.ndarray) -> np.ndarray:
            return dilate_paint(noise, self.planet_cfg)

        return format_noise

    @profile
    def display_preview(self, changed: int = None):
        """
        Display the preview for the DEMPanel class.
        """
        self.noise_preview.noise_format_func = self.noise_format_func()
        self.noise_preview.set_noise_settings([self.noise_settings])

    def update_view_noise(self):
        if self.view_api is not None:
            self.view_api.set_noise_format_func(self.noise_format_func())
            self.view_api.set_noise_settings([self.noise_settings])
        self.callback()


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = DEMSketchPanel()
    window.show()
    sys.exit(app.exec_())
