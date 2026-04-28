import numpy as np
from typing import TYPE_CHECKING, Callable
from PyQt5.QtWidgets import (
    QWidget,
    QPushButton,
    QCheckBox,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QFileDialog,
)

from planetAI.src.data.utils import PlanetConfig
from planetAI.src.data.noise_settings import NoiseSettings, MeanFilterSettings
from planetAI.src.data.sketch_gen import dilate_paint
from ..interface_types import (
    DEMEventMetadata,
    EventTypeEnum,
    ToolEnum,
)
from .SavablePanel import SaveablePanel
from .GlobePanel import GlobePanel, GlobePanelMode
from .SettingsPanel import SettingsPanel
from .ToolPanel import ToolPanel

if TYPE_CHECKING:
    from ..view import View


class DEMPanel(QWidget, SaveablePanel):
    NAME = "dem_panel"

    def __init__(
        self,
        parent=None,
        import_callback=lambda filename: None,
        shape: tuple[int, int] = (256, 512),
        view_api: "View" = None,
        on_control_change: Callable[[DEMEventMetadata], None] = lambda metadata: None,
        reset_callback: Callable[[DEMEventMetadata], None] = lambda metadata: None,
        derive_callback: Callable[[], None] = lambda: None,
    ):
        super().__init__(parent)
        self.import_callback = import_callback
        self.noise_settings_0 = NoiseSettings(use_max=True)
        self.noise_settings_0.set_display_name("Continents")
        self.noise_settings_1 = NoiseSettings()
        self.noise_settings_1.set_display_name("Mountains")
        self.mean_filter = MeanFilterSettings()
        self.change_callback = on_control_change
        self.planet_cfg = PlanetConfig()
        self.shape = shape
        self.preview_shape = (256, 256)
        self.noise_0 = np.ones(self.shape, dtype=np.float32)
        self.noise_1 = np.ones(self.shape, dtype=np.float32)
        self.view_api = view_api
        self.reset_callback = reset_callback
        self.derive_callback = derive_callback
        self.create_widgets()
        self.load_config_dict()

    def create_widgets(self):
        main_layout = QVBoxLayout(self)

        # Settings row
        settings_group_box = QGroupBox("Sketch Controls")
        settings_layout = QVBoxLayout(settings_group_box)

        self.lock_ocean_button = QCheckBox("Lock Ocean")
        settings_layout.addWidget(self.lock_ocean_button)

        self.sketch_display_button = QCheckBox("Display as Sketch")
        self.sketch_display_button.toggled.connect(self.callback)
        settings_layout.addWidget(self.sketch_display_button)

        for button_text, callback in [
            ("Import", lambda: self._show_import_dialog()),
            ("Reset", lambda: self.reset_callback(self.get_event_metadata(None, None))),
            ("Derive", self.derive_callback),
        ]:
            button = QPushButton(button_text)
            button.clicked.connect(callback)
            settings_layout.addWidget(button)

        # Main content layout
        content_layout = QHBoxLayout()

        # Left column with settings and tools
        left_column = QVBoxLayout()
        left_column.addWidget(settings_group_box)

        # Replace tool container with ToolPanel
        self.tool_panel = ToolPanel(self)
        left_column.addWidget(self.tool_panel)

        content_layout.addLayout(left_column)

        # Middle column with noise settings
        middle_column = QVBoxLayout()
        self.noise_settings_panel_0 = SettingsPanel(
            self,
            self.noise_settings_0,
            lambda: None,
            onchange=lambda: self.display_preview(changed=0),
            columns=5,
        )
        middle_column.addWidget(self.noise_settings_panel_0)
        self.noise_settings_panel_1 = SettingsPanel(
            self,
            self.noise_settings_1,
            lambda: None,
            onchange=lambda: self.display_preview(changed=1),
            columns=5,
        )
        middle_column.addWidget(self.noise_settings_panel_1)
        content_layout.addLayout(middle_column)

        # Right column with preview and filter
        right_column = QVBoxLayout()
        self.noise_preview = GlobePanel(
            self,
            preview_size=self.preview_shape[0],
            shape=self.shape,
            view_api=self.view_api,
            globe_mode=GlobePanelMode.NOISE,
            initial_zoom=-1,
        )
        self.noise_preview.set_noise_settings(
            [self.noise_settings_0, self.noise_settings_1]
        )

        self.mean_filter_panel = SettingsPanel(
            self,
            self.mean_filter,
            lambda: None,
            onchange=lambda: self.display_preview(),
        )

        right_column.addWidget(self.mean_filter_panel)
        right_column.addWidget(self.noise_preview)
        content_layout.addLayout(right_column)

        main_layout.addLayout(content_layout)

    def callback(self, event=None, final=False):
        self.change_callback(self.get_event_metadata(None, None))
        self.display_preview()

    def get_event_metadata(self, x, y) -> DEMEventMetadata:
        metadata = DEMEventMetadata(
            type=EventTypeEnum.DEM,
            x=x,
            y=y,
            brush_size=self.tool_panel.brush_size,
            brush=self.tool_panel.tool_var == ToolEnum.BRUSH.value,
            lasso=self.tool_panel.tool_var == ToolEnum.LASSO.value,
            fill=self.tool_panel.tool_var == ToolEnum.FILL.value,
            noise_settings=[self.noise_settings_0, self.noise_settings_1],
            noise=self.noise,
            mean_filter=self.mean_filter,
            roughness=self.tool_panel.roughness,
            lock_ocean=self.lock_ocean_button.isChecked(),
            display_sketch=self.sketch_display_button.isChecked(),
            preview_coords=(self.noise_preview.latitude, self.noise_preview.longitude),
            view_coords=self.view_api.get_input_coords() if self.view_api else (None, None),
        )
        return metadata

    def get_config_dict(self):
        return {
            "display_sketch": self.sketch_display_button.isChecked(),
            "lock_ocean": self.lock_ocean_button.isChecked(),
            "noise_settings": [
                self.noise_settings_0.get_config_dict(),
                self.noise_settings_1.get_config_dict(),
            ],
        }

    def from_config_dict(self, config_dict):
        self.lock_ocean_button.setChecked(config_dict.get("lock_ocean", False))
        self.sketch_display_button.setChecked(config_dict.get("display_sketch", False))
        self.noise_settings_0.from_config_dict(config_dict["noise_settings"][0])
        self.noise_settings_1.from_config_dict(config_dict["noise_settings"][1])
        self.display_preview()

    @property
    def noise(self):
        return self.noise_0 * self.noise_1

    def noise_format_func(self) -> Callable[[np.ndarray], np.ndarray]:
        def format_noise(noise: np.ndarray) -> np.ndarray:
            if self.sketch_display_button.isChecked():
                return dilate_paint(noise, self.planet_cfg)
            return noise

        return format_noise

    def display_preview(self, changed: int = None):
        self.noise_preview.noise_format_func = self.noise_format_func()
        if self.view_api is not None:
            self.view_api.set_noise_format_func(self.noise_format_func())
            self.view_api.set_noise_settings([self.noise_settings_0, self.noise_settings_1])
        self.noise_preview.set_noise_settings(
            [self.noise_settings_0, self.noise_settings_1]
        )

    def _show_import_dialog(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Import File",
            "",
            "All Files (*.*)"
        )
        if filename:
            self.import_callback(filename)


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    panel = DEMPanel()
    panel.show()
    sys.exit(app.exec_())
