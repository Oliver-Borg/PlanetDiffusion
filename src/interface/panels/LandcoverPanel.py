from PyQt5.QtWidgets import (
    QWidget,
    QCheckBox,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QGroupBox,
    QPushButton,
    QFileDialog,
)
import numpy as np
from typing import TYPE_CHECKING, Callable

from planetAI.src.data.noise_settings import EdgeDistanceSettings
from planetAI.src.data.utils import gray_to_land
from planetAI.src.data.noise_settings import NoiseSettings
from ..interface_types import (
    LandcoverClass,
    LandCoverEventMetadata,
    EventTypeEnum,
    EditTypeEnum,
    ToolEnum,
)
from ..utils import format_landcover_noise
from .SavablePanel import SaveablePanel
from .GlobePanel import GlobePanel, GlobePanelMode
from .SettingsPanel import SettingsPanel
from .LandcoverRow import LandCoverRow
from .ToolPanel import ToolPanel
from ..widgets.LabeledSlider import FloatLabeledSlider

if TYPE_CHECKING:
    from ..view import View


class LandcoverPanel(QWidget, SaveablePanel):
    """
    This is a reusable panel class that contains two rows.
    The first row contains buttons for the different landcover types.
    The second row contains the subclasses of the selected landcover type, by temperature.
    These subclasses are only shown when hovering over the landcover type or after it is selected.
    There is a slider for controlling the size of the brush.
    """

    NAME = "landcover_panel"

    def __init__(
        self,
        parent=None,
        landcover_classes: list[LandcoverClass] = [],
        shape: tuple[int, int] = (256, 512),
        view_api: "View" = None,
        on_control_change: Callable[
            [LandCoverEventMetadata], None
        ] = lambda metadata: None,
        reset_callback: Callable[
            [LandCoverEventMetadata], None
        ] = lambda metadata: None,
        import_callback: Callable[[str], None] = lambda filename: None,
        derive_callback: Callable[[], None] = lambda: None
    ):
        """
        Constructor for the LandcoverPanel class.
        """
        super().__init__(parent)
        self.landcover_classes = landcover_classes
        self.length = len(landcover_classes)
        self.max_subclasses = max(
            [len(landcover_class.subclasses) for landcover_class in landcover_classes]
            + [0]
        )
        self.ratio_value = 0.9
        self.lock_ocean_value = False
        self.modal_view_value = False
        self.noise_settings = NoiseSettings()
        self.noise_settings.set_display_name("Landcover")
        self.edge_distance_settings = EdgeDistanceSettings()
        self.change_callback = on_control_change
        self.shape = shape
        self.preview_shape = (256, 256)
        self.noise = np.zeros(self.shape, dtype=np.float32)
        self.view_api = view_api
        self.reset_callback = reset_callback
        self.import_callback = import_callback
        self.derive_callback = derive_callback
        self.create_widgets()
        self.landcover_row.select_primary_class(landcover_classes[0], 0)
        self.load_config_dict()
        self.callback()

    def create_widgets(self):
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        right_layout = QVBoxLayout()
        left_layout = QVBoxLayout()

        # Create landcover row
        self.landcover_row = LandCoverRow(
            self,
            landcover_classes=self.landcover_classes,
            on_change_callback=lambda: self.callback(),
        )
        left_layout.addWidget(self.landcover_row)

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

        # Settings frame
        self.settings_group_box = QGroupBox("Sketch Controls")
        settings_layout = QVBoxLayout()
        self.settings_group_box.setLayout(settings_layout)

        self.lock_ocean_button = QCheckBox("Lock Ocean")
        self.lock_ocean_button.setChecked(False)
        self.lock_ocean_button.stateChanged.connect(lambda: self.callback())

        self.modal_view_button = QCheckBox("Modal View")
        self.modal_view_button.setChecked(False)
        self.modal_view_button.stateChanged.connect(self.callback)

        for widget in [
            self.lock_ocean_button,
            self.modal_view_button,
        ]:
            settings_layout.addWidget(widget)

        import_button = QPushButton("Import")
        import_button.clicked.connect(lambda: self._show_import_dialog())
        settings_layout.addWidget(import_button)

        derive_button = QPushButton("Derive")
        derive_button.clicked.connect(lambda: self._derive_callback())
        settings_layout.addWidget(derive_button)

        tools_settings_layout.addWidget(self.settings_group_box)

        # Edge filter frame
        self.edge_filter_frame = QFrame()
        edge_filter_layout = QVBoxLayout()
        self.edge_filter_frame.setLayout(edge_filter_layout)
        self.edge_filter_panel = SettingsPanel(
            self.edge_filter_frame,
            self.edge_distance_settings,
            lambda: None,
            onchange=lambda: self.edge_distance_change(),
        )
        edge_filter_layout.addWidget(self.edge_filter_panel)
        tools_settings_layout.addWidget(self.edge_filter_frame)

        # Noise settings frame and preview
        noise_container = QHBoxLayout()

        self.noise_settings_frame = QFrame()
        noise_layout = QVBoxLayout()
        self.noise_settings_frame.setLayout(noise_layout)

        self.noise_settings_panel = SettingsPanel(
            self.noise_settings_frame,
            self.noise_settings,
            columns=2,
            onchange=self.callback,
        )
        noise_layout.addWidget(self.noise_settings_panel)
        noise_container.addWidget(self.noise_settings_frame)

        preview_group_box = QGroupBox("Noise Preview")
        preview_container = QVBoxLayout(preview_group_box)
        self.noise_preview = GlobePanel(
            self.noise_settings_frame,
            preview_size=self.preview_shape[0],
            shape=self.shape,
            view_api=self.view_api,
            globe_mode=GlobePanelMode.NOISE,
            initial_zoom=-1,
        )
        preview_container.addWidget(self.noise_preview)

        # Preview settings

        self.ratio_slider = FloatLabeledSlider("Primary ratio", 0, 1, 0.9, 0.01)
        self.ratio_slider.slider.valueChanged.connect(lambda: self.callback())
        preview_container.addWidget(self.ratio_slider)

        noise_container.addWidget(preview_group_box)
        right_layout.addLayout(noise_container)
        left_layout.addLayout(tools_settings_layout)
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

    def _derive_callback(self):
        # Force a noise update
        self.display_preview()
        self.derive_callback()
        self.callback()

    def _show_import_dialog(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Import File",
            "",
            "All Files (*.*)"
        )
        if filename:
            self.import_callback(filename)

    def display_preview(self):
        """
        Display the preview for the LandcoverPanel class.
        """
        if self.view_api is not None:
            self.view_api.set_noise_format_func(self.noise_format_func())
        self.noise_preview.noise_format_func = self.noise_format_func()
        if self.view_api is not None:
            self.view_api.set_noise_settings([self.noise_settings])
        self.noise_preview.set_noise_settings([self.noise_settings])

    def noise_format_func(self) -> Callable[[np.ndarray], np.ndarray]:
        def format_noise(noise: np.ndarray) -> np.ndarray:
            noise = format_landcover_noise(noise, self.get_event_metadata(None, None))
            noise = gray_to_land(noise)
            return noise

        return format_noise

    def callback(self, event=None, final=False):
        self.display_preview()
        self.change_callback(self.get_event_metadata(None, None))

    def edge_distance_change(self):
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

    def get_event_metadata(
        self, x: int, y: int, edit_type: EditTypeEnum | None = None
    ) -> LandCoverEventMetadata:
        """
        Get the event for the LandcoverPanel class.
        """
        if (x is None or y is None) and edit_type is None:
            edit_type = EditTypeEnum.OTHER_CHANGE
        elif edit_type is None:
            edit_type = EditTypeEnum.DRAW
        metadata = LandCoverEventMetadata(
            type=EventTypeEnum.LANDCOVER,
            x=x,
            y=y,
            primary_class=self.landcover_row.primary_class,
            primary_subclass=self.landcover_row.primary_subclass,
            secondary_class=self.landcover_row.secondary_class,
            brush_size=self.tool_panel.brush_size,
            tool=ToolEnum(self.tool_panel.tool_var),
            noise_settings=self.noise_settings,
            primary_ratio=self.ratio_slider.value,
            noise=self.noise,
            edge_distance_settings=self.edge_distance_settings,
            roughness=self.tool_panel.roughness,
            erase=self.tool_panel.erase,
            lock_ocean=self.lock_ocean_var,
            modal_view=self.modal_view_var,
            preview_coords=(self.noise_preview.latitude, self.noise_preview.longitude),
            view_coords=(
                self.view_api.get_input_coords()
                if self.view_api is not None
                else (None, None)
            ),
            edit_type=edit_type,
        )
        return metadata

    @property
    def lock_ocean_var(self):
        return self.lock_ocean_button.isChecked()

    @property
    def modal_view_var(self):
        return self.modal_view_button.isChecked()

    def get_config_dict(self):
        return {
            "noise_settings": self.noise_settings.get_config_dict(),
            "ratio": self.ratio_slider.value,
            "lock_ocean": self.lock_ocean_var,
            "modal_view": self.modal_view_var,
        }

    def from_config_dict(self, config_dict: dict):
        self.noise_settings.from_config_dict(config_dict["noise_settings"])
        self.ratio_slider.setValue(int(config_dict["ratio"] * 100))
        self.lock_ocean_button.setChecked(config_dict["lock_ocean"])
        self.modal_view_button.setChecked(config_dict["modal_view"])


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication

    app = QApplication([])
    landcover_classes = [
        LandcoverClass(
            f"Test {c}",
            25*c,
            f"#{c}0{c}0{c}0",
            [LandcoverClass(f"Test {i}", 50*i, f"#{i*2}00000", []) for i in range(5)],
        )
        for c in range(10)
    ]
    landcover_panel = LandcoverPanel(landcover_classes=landcover_classes)
    landcover_panel.show()
    app.exec_()
