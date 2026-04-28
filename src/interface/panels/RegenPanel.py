from typing import TYPE_CHECKING, Callable
from PyQt5.QtWidgets import (
    QWidget,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QButtonGroup,
    QSpacerItem,
    QSizePolicy,
)
from PyQt5.QtCore import pyqtSignal
from planetAI.src.data.utils import PlanetConfig
from ..interface_types import (
    EventTypeEnum,
    OutputDisplayType,
    RegenEventMetadata,
)
from .SavablePanel import SaveablePanel
from .Enums import BrushModeEnum

if TYPE_CHECKING:
    from ..view import View


class RegenPanel(QWidget, SaveablePanel):
    """
    A panel for controlling regeneration of parts of the globe.
    """

    NAME = "regen_panel"
    controlChanged = pyqtSignal(RegenEventMetadata)

    def __init__(
        self,
        parent=None,
        preview_size: int = 512,
        derive_callback: Callable[[int, int], None] = lambda: None,
        shape: tuple[int, int] = (4096, 8192),
        view_api: "View" = None,
        planet_config: PlanetConfig = PlanetConfig(),
        on_control_change: Callable[[RegenEventMetadata], None] = lambda metadata: None,
        reset_callback: Callable[[RegenEventMetadata], None] = lambda metadata: None,
    ):
        super().__init__(parent)
        self.preview_size = preview_size
        self.shape = shape
        self.view_api = view_api
        self.planet_config = planet_config
        self.display_type_var = 0
        self.mode_var = 0
        self.change_callback = on_control_change
        self.derive_callback = derive_callback
        self.reset_callback = reset_callback
        self.main_layout = QVBoxLayout(self)
        self.setMaximumWidth(400)
        self.setMaximumHeight(300)
        self.setContentsMargins(10, 10, 10, 10)
        self.create_widgets()
        self.load_config_dict()

    def create_widgets(self):
        # Add spacers to center the content
        h_spacer_left = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        h_spacer_right = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        v_spacer_top = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        v_spacer_bottom = QSpacerItem(
            20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding
        )

        self.main_layout.addItem(v_spacer_top)

        # Controls layout
        controls_layout = QHBoxLayout()
        controls_layout.addItem(h_spacer_left)

        # Tool frame
        tool_frame = QWidget()
        tool_layout = QVBoxLayout(tool_frame)
        tool_layout.addWidget(QLabel("Tool"))

        self.tool_group = QButtonGroup(self)
        self.brush_mode_button = QRadioButton("Brush")
        self.erase_mode_button = QRadioButton("Erase")
        self.brush_mode_button.setChecked(True)
        self.tool_group.addButton(self.brush_mode_button, 0)
        self.tool_group.addButton(self.erase_mode_button, 1)
        self.tool_group.buttonClicked.connect(self._update_mode)

        tool_layout.addWidget(self.brush_mode_button)
        tool_layout.addWidget(self.erase_mode_button)

        # Display frame
        display_frame = QWidget()
        display_layout = QVBoxLayout(display_frame)
        display_layout.addWidget(QLabel("Display Type"))

        self.display_group = QButtonGroup(self)
        self.display_modal_button = QRadioButton("Satellite")
        self.display_dem_button = QRadioButton("DEM")
        self.display_modal_button.setChecked(True)
        self.display_group.addButton(
            self.display_modal_button, OutputDisplayType.SATELLITE.value
        )
        self.display_group.addButton(
            self.display_dem_button, OutputDisplayType.DEM.value
        )
        self.display_group.buttonClicked.connect(self.callback)

        display_layout.addWidget(self.display_modal_button)
        display_layout.addWidget(self.display_dem_button)

        # Reset button
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(
            lambda: self.reset_callback(self.get_event_metadata(None, None))
        )

        # Add all widgets to main layout
        controls_layout.addWidget(tool_frame)
        controls_layout.addWidget(display_frame)

        controls_layout.addItem(h_spacer_right)
        self.main_layout.addLayout(controls_layout)
        self.main_layout.addWidget(self.reset_button)
        self.main_layout.addItem(v_spacer_bottom)

    def _update_mode(self, button):
        self.mode_var = self.tool_group.id(button)
        self.callback()

    def callback(self):
        self.change_callback(self.get_event_metadata(None, None))

    def get_event_metadata(self, x: int, y: int) -> RegenEventMetadata:
        metadata = RegenEventMetadata(
            type=EventTypeEnum.REGEN,
            x=x,
            y=y,
            brush_size=0,
            erase=self.mode_var == BrushModeEnum.ERASE.value,
            display_type=OutputDisplayType(self.display_group.checkedId()),
        )
        return metadata


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    panel = RegenPanel()
    panel.show()
    sys.exit(app.exec_())
