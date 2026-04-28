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
    QGridLayout,
)
from PyQt5.QtCore import pyqtSignal
from planetAI.src.data.utils import PlanetConfig
from planetAI.src.data.landcover_utils import used_classes, landcover_mapping
from planetAI.src.data.mean_precip import PrecipSketch
from ..widgets.LabeledSlider import FloatLabeledSlider, IntLabeledSlider
from ..interface_types import (
    RiverDerivationSettings,
    RiverEventMetadata,
    RiverDisplayType,
    EventTypeEnum,
)
from .SavablePanel import SaveablePanel
from .Enums import BrushModeEnum

if TYPE_CHECKING:
    from ..view import View


class MultiplierSliderPanel(QWidget, SaveablePanel):
    valueChanged = pyqtSignal()  # Add this signal

    def __init__(
        self,
        parent=None,
        keys_and_labels: dict[int, str] = [],
        name: str = "MULTIPLIER_PANEL",
    ):
        self.NAME = name
        super().__init__(parent)
        self.keys_and_labels = keys_and_labels
        self.sliders: dict[int, FloatLabeledSlider] = {}
        self.main_layout = QGridLayout(self)
        self.setMaximumWidth(300)
        self.setContentsMargins(10, 10, 10, 10)
        self.create_widgets()
        self.load_config_dict()

    def create_widgets(self):
        row = 0
        col = 0
        for key, label in self.keys_and_labels.items():
            slider = FloatLabeledSlider(
                f"{label} Multiplier", min_val=0, max_val=10.0, value=1.0, step=0.1
            )
            slider.valueChanged.connect(self.valueChanged.emit)  # Connect the signal
            self.sliders[key] = slider
            self.main_layout.addWidget(slider, row, col)
            col = (col + 1) % 2  # Toggle between 0 and 1
            if col == 0:
                row += 1

    def get_config_dict(self):
        config_dict = {}
        for key, slider in self.sliders.items():
            config_dict[str(key)] = slider.value
        return config_dict

    def from_config_dict(self, config_dict):
        for key, slider in self.sliders.items():
            str_key = str(key)
            if str_key in config_dict:
                slider.setValue(config_dict[str_key])

    def get_multipliers(self) -> dict[int, float]:
        return {key: slider.value for key, slider in self.sliders.items()}


class RiverPanel(QWidget, SaveablePanel):
    """
    A panel for displaying rivers.
    """

    NAME = "river_panel"
    controlChanged = pyqtSignal(RiverEventMetadata)

    def __init__(
        self,
        parent=None,
        preview_size: int = 512,
        derive_callback: Callable[
            [RiverDerivationSettings], None
        ] = lambda settings: None,
        shape: tuple[int, int] = (4096, 8192),
        view_api: "View" = None,
        planet_config: PlanetConfig = PlanetConfig(),
        on_control_change: Callable[[RiverEventMetadata], None] = lambda metadata: None,
        reset_callback: Callable[[RiverEventMetadata], None] = lambda metadata: None,
        on_precip_change: Callable[[RiverDerivationSettings], None] = lambda settings: None,
    ):
        super().__init__(parent)
        self.preview_size = preview_size
        self.shape = shape
        self.view_api = view_api
        self.planet_config = planet_config
        self.landcover_keys_and_labels = {
            landcover_mapping[c].gray_colour: landcover_mapping[c].display_name
            for c in used_classes
        }
        self.temp_keys_and_labels = {
            i: label
            for i, label in zip(
                self.planet_config.temp_colour_list(), self.planet_config.temp_labels()
            )
        }
        self.precip_sketcher = PrecipSketch(planet_config)
        self.full_mapping = self.precip_sketcher.get_full_mapping(
            land_mults={}, temp_mults={}
        )
        self.mapping_image = self.precip_sketcher.get_full_mapping_matrix(
            self.full_mapping,
            self.landcover_keys_and_labels,
            self.temp_keys_and_labels,
        )
        self.mode_var = 0
        self.display_type_var = 0
        self.change_callback = on_control_change
        self.derive_callback = derive_callback
        self.on_precip_change = on_precip_change
        self.reset_callback = reset_callback
        self.main_layout = QVBoxLayout(self)
        self.setMinimumWidth(1200)  # Increased from 900
        self.setMaximumWidth(1600)  # Increased from 1080
        self.setMaximumHeight(540)
        self.setContentsMargins(10, 10, 10, 10)
        self.create_widgets()
        self.load_config_dict()

    def create_widgets(self):
        # Add spacers to center the content
        v_spacer_top = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        v_spacer_bottom = QSpacerItem(
            20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding
        )

        self.main_layout.addItem(v_spacer_top)

        # Derive frame
        derive_frame = QWidget()
        derive_layout = QVBoxLayout(derive_frame)

        self.base_weight_slider = FloatLabeledSlider(
            "Base Weight", min_val=0, max_val=3.0, value=0, step=0.05
        )
        self.noise_factor_slider = FloatLabeledSlider(
            "Noise Factor", min_val=0, max_val=1.0, value=0.5, step=0.05
        )
        self.river_max_slider = FloatLabeledSlider(
            "River Max", min_val=0, max_val=255, value=255, step=0.1
        )
        self.river_min_slider = FloatLabeledSlider(
            "River Min", min_val=0, max_val=255, value=0, step=0.1
        )
        self.river_max_slider.valueChanged.connect(lambda x: self.callback())
        self.river_min_slider.valueChanged.connect(lambda x: self.callback())

        self.derive_button = QPushButton("Derive Rivers")
        self.derive_button.clicked.connect(
            lambda: self.derive_callback(
                self.settings
            )
        )

        derive_layout.addWidget(self.derive_button)
        derive_layout.addWidget(self.base_weight_slider)
        derive_layout.addWidget(self.noise_factor_slider)
        derive_layout.addWidget(self.river_max_slider)
        derive_layout.addWidget(self.river_min_slider)

        # Controls layout
        controls_layout = QHBoxLayout()

        # Brush size
        brush_container = QWidget()
        brush_layout = QVBoxLayout(brush_container)
        self.brush_size_slider = IntLabeledSlider("Brush Size", 0, 30, 0)
        self.weight_value_slider = FloatLabeledSlider(
            "Weight Value", min_val=0, max_val=10.0, value=0, step=0.05
        )
        self.efficiency_value_slider = FloatLabeledSlider(
            "Efficiency", min_val=0, max_val=1.0, value=1.0, step=0.05
        )
        self.iterations_value_slider = IntLabeledSlider(
            "Iterations", min_val=1, max_val=15, value=1
        )
        self.smoothing_value_slider = IntLabeledSlider(
            "Smoothing", min_val=0, max_val=15, value=0
        )
        brush_layout.addWidget(self.brush_size_slider)
        brush_layout.addWidget(self.weight_value_slider)
        brush_layout.addWidget(self.efficiency_value_slider)
        brush_layout.addWidget(self.iterations_value_slider)
        brush_layout.addWidget(self.smoothing_value_slider)

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

        self.display_modal_button = QRadioButton("Modal")
        self.display_dem_button = QRadioButton("DEM")
        self.display_upa_button = QRadioButton("UPA")
        self.display_weights_button = QRadioButton("Weights")
        self.display_efficiency_button = QRadioButton("Efficiency")
        self.display_upa_button.setChecked(True)
        self.display_group.addButton(
            self.display_modal_button, RiverDisplayType.MODAL.value
        )
        self.display_group.addButton(
            self.display_dem_button, RiverDisplayType.DEM.value
        )
        self.display_group.addButton(
            self.display_upa_button, RiverDisplayType.UPA.value
        )
        self.display_group.addButton(
            self.display_weights_button, RiverDisplayType.WEIGHTS.value
        )
        self.display_group.addButton(
            self.display_efficiency_button, RiverDisplayType.EFFICIENCY.value
        )
        self.display_group.buttonClicked.connect(self.callback)

        display_layout.addWidget(self.display_modal_button)
        display_layout.addWidget(self.display_dem_button)
        display_layout.addWidget(self.display_upa_button)
        display_layout.addWidget(self.display_weights_button)
        display_layout.addWidget(self.display_efficiency_button)

        # Reset button
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(
            lambda: self.reset_callback(self.get_event_metadata(None, None))
        )

        self.update_weights_button = QPushButton("Update Weights")
        self.update_weights_button.clicked.connect(
            lambda: self.on_precip_change(self.settings)
        )

        # Multiplier frame and tab groups
        self.multiplier_frame = QWidget()
        multiplier_layout = QHBoxLayout(self.multiplier_frame)
        multiplier_layout.setContentsMargins(0, 0, 0, 0)
        multiplier_layout.setSpacing(10)

        self.landcover_multiplier_panel = MultiplierSliderPanel(
            keys_and_labels=self.landcover_keys_and_labels,
            name="LANDCOVER_MULTIPLIER_PANEL",
        )
        self.temp_multiplier_panel = MultiplierSliderPanel(
            keys_and_labels=self.temp_keys_and_labels, name="TEMP_MULTIPLIER_PANEL"
        )
        # Connect the signals
        self.landcover_multiplier_panel.valueChanged.connect(self.update_mapping_matrix)
        self.temp_multiplier_panel.valueChanged.connect(self.update_mapping_matrix)

        multiplier_layout.addWidget(self.landcover_multiplier_panel)
        multiplier_layout.addWidget(self.temp_multiplier_panel)

        self.matrix_image_frame = QWidget()
        matrix_image_layout = QVBoxLayout(self.matrix_image_frame)
        matrix_image_layout.setContentsMargins(0, 0, 0, 0)
        matrix_image_layout.setSpacing(10)
        self.mapping_image_label = QLabel()
        self.mapping_image_label.setPixmap(
            self.mapping_image.toqpixmap().scaledToWidth(250)
        )
        matrix_image_layout.addWidget(QLabel("Precipitation Mapping"))
        matrix_image_layout.addWidget(self.mapping_image_label)
        multiplier_layout.addWidget(self.matrix_image_frame)

        # Add all widgets to main layout with stretch factors
        controls_layout.addWidget(derive_frame, 1)
        controls_layout.addWidget(brush_container, 2)
        controls_layout.addWidget(tool_frame, 1)
        controls_layout.addWidget(display_frame, 1)
        controls_layout.addWidget(self.reset_button, 1)
        controls_layout.addWidget(self.update_weights_button, 1)
        controls_layout.addWidget(self.multiplier_frame, 4)

        self.main_layout.addLayout(controls_layout)
        self.main_layout.addItem(v_spacer_bottom)

    @property
    def settings(self):
        return RiverDerivationSettings(
            base_weight=self.base_weight,
            noise_factor=self.noise_factor,
            smoothing=self.smoothing_var,
            full_mapping=self.full_mapping,
            noise=None,
            use_sketch=True,
        )

    @property
    def brush_size(self):
        return self.brush_size_slider.value

    @property
    def iterations_var(self):
        return self.iterations_value_slider.value

    @property
    def smoothing_var(self):
        return self.smoothing_value_slider.value

    @property
    def weight_value(self):
        return self.weight_value_slider.value

    @property
    def efficiency_value(self):
        return self.efficiency_value_slider.value

    @property
    def base_weight(self):
        return self.base_weight_slider.value

    @property
    def noise_factor(self):
        return self.noise_factor_slider.value

    @property
    def river_max(self):
        return self.river_max_slider.value

    @property
    def river_min(self):
        return self.river_min_slider.value

    def _update_mode(self, button):
        self.mode_var = self.tool_group.id(button)
        self.callback()

    def callback(self):
        self.change_callback(self.get_event_metadata(None, None))

    def get_event_metadata(self, x: int, y: int) -> RiverEventMetadata:
        metadata = RiverEventMetadata(
            type=EventTypeEnum.RIVER,
            x=x,
            y=y,
            brush_size=self.brush_size,
            weight_value=self.weight_value,
            efficiency_value=self.efficiency_value,
            erase=self.mode_var == BrushModeEnum.ERASE.value,
            display_type=RiverDisplayType(self.display_group.checkedId()),
            full_mapping=self.full_mapping,
            river_max=self.river_max,
            river_min=self.river_min,
            settings=self.settings,
        )
        return metadata

    def update_mapping_matrix(self):
        land_mults = self.landcover_multiplier_panel.get_multipliers()
        temp_mults = self.temp_multiplier_panel.get_multipliers()
        self.full_mapping = self.precip_sketcher.get_full_mapping(
            land_mults=land_mults, temp_mults=temp_mults
        )
        self.mapping_image = self.precip_sketcher.get_full_mapping_matrix(
            self.full_mapping,
            self.landcover_keys_and_labels,
            self.temp_keys_and_labels,
        )
        self.mapping_image_label.setPixmap(
            self.mapping_image.toqpixmap().scaledToWidth(250)
        )


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    panel = RiverPanel()
    panel.show()
    sys.exit(app.exec_())
    panel.show()
    sys.exit(app.exec_())
    panel.show()
    sys.exit(app.exec_())
