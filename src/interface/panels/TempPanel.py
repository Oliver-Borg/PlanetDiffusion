from PyQt5.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QCheckBox,
    QVBoxLayout,
    QHBoxLayout,
    QSizePolicy,
    QGroupBox,
)
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

from planetAI.src.data.noise_settings import NoiseSettings
from planetAI.src.data.utils import profile
from planetAI.src.data.sphere_mapping import SphereMapping

from ..interface_types import (
    TempPresetEventMetadata,
    EventTypeEnum,
)
from ..utils import create_temp_preset

from .SavablePanel import SaveablePanel
from .SettingsPanel import SettingsPanel
from .TempSelector import TempSelector
from ..widgets.LabeledSlider import IntLabeledSlider, FloatLabeledSlider


class TempPanel(QWidget, SaveablePanel):
    """
    This is a reusable panel class that contains rows for controlling temperature.
    There is a panel containing a number of circles,
    where the x coordinate of each circle is the latitude and the y coordinate is the temperature.
    There two additional sliders for Weight and Pivot and a button to reset the values.
    On the right there is a preview of a circle with the temperature bands.
    TODO: Add some fun defaults for climates
    """

    NAME = "temp_panel"

    def __init__(
        self,
        parent=None,
        shape=(256, 512),
        def_num_latitudes=9,
        max_temp=37,
        min_temp=-45,
        on_control_change: Callable[
            [TempPresetEventMetadata], None
        ] = lambda metadata: None,
        reset_callback: Callable[
            [TempPresetEventMetadata], None
        ] = lambda metadata: None,
    ):
        """
        Constructor for the TempPanel class.
        """
        super().__init__(parent)
        self.def_num_latitudes = def_num_latitudes
        self.max_temp = max_temp
        self.min_temp = min_temp
        self.preview_size = 256
        self.circle_mask = np.zeros((self.preview_size, self.preview_size, 3), dtype=bool)
        for i in range(self.preview_size):
            for j in range(self.preview_size):
                if (i - self.preview_size // 2) ** 2 + (j - self.preview_size // 2) ** 2 < (
                    self.preview_size // 2
                ) ** 2:
                    self.circle_mask[i, j, :] = True
        self.shape = shape
        self.noise = np.zeros(shape, dtype=np.float32)
        self.change_callback = on_control_change
        self.noise_settings = NoiseSettings()
        self.display_as_sketch = False
        self.reset_callback = reset_callback
        self.main_layout = QHBoxLayout(self)
        self.create_widgets()
        self.set_defaults()
        self.load_config_dict()
        self.update_noise()

    def create_widgets(self):
        """
        Create the widgets for the TempPanel class using box layouts
        """
        # Left section - Temperature Selector
        left_widget = QWidget()
        left_section = QVBoxLayout(left_widget)
        left_section.setContentsMargins(0, 0, 0, 0)

        self.temp_selector = TempSelector(
            self,
            def_num_latitudes=self.def_num_latitudes,
            max_temp=self.max_temp,
            min_temp=self.min_temp,
            on_change=self.callback,
            update_labels=self.update_labels,
        )
        self.temp_selector.setMinimumSize(512, 256)  # Set minimum size
        self.temp_selector.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_section.addWidget(self.temp_selector)

        # Middle section - Controls
        middle_widget = QGroupBox("Temperature Controls")
        middle_widget.setMinimumWidth(150)  # Set minimum width for controls
        middle_section = QVBoxLayout(middle_widget)
        middle_section.setSpacing(10)  # Add some spacing between elements

        # Labels container
        labels_container = QVBoxLayout()
        self.latitude_label = QLabel("Lat: 0")
        self.temperature_label = QLabel("Temp: 0")
        labels_container.addWidget(self.latitude_label)
        labels_container.addWidget(self.temperature_label)
        middle_section.addLayout(labels_container)

        # Sliders container
        sliders_container = QVBoxLayout()
        self.weight_slider = FloatLabeledSlider(
            "Noise Weight",
            min_val=0,
            max_val=1,
            value=0,
            step=0.05,
        )
        self.weight_slider.valueChanged.connect(self.callback)
        sliders_container.addWidget(self.weight_slider)
        self.pivot_slider = IntLabeledSlider(
            "Pivot",
            min_val=-45,
            max_val=45,
            value=0,
        )
        self.pivot_slider.valueChanged.connect(self.callback)
        sliders_container.addWidget(self.pivot_slider)
        self.steepness_slider = IntLabeledSlider(
            "Steepness",
            min_val=0,
            max_val=40,
            value=20,
        )
        self.steepness_slider.valueChanged.connect(self.callback)
        sliders_container.addWidget(self.steepness_slider)
        self.factor_slider = IntLabeledSlider(
            "Factor",
            min_val=0,
            max_val=50,
            value=25,
        )
        self.factor_slider.valueChanged.connect(self.callback)
        sliders_container.addWidget(self.factor_slider)
        sliders_container.addStretch()

        middle_section.addLayout(sliders_container)

        # Controls container
        controls_container = QVBoxLayout()
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset)
        self.sketch_display = QCheckBox("Display as Sketch")
        self.sketch_display.stateChanged.connect(self.callback)
        self.modal_view = QCheckBox("Modal View")
        self.modal_view.stateChanged.connect(self.callback)

        controls_container.addWidget(self.reset_button)
        controls_container.addWidget(self.sketch_display)
        controls_container.addWidget(self.modal_view)
        controls_container.addStretch()
        middle_section.addLayout(controls_container)

        # Right section - Preview and Settings
        right_widget = QWidget()
        right_section = QHBoxLayout(right_widget)
        right_section.setContentsMargins(0, 0, 0, 0)

        # Preview
        self.preview_label = QLabel()
        self.preview_label.setFixedSize(self.preview_size, self.preview_size)
        right_section.addWidget(self.preview_label)

        # Noise Settings
        self.noise_settings_panel = SettingsPanel(
            self, self.noise_settings, lambda: None, lambda: self.update_noise()
        )
        right_section.addWidget(self.noise_settings_panel)

        # Add all sections to main layout
        self.main_layout.addWidget(left_widget)
        self.main_layout.addWidget(middle_widget)
        self.main_layout.addWidget(right_widget)

        # Set main layout properties
        self.main_layout.setSpacing(2)
        self.main_layout.setContentsMargins(2, 2, 2, 2)

    def update_noise(self):
        """
        Update the noise settings for the TempPanel class.
        """
        sphere_mapping = SphereMapping(shape=self.shape)
        noise = sphere_mapping.generate_noise(self.get_event_metadata(None, None).noise_settings)
        self.noise = noise * 255
        self.callback(None)

    @profile
    def callback(self, event=None):
        self.draw_preview()
        self.change_callback(self.get_event_metadata(None, None))

    def reset(self):
        """
        Reset the values for the TempPanel class.
        """
        self.set_defaults()
        self.reset_callback(self.get_event_metadata(None, None))
        self.callback(None)

    def set_defaults(self):
        """
        Set the default values for the sliders.
        """
        self.weight_slider.setValue(0)
        self.pivot_slider.setValue(0)
        self.temp_selector.set_defaults()
        self.temp_selector.redraw()

    def update_labels(self, event):
        """
        Update the labels for the TempPanel class.
        """
        x, y = event.x(), event.y()
        latitude, temperature = self.temp_selector.coords_to_temp(x, y)
        latitude = min(90, max(-90, latitude))
        temperature = min(self.max_temp, max(self.min_temp, temperature))
        self.latitude_label.setText(f"Lat: {latitude:.1f}")
        self.temperature_label.setText(f"Temp: {temperature:.1f}")

    @profile
    def draw_preview(self):
        """
        Draw the preview using PyQt5
        """
        preview_size = self.preview_size
        preview = create_temp_preset(
            self.get_event_metadata(None, None), (preview_size, preview_size), as_sketch=True
        )

        # Create circle mask
        cmap = plt.get_cmap("coolwarm")
        preview = cmap(preview)
        preview = (preview[:, :, :3] * 255).astype(np.uint8)
        preview[~self.circle_mask] = 128
        preview = preview.clip(0, 255)

        # Convert numpy array to QImage and display
        height, width, channels = preview.shape
        bytes_per_line = channels * width
        qt_image = QImage(
            preview.data, width, height, bytes_per_line, QImage.Format_RGB888
        )
        pixmap = QPixmap.fromImage(qt_image)
        self.preview_label.setPixmap(pixmap)

    def get_event_metadata(self, x, y) -> TempPresetEventMetadata:
        """
        Get the event metadata with PyQt5 values
        """
        lat_temp_list = [
            self.temp_selector.coords_to_temp(px, py)
            for px, py in sorted(self.temp_selector.points, key=lambda p: p[0])
        ]
        metadata = TempPresetEventMetadata(
            type=EventTypeEnum.TEMP_PRESET,
            x=x,
            y=y,
            brush_size=0,
            lat_temp_list=lat_temp_list,
            noise_weight=self.weight_slider.value,
            pivot=self.pivot_slider.value,
            min_temp=self.min_temp,
            max_temp=self.max_temp,
            noise_settings=self.noise_settings,
            noise=self.noise,
            display_as_sketch=self.sketch_display.isChecked(),
            modal_view=self.modal_view.isChecked(),
            steepness=self.steepness_slider.value,
            factor=self.factor_slider.value,
        )
        return metadata

    def get_config_dict(self):
        return {
            "noise_weight": self.weight_slider.value,
            "pivot": self.pivot_slider.value,
            "display_as_sketch": self.sketch_display.isChecked(),
            "noise_settings": self.noise_settings.get_config_dict(),
            "modal_view": self.modal_view.isChecked(),
            "points": self.temp_selector.points,
            "steepness": self.steepness_slider.value,
            "factor": self.factor_slider.value,
        }

    def from_config_dict(self, config_dict):
        self.weight_slider.setValue(config_dict["noise_weight"])
        self.pivot_slider.setValue(config_dict["pivot"])
        self.sketch_display.setChecked(config_dict["display_as_sketch"])
        self.noise_settings.from_config_dict(config_dict["noise_settings"])
        self.modal_view.setChecked(config_dict["modal_view"])
        self.temp_selector.points = config_dict["points"]
        self.steepness_slider.setValue(int(config_dict.get("steepness", 2.0) * 10))
        self.factor_slider.setValue(int(config_dict.get("factor", 4.0) * 10))
        self.temp_selector.redraw()


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = TempPanel()
    window.show()
    sys.exit(app.exec_())
