from PyQt5.QtWidgets import (
    QWidget,
    QCheckBox,
    QVBoxLayout,
)

from .interface_types import landcover_display_func
from .panels.GlobePanel import GlobePanel
from .atlas_storage import AtlasStorage
from .interface_types import AtlasType
from .widgets.LabeledSlider import FloatLabeledSlider, IntLabeledSlider


class TempIconPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        atlas_storage = AtlasStorage()
        atlas_storage._load_fast_earth_data('./planetAI/data')

        self.globe_panel = GlobePanel(self)

        # Create sliders and checkbox
        # TODO Change to int sliders for icon size and icon spacing
        self.icon_size_slider = IntLabeledSlider("Icon size", 0, 100, 20)
        self.icon_size_slider.valueChanged.connect(self.update_postprocess)

        self.icon_spacing_slider = IntLabeledSlider("Icon spacing", 0, 100, 20)
        self.icon_spacing_slider.valueChanged.connect(self.update_postprocess)

        self.shading_slider = FloatLabeledSlider("Shading factor", 0.0, 1.0, 0.8, 0.05)
        self.shading_slider.valueChanged.connect(self.update_postprocess)

        self.brightness_slider = FloatLabeledSlider("Icon brightness", -1.0, 1.0, 0.2, 0.05)
        self.brightness_slider.valueChanged.connect(self.update_postprocess)

        self.outlines_checkbox = QCheckBox("Show outlines")
        self.outlines_checkbox.toggled.connect(self.update_postprocess)

        self.use_components_checkbox = QCheckBox("Use components")
        self.use_components_checkbox.toggled.connect(self.update_postprocess)

        self.use_lines_checkbox = QCheckBox("Use lines")
        self.use_lines_checkbox.toggled.connect(self.update_postprocess)

        self.use_circles_checkbox = QCheckBox("Use circles")
        self.use_circles_checkbox.toggled.connect(self.update_postprocess)

        # Layout setup
        layout = QVBoxLayout()
        layout.addWidget(self.globe_panel)
        layout.addWidget(self.icon_size_slider)
        layout.addWidget(self.icon_spacing_slider)
        layout.addWidget(self.shading_slider)
        layout.addWidget(self.brightness_slider)
        layout.addWidget(self.outlines_checkbox)
        layout.addWidget(self.use_components_checkbox)
        layout.addWidget(self.use_lines_checkbox)
        layout.addWidget(self.use_circles_checkbox)

        self.setLayout(layout)

        # Set up globe panel with atlas and display function
        self.set_postprocess()
        self.globe_panel.set_atlas(atlas_storage.get(AtlasType.EARTH_LANDCOVER_SKETCH))

    def update_postprocess(self):
        self.set_postprocess()
        self.globe_panel.display_preview()

    def set_postprocess(self):
        self.globe_panel.post_process_function = lambda x: landcover_display_func(
            x,
            self.icon_size_slider.value,
            self.icon_spacing_slider.value,
            self.outlines_checkbox.isChecked(),
            self.shading_slider.value,
            self.use_components_checkbox.isChecked(),
            self.use_lines_checkbox.isChecked(),
            self.brightness_slider.value,
            self.use_circles_checkbox.isChecked(),
        )


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication

    app = QApplication([])
    temp_icon_panel = TempIconPanel()
    temp_icon_panel.show()
    app.exec_()
