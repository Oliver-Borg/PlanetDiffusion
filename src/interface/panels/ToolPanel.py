from PyQt5.QtWidgets import (
    QWidget,
    QRadioButton,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QCheckBox,
    QPushButton,
    QMessageBox,
    QGroupBox,
)
from ..interface_types import ToolEnum
from ..widgets.LabeledSlider import IntLabeledSlider, FloatLabeledSlider
from typing import Callable


class ToolPanel(QWidget):
    def __init__(
        self,
        parent=None,
        callback: Callable = lambda: None,
        confirm_callback: Callable | None = None,
        cancel_callback: Callable | None = None,
        reset_callback: Callable | None = None,
    ):
        super().__init__(parent)
        self.tool_var = ToolEnum.BRUSH.value
        self.callback = callback
        self.confirm_callback = confirm_callback
        self.cancel_callback = cancel_callback
        self.reset_callback = reset_callback
        self.create_widgets()

    def create_widgets(self):
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        group_box = QGroupBox("Sketching Tools")
        group_layout = QVBoxLayout()
        group_box.setLayout(group_layout)
        main_layout.addWidget(group_box)

        panel_layout = QHBoxLayout()
        group_layout.addLayout(panel_layout)

        # Tool buttons
        self.tool_frame = QFrame()
        tool_layout = QVBoxLayout()
        self.tool_frame.setLayout(tool_layout)

        self.brush_button = QRadioButton("Brush")
        self.fill_button = QRadioButton("Fill")
        self.lasso_button = QRadioButton("Lasso")
        self.brush_button.setChecked(True)

        for button in [self.brush_button, self.fill_button, self.lasso_button]:
            button.clicked.connect(self.draw_tool_options)
            tool_layout.addWidget(button)

        panel_layout.addWidget(self.tool_frame)

        # Tool options
        self.tool_options_frame = QFrame()
        options_layout = QVBoxLayout()
        self.tool_options_frame.setLayout(options_layout)
        panel_layout.addWidget(self.tool_options_frame)

        self.all_tool_options: dict[
            str, IntLabeledSlider | FloatLabeledSlider | QCheckBox
        ] = {
            "brush_size": IntLabeledSlider("Brush Size", 0, 15, 0),
            "roughness": FloatLabeledSlider("Roughness", 0, 2, 0.0, 0.05),
            "max_difference": IntLabeledSlider("Max Difference", 0, 255, 0),
            "erase": QCheckBox("Erase"),
        }

        for control in self.all_tool_options.values():
            if isinstance(control, QCheckBox):
                control.toggled.connect(self.callback)
            else:
                control.valueChanged.connect(self.callback)

        self.tool_sub_options = {
            # TODO Make roughness work well and then re-add it here "roughness"
            ToolEnum.BRUSH: ["brush_size", "erase"],
            ToolEnum.FILL: ["max_difference"],
            ToolEnum.LASSO: ["erase"],
        }

        # Buttons
        button_layout = QHBoxLayout()
        if self.confirm_callback is not None:
            confirm_button = QPushButton("Confirm")
            confirm_button.clicked.connect(self.confirm_callback)
            button_layout.addWidget(confirm_button)

        if self.cancel_callback is not None:
            cancel_button = QPushButton("Cancel")
            cancel_button.clicked.connect(self.cancel_callback)
            button_layout.addWidget(cancel_button)

        if self.reset_callback is not None:
            reset_button = QPushButton("Reset")
            reset_button.clicked.connect(self.reset)
            button_layout.addWidget(reset_button)

        group_layout.addLayout(button_layout)

        self.draw_tool_options()

    def reset(self):
        if (
            QMessageBox.question(
                self,
                "Reset Sketch",
                "Are you sure to reset the sketch?",
                QMessageBox.Yes | QMessageBox.No,
            )
            == QMessageBox.Yes
        ):
            self.reset_callback()

    @property
    def is_brush(self):
        return self.tool_var == ToolEnum.BRUSH.value

    @property
    def is_fill(self):
        return self.tool_var == ToolEnum.FILL.value

    @property
    def is_lasso(self):
        return self.tool_var == ToolEnum.LASSO.value

    @property
    def brush_size(self):
        return self.all_tool_options["brush_size"].value if self.is_brush else 0

    @property
    def roughness(self):
        return self.all_tool_options["roughness"].value if self.is_brush else 0

    @property
    def max_difference(self):
        return self.all_tool_options["max_difference"].value

    @property
    def erase(self):
        return self.all_tool_options["erase"].isChecked() and not self.is_fill

    def draw_tool_options(self):
        for widget in self.all_tool_options.values():
            widget.hide()

        if self.brush_button.isChecked():
            self.tool_var = ToolEnum.BRUSH.value
        elif self.fill_button.isChecked():
            self.tool_var = ToolEnum.FILL.value
        elif self.lasso_button.isChecked():
            self.tool_var = ToolEnum.LASSO.value

        tool_enum = ToolEnum(self.tool_var)
        for widget_name in self.tool_sub_options[tool_enum]:
            widget = self.all_tool_options[widget_name]
            widget.show()
            self.tool_options_frame.layout().addWidget(widget)
