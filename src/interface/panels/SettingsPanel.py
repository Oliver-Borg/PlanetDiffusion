from math import ceil
from PyQt5.QtWidgets import (
    QFrame,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QCheckBox,
    QLabel,
    QRadioButton,
    QColorDialog,
    QButtonGroup,
    QGroupBox,
)
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QColor
from planetAI.src.data.noise_settings import (
    SettingsBase,
    DropDownConfig,
    SliderConfig,
)
from planetAI.src.data.utils import profile
from ..widgets.LabeledSlider import FloatLabeledSlider, IntLabeledSlider


class SettingsPanel(QFrame):
    def __init__(
        self,
        parent=None,
        settings: SettingsBase = None,
        list_updater=lambda: None,
        onchange=lambda: None,
        columns=2,
        onfinish=lambda: None,
    ):
        super().__init__(parent)
        self.settings = settings
        self.list_updater = list_updater
        self.onchange = onchange
        self.onfinish = onfinish
        self.columns = columns
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(4, 4, 4, 4)
        self.main_layout.setSpacing(4)
        self.widget_frame = QFrame(self)
        self.main_layout.addWidget(self.widget_frame)
        self.create_widgets()

    def set_name(self, setter):
        setter(self.name_entry.text())
        self.list_updater()

    @profile
    def on_change(self, setter, value):
        setter(value)
        self.onchange()

    def create_widgets(self):
        _, name, _, _ = self.settings.get_display_name_tuple()
        group_box = QGroupBox(name)
        widget_layout = QVBoxLayout(self.widget_frame)
        widget_layout.setContentsMargins(2, 2, 2, 2)
        widget_layout.setSpacing(2)
        group_box.setLayout(widget_layout)

        # Header frame
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(2, 2, 2, 2)
        header_layout.setSpacing(4)
        widget_layout.addLayout(header_layout)

        # Name entry
        # _, value, setter, _ = self.settings.get_display_name_tuple()
        # self.name_entry = QLineEdit(value)
        # self.name_entry.returnPressed.connect(lambda s=setter: self.set_name(s))
        # header_layout.addWidget(self.name_entry)

        # Simple mode toggle
        self.simple_mode_check = QCheckBox("Simple mode")
        self.simple_mode_check.setChecked(self.settings.simple)
        self.simple_mode_check.stateChanged.connect(
            lambda state: self.set_simple_mode(bool(state))
        )
        header_layout.addWidget(self.simple_mode_check, alignment=Qt.AlignRight)

        # Color button
        # name, value, setter, _ = self.settings.get_display_colour_tuple()
        # self.colour_button = QPushButton(name)
        # self.colour_button.setStyleSheet(f"background-color: {value}")
        # self.colour_button.clicked.connect(
        #     lambda s=setter, v=value: self.choose_colour(s, v)
        # )
        # header_layout.addWidget(self.colour_button)

        # Controls frame with column layout
        self.controls_frame = QFrame(self.widget_frame)
        self.controls_layout = QHBoxLayout(self.controls_frame)
        self.controls_layout.setContentsMargins(2, 2, 2, 2)
        self.controls_layout.setSpacing(2)
        widget_layout.addWidget(self.controls_frame)
        self.create_controls()
        self.main_layout.addWidget(group_box)

    def clear_controls(self):
        # Remove all widgets from controls layout
        while self.controls_layout.count():
            item = self.controls_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def create_controls(self):
        self.clear_controls()

        # Create column layouts
        column_frames = []
        column_layouts = []
        for _ in range(self.columns):
            column_frame = QFrame(self.controls_frame)
            column_layout = QVBoxLayout(column_frame)
            column_layout.setContentsMargins(2, 2, 2, 2)
            column_layout.setSpacing(2)
            self.controls_layout.addWidget(column_frame)
            column_frames.append(column_frame)
            column_layouts.append(column_layout)

        # Distribute controls across columns
        controls = self.settings.get_controls()

        # Calculate rows needed
        num_rows = 0
        for control in controls:
            if isinstance(control, SliderConfig):
                num_rows += 2
            elif isinstance(control, DropDownConfig):
                num_rows += len(control.options) + 1
            else:
                num_rows += 1

        num_rows = max(num_rows, 1)
        rows_per_column = ceil(num_rows / self.columns)
        current_column = 0
        current_row = 0

        for control in controls:
            control_frame = QFrame()
            control_layout = QVBoxLayout(control_frame)
            control_layout.setContentsMargins(2, 2, 2, 2)
            control_layout.setSpacing(2)

            # Calculate how many rows this control will use
            rows_needed = 0
            if isinstance(control, SliderConfig):
                rows_needed = 2
                name, value, setter, min_val, max_val, cast, step = control.get_tuple()
                slider = (
                    FloatLabeledSlider(name, min_val, max_val, value, step)
                    if cast != int
                    else IntLabeledSlider(name, min_val, max_val, value)
                )
                slider.valueChanged.connect(
                    lambda v, s=setter, c=cast: self.on_change(s, c(v))
                )
                slider.sliderReleased.connect(self.onfinish)
                control_layout.addWidget(slider)
            elif isinstance(control, DropDownConfig):
                rows_needed = len(control.options) + 1
                name = control.label
                options = control.options

                # Create label
                label = QLabel(name)
                control_layout.addWidget(label)

                # Create radio button group
                button_group = QButtonGroup(control_frame)

                for i, option in enumerate(options):
                    radio = QRadioButton(option.label)
                    radio.setChecked(i == 0)
                    radio.clicked.connect(
                        lambda checked, s=control.setter, t=control.type_, v=option.value: self.on_change(
                            s, t(v)
                        )
                    )
                    radio.clicked.connect(self.onfinish)
                    button_group.addButton(radio)
                    control_layout.addWidget(radio)
            else:
                rows_needed = 1

            # Check if we need to move to next column
            if (
                current_row + rows_needed > rows_per_column
                and current_column < self.columns - 1
            ):
                current_column += 1
                current_row = 0

            column_layouts[current_column].addWidget(control_frame)
            current_row += rows_needed

    def set_simple_mode(self, value: bool):
        self.settings.simple = value
        self.refresh_widgets()

    def refresh_widgets(self):
        self.create_controls()

    def choose_colour(self, setter, value: str):
        color = QColorDialog.getColor(QColor(value), self)
        if color.isValid():
            color_str = color.name()
            setter(color_str)
            self.colour_button.setStyleSheet(f"background-color: {color_str}")
            self.list_updater()


class SettingListItem(QFrame):
    clicked = pyqtSignal()

    def __init__(self, parent=None, settings: SettingsBase = None, onclick=None):
        super().__init__(parent)
        self.settings = settings
        self.onclick = onclick

        # Set up layout
        layout = QHBoxLayout(self)
        self.setFrameStyle(QFrame.Panel | QFrame.Raised)

        # Create widgets
        _, value, _, _ = self.settings.get_display_colour_tuple()
        self.colour_label = QLabel()
        self.colour_label.setStyleSheet(f"background-color: {value}")
        self.colour_label.setFixedSize(20, 20)
        layout.addWidget(self.colour_label)

        _, name, _, _ = self.settings.get_display_name_tuple()
        self.name_button = QPushButton(name)
        self.name_button.clicked.connect(self.onclick)
        layout.addWidget(self.name_button)
