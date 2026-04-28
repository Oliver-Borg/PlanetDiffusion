from PyQt5.QtWidgets import (
    QWidget,
    QPushButton,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QGroupBox,
)
from PyQt5.QtCore import Qt
from typing import Callable

from planetAI.src.data.utils import rgb_to_hex_str
from ..interface_types import LandcoverClass
from .LandcoverSubclassRow import LandcoverSubclassRow


class LandCoverRow(QWidget):
    def __init__(
        self,
        parent=None,
        landcover_classes: list[LandcoverClass] = [],
        on_change_callback: Callable = lambda: None,
    ):
        super().__init__(parent)
        self.callback = on_change_callback
        self.landcover_classes = landcover_classes
        self.length = len(landcover_classes)
        self.max_subclasses = max(
            [len(landcover_class.subclasses) for landcover_class in landcover_classes]
            + [0]
        )
        self.primary_class: LandcoverClass = None
        self.secondary_class: LandcoverClass = None
        self.create_widgets()

    def create_widgets(self):
        group_box = QGroupBox("Landcover Class Selector")
        group_box_layout = QVBoxLayout(group_box)
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Landcover button frame
        self.landcover_class_frame = QFrame()
        button_layout = QHBoxLayout()
        self.landcover_class_frame.setLayout(button_layout)

        self.landcover_buttons: list[QPushButton] = []
        for i, landcover_class in enumerate(self.landcover_classes):
            button = QPushButton(landcover_class.name)
            button.setStyleSheet(
                f"background-color: {landcover_class.displaycolour}; "
                f"color: {landcover_class.textcolour};"
            )
            button.clicked.connect(
                lambda checked, lc=landcover_class, col=i: self.select_primary_class(
                    lc, col
                )
            )
            button.setContextMenuPolicy(Qt.CustomContextMenu)
            button.customContextMenuRequested.connect(
                lambda pos, lc=landcover_class, col=i: self.select_secondary_class(
                    lc, col
                )
            )
            button_layout.addWidget(button)
            self.landcover_buttons.append(button)

        # Add a container for the subclass frame
        self.subclass_container = QFrame()
        subclass_layout = QVBoxLayout()
        self.subclass_container.setLayout(subclass_layout)

        # Colour display frame
        self.colour_display_frame = QFrame()
        display_layout = QHBoxLayout()
        self.colour_display_frame.setLayout(display_layout)

        # Primary class row
        primary_row = QHBoxLayout()
        self.primary_class_label = QLabel("Primary Class")
        self.primary_class_display = QLabel()
        self.primary_class_name_label = QLabel("None")
        for widget in [
            self.primary_class_label,
            self.primary_class_display,
            self.primary_class_name_label,
        ]:
            primary_row.addWidget(widget)

        # Secondary class row
        secondary_row = QHBoxLayout()
        self.secondary_class_label = QLabel("Secondary Class")
        self.secondary_class_display = QLabel()
        self.secondary_class_name_label = QLabel("None")
        for widget in [
            self.secondary_class_label,
            self.secondary_class_display,
            self.secondary_class_name_label,
        ]:
            secondary_row.addWidget(widget)

        # Temperature row
        temp_row = QHBoxLayout()
        self.temp_label = QLabel("Temperature")
        self.temp_display = QLabel()
        self.temp_name_label = QLabel("None")
        for widget in [self.temp_label, self.temp_display, self.temp_name_label]:
            temp_row.addWidget(widget)

        # Configure displays
        for display in [
            self.primary_class_display,
            self.secondary_class_display,
            self.temp_display,
        ]:
            display.setMinimumWidth(80)
            display.setStyleSheet("background-color: black;")

        display_layout.addLayout(primary_row)
        display_layout.addLayout(secondary_row)
        display_layout.addLayout(temp_row)

        group_box_layout.addWidget(self.landcover_class_frame)
        group_box_layout.addWidget(self.subclass_container)
        group_box_layout.addWidget(self.colour_display_frame)
        main_layout.addWidget(group_box)

    def remove_subclasses(self):
        """
        Remove the subclasses for the LandcoverPanel class.
        """
        if hasattr(self, "landcover_subclass_frame"):
            self.landcover_subclass_frame.deselect()
            self.landcover_subclass_frame.hide()
            self.landcover_subclass_frame.deleteLater()
        for button in self.landcover_buttons:
            button.setDown(False)

    def show_subclasses(self, landcover_class: LandcoverClass, col: int):
        """
        Show the subclasses for the LandcoverPanel class.
        """
        if landcover_class is None:
            return
        self.remove_subclasses()
        self.landcover_subclass_frame = LandcoverSubclassRow(
            self.subclass_container,
            landcover_class.subclasses,
            select_callback=self.set_colour_displays,
        )
        self.subclass_container.layout().addWidget(self.landcover_subclass_frame)
        self.landcover_subclass_frame.show()

    @property
    def primary_subclass(self) -> LandcoverClass:
        return self.landcover_subclass_frame.get_selected()

    def set_colour_displays(self):
        """
        Set the colour displays for the LandcoverPanel class.
        """
        if self.primary_class is not None:
            self.primary_class_display.setStyleSheet(
                f"background-color: {self.primary_class.displaycolour};"
            )
            self.primary_class_name_label.setText(self.primary_class.name)

        if self.secondary_class is not None:
            self.secondary_class_display.setStyleSheet(
                f"background-color: {self.secondary_class.displaycolour};"
            )
            self.secondary_class_name_label.setText(self.secondary_class.name)

        if self.primary_subclass is not None:
            self.temp_name_label.setText(self.primary_subclass.name)
            self.temp_display.setStyleSheet(
                f"background-color: {rgb_to_hex_str([self.primary_subclass.colour] * 3)};"
            )
            subclass_index = self.primary_class.subclasses.index(self.primary_subclass)
            self.primary_class_display.setStyleSheet(
                f"background-color: {self.primary_class.subclasses[subclass_index].displaycolour};"
            )
            if self.secondary_class is not None:
                self.secondary_class_display.setStyleSheet(
                    f"background-color: {self.secondary_class.subclasses[subclass_index].displaycolour};"
                )
        else:
            self.temp_name_label.setText("None")
            self.temp_display.setStyleSheet("background-color: black;")

        if self.secondary_class is None:
            self.secondary_class_display.setStyleSheet("background-color: black;")
            self.secondary_class_name_label.setText("None")

        self.callback()

    def select_primary_class(self, landcover_class: LandcoverClass, col: int):
        """
        Select the landcover class for the LandcoverPanel class.
        """
        self.primary_class = landcover_class
        self.secondary_class = None
        self.show_subclasses(landcover_class, col)
        for i, button in enumerate(self.landcover_buttons):
            button.setDown(i == col)
        self.set_colour_displays()

    def select_secondary_class(self, landcover_class: LandcoverClass, col: int):
        """
        Select the secondary class for the LandcoverPanel class.
        """
        self.secondary_class = landcover_class
        self.landcover_buttons[col].setDown(True)
        self.set_colour_displays()
