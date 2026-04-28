from PyQt5.QtWidgets import QWidget, QPushButton, QHBoxLayout
from ..interface_types import LandcoverClass


class LandcoverSubclassRow(QWidget):
    """
    This is a class for the landcover subclass row.
    """

    def __init__(
        self,
        parent=None,
        landcover_subclasses: list[LandcoverClass] = [],
        select_callback=lambda: None,
    ):
        """
        Constructor for the LandcoverSubclassRow class.
        """
        super().__init__(parent)
        self.landcover_subclasses = landcover_subclasses
        self.selected_landcover_class = None
        self.select_callback = select_callback
        self.create_widgets()

    def create_widgets(self):
        """
        Create the widgets for the LandcoverSubclassRow class.
        """
        layout = QHBoxLayout()
        self.buttons: list[QPushButton] = []

        for i, landcover_subclass in enumerate(self.landcover_subclasses):
            button = QPushButton(landcover_subclass.name)
            button.setStyleSheet(
                f"background-color: {landcover_subclass.displaycolour}; "
                f"color: {landcover_subclass.textcolour};"
            )
            button.clicked.connect(
                lambda checked, lsc=landcover_subclass, col=i: self.set_selected(
                    lsc, col
                )
            )
            layout.addWidget(button)
            self.buttons.append(button)

        self.setLayout(layout)

    def set_selected(self, landcover_subclass, col):
        """
        Set the selected landcover subclass.
        """
        self.selected_landcover_class = landcover_subclass
        for i, button in enumerate(self.buttons):
            if i == col:
                button.setDown(True)
            else:
                button.setDown(False)
        self.select_callback()

    def get_selected(self) -> LandcoverClass:
        """
        Get the selected landcover subclass.
        """
        return self.selected_landcover_class

    def deselect(self):
        """
        Deselect the landcover subclass.
        TODO Consider having a None subclass instead
        """
        self.selected_landcover_class = None
        for button in self.buttons:
            button.setDown(False)
        self.select_callback()
