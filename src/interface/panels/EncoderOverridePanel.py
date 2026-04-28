from PyQt5.QtWidgets import (
    QFrame,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QButtonGroup,
    QWidget,
)
from planetAI.src.data.dataset import EncoderOverride


class EncoderOverridePanel(QFrame):
    def __init__(self, parent=None, encoder_override: EncoderOverride = None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.encoder_override = encoder_override
        self.options = [None, False, True]
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setSpacing(2)
        self.main_layout.setContentsMargins(4, 4, 4, 4)
        self.button_groups = {}  # Store button groups for each field
        self.create_widgets()

    def create_widgets(self):
        self.main_layout.addWidget(QLabel("Encoder Overrides"))

        for key in self.encoder_override.__dict__:
            row_widget = QWidget()
            row_layout = QVBoxLayout(row_widget)
            row_layout.setSpacing(1)
            row_layout.setContentsMargins(0, 0, 0, 0)

            # Label
            label = QLabel(key.replace("_", " ").title())
            row_layout.addWidget(label)

            # Radio buttons
            button_container = QWidget()
            button_layout = QHBoxLayout(button_container)
            button_layout.setSpacing(4)
            button_layout.setContentsMargins(0, 0, 0, 0)
            button_group = QButtonGroup(self)
            self.button_groups[key] = button_group

            for i, val in enumerate(self.options):
                button = QRadioButton(str(val))
                button_group.addButton(button, i)
                button_layout.addWidget(button)
                if i == 0:  # Set first option (None) as default
                    button.setChecked(True)

            button_group.buttonClicked.connect(
                lambda _, k=key: self.set_class_value(
                    k, self.button_groups[k].checkedId()
                )
            )

            row_layout.addWidget(button_container)
            self.main_layout.addWidget(row_widget)

    def set_class_value(self, key: str, value: int) -> None:
        setattr(self.encoder_override, key, self.options[value])
