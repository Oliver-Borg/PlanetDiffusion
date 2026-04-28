from dataclasses import Field
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QCheckBox, QLineEdit, QFrame
from typing import ClassVar, Dict, Protocol, Any, Type, TypeVar
from ..widgets.LabeledSlider import FloatLabeledSlider, IntLabeledSlider

T = TypeVar("T")


class IsDataclass(Protocol):
    __dataclass_fields__: ClassVar[Dict[str, Any]]


class DataClassPanel(QFrame):
    NAME = "dataclass_panel"

    def __init__(self, parent=None, data_class: IsDataclass = None, title: str = ""):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.data_class = data_class
        self.title = title
        self.main_layout = QVBoxLayout(self)
        self.create_widgets()

    def create_widgets(self):
        self.main_layout.addWidget(QLabel(self.title))

        for field_name in self.data_class.__dataclass_fields__:
            field: Field = self.data_class.__dataclass_fields__[field_name]
            value = getattr(self.data_class, field_name)

            if field.type in [int, float]:
                if value is None:
                    value = 0
                _min = field.metadata.get("min", value / 2)
                _max = field.metadata.get("max", value * 2)
                _step = field.metadata.get("step", 1)

            mutable = field.metadata.get("mutable", False)
            label_text = field_name.replace("_", " ").title()
            if not mutable:
                continue

            if field.type == bool:
                widget = QCheckBox(label_text)
                widget.setChecked(value)
                widget.stateChanged.connect(
                    lambda state, fn=field_name, ft=field.type: self.set_class_value(
                        fn, bool(state), ft
                    )
                )

            elif field.type == int:
                widget = IntLabeledSlider(label_text, _min, _max, value, _step)
                widget.valueChanged.connect(
                    lambda val, fn=field_name, ft=field.type: self.set_class_value(
                        fn, val, ft
                    )
                )

            elif field.type == float:
                widget = FloatLabeledSlider(label_text, _min, _max, value, _step)
                widget.valueChanged.connect(
                    lambda val, fn=field_name, ft=field.type: self.set_class_value(
                        fn, val, ft
                    )
                )

            elif field.type == str:
                widget = QWidget()
                widget_layout = QVBoxLayout(widget)
                widget_layout.addWidget(QLabel(label_text))
                line_edit = QLineEdit(value)
                line_edit.returnPressed.connect(
                    lambda le=line_edit, fn=field_name, ft=field.type: self.set_class_value(
                        fn, le.text(), ft
                    )
                )
                widget_layout.addWidget(line_edit)

            self.main_layout.addWidget(widget)

    def set_class_value(self, key: str, value: T, value_type: Type[T]) -> None:
        setattr(self.data_class, key, value_type(value))
