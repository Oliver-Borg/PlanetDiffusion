from PyQt5.QtWidgets import QFrame, QHBoxLayout, QVBoxLayout, QLabel, QSlider, QLineEdit
from PyQt5.QtCore import Qt, pyqtSignal


class LabeledSlider(QFrame):
    valueChanged = pyqtSignal(float)
    sliderReleased = pyqtSignal()

    def __init__(
        self,
        label: str,
        min_val: float | int,
        max_val: float | int,
        value: float | int,
        step: float | int,
        cast=float,
    ):
        super().__init__()
        self.step = step
        self.cast = cast
        self.min_val = min_val
        self.max_val = max_val

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # Label
        self.label = QLabel(label)
        layout.addWidget(self.label)

        # Slider and value input container
        slider_container = QHBoxLayout()
        layout.addLayout(slider_container)

        # Slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setFixedHeight(16)
        self.slider.setMinimum(int(min_val / step))
        self.slider.setMaximum(int(max_val / step))
        self.slider.setValue(int(value / step))
        self.slider.setPageStep(1)
        self.slider.valueChanged.connect(self._on_slider_change)
        self.slider.sliderReleased.connect(self.sliderReleased)
        slider_container.addWidget(self.slider)

        # Value input
        self.value_input = QLineEdit()
        self.value_input.setFixedWidth(50)
        self.value_input.setText(str(value))
        self.value_input.returnPressed.connect(self._on_input_change)
        self.value_input.editingFinished.connect(self._on_input_change)
        slider_container.addWidget(self.value_input)

    def _on_slider_change(self, value):
        actual_value = self.cast(value * self.step)
        self.value_input.setText(f"{actual_value:.3f}" if isinstance(actual_value, float) else str(actual_value))
        self.valueChanged.emit(actual_value)

    def _validate_and_set_input(self, text_value):
        try:
            value = self.cast(float(text_value))
            value = max(self.min_val, min(self.max_val, value))
            return value
        except ValueError:
            return None

    def _on_input_change(self):
        value = self._validate_and_set_input(self.value_input.text())
        if value is not None:
            self.slider.setValue(int(value / self.step))
            self._on_slider_change(int(value / self.step))
        else:
            self.value_input.setText(str(self.cast(self.slider.value() * self.step)))


class FloatLabeledSlider(LabeledSlider):
    def __init__(self, label: str, min_val: float, max_val: float, value: float, step: float):
        super().__init__(label, min_val, max_val, value, step, float)

    def _validate_and_set_input(self, text_value):
        try:
            value = float(text_value)
            value = max(self.min_val, min(self.max_val, value))
            return value
        except ValueError:
            return None

    @property
    def value(self) -> float:
        return float(self.slider.value() * self.step)

    def setValue(self, value: float):
        self.slider.setValue(int(value / self.step))
        str_val = f"{value:.3f}".rstrip(".0") if isinstance(value, float) else str(value)
        self.value_input.setText(str_val)
        self.valueChanged.emit(value)


class IntLabeledSlider(LabeledSlider):
    def __init__(self, label: str, min_val: int, max_val: int, value: int, step: int = 1):
        super().__init__(label, min_val, max_val, value, step, int)

    def _validate_and_set_input(self, text_value):
        try:
            value = int(text_value)
            value = max(self.min_val, min(self.max_val, value))
            return value
        except ValueError:
            return None

    @property
    def value(self) -> int:
        return int(self.slider.value() * self.step)

    def setValue(self, value: int):
        self.slider.setValue(value)
        self.value_input.setText(str(value))
        self.valueChanged.emit(value)
