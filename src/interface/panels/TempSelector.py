from PyQt5.QtWidgets import QWidget, QGroupBox
from PyQt5.QtGui import QPainter
from PyQt5.QtCore import Qt, QPoint, pyqtSignal
from typing import Callable, List, Tuple


class TempCanvas(QWidget):
    pointMoved = pyqtSignal()
    mouseMoveSignal = pyqtSignal(QPoint)

    def __init__(self, parent=None, width=512, height=256):
        super().__init__(parent)
        self.setMinimumSize(width, height)
        self.points = []
        self.selected_index: int = None
        self.padding = 10

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw lines
        if len(self.points) > 1:
            for i in range(len(self.points) - 1):
                painter.drawLine(self.points[i], self.points[i + 1])

        # Draw points
        for i, point in enumerate(self.points):
            if i == self.selected_index:
                painter.setBrush(Qt.gray)
            else:
                painter.setBrush(Qt.black)
            painter.drawEllipse(point, 5, 5)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = event.pos()
            selected_index = None
            min_dist = float("inf")

            for i, point in enumerate(self.points):
                dist = (point.x() - pos.x()) ** 2 + (point.y() - pos.y()) ** 2
                if dist < min_dist:
                    min_dist = dist
                    selected_index = i

            if min_dist < 100:
                self.selected_index = selected_index
                self.update()
            else:
                self.selected_index = None

    def mouseMoveEvent(self, event):
        # TODO Add a different event that catches a non-click mouse move and just emit the signal
        self.mouseMoveSignal.emit(event.pos())
        if self.selected_index is not None:
            new_x = max(self.padding, min(self.width() - self.padding, event.x()))
            new_y = max(self.padding, min(self.height() - self.padding, event.y()))
            idx = self.selected_index
            self.points[idx] = QPoint(new_x, new_y)
            self.update()
            self.pointMoved.emit()

    def mouseReleaseEvent(self, event):
        if self.selected_index is not None:
            self.selected_index = None
            self.points.sort(key=lambda p: p.x())
            self.points[0] = QPoint(self.padding, self.points[0].y())
            self.points[-1] = QPoint(self.width() - self.padding, self.points[-1].y())
            self.update()
            self.pointMoved.emit()


class TempSelector(QWidget):
    def __init__(
        self,
        parent=None,
        def_num_latitudes=9,
        max_temp=37,
        min_temp=-45,
        on_change: Callable = lambda: None,
        update_labels: Callable = lambda point: None,
    ):
        super().__init__(parent)
        self.def_num_latitudes = def_num_latitudes
        self.max_temp = max_temp
        self.min_temp = min_temp
        self.callback = on_change
        self.update_labels = update_labels
        group_box = QGroupBox("Temperature Selector", self)
        self.canvas = TempCanvas(group_box)
        self.canvas.pointMoved.connect(self.callback)
        self.canvas.mouseMoveSignal.connect(self.handle_mouse_move)

    def handle_mouse_move(self, point):
        self.update_labels(point)

    def redraw(self):
        self.canvas.update()
        self.callback()

    @property
    def points(self) -> List[Tuple[float, float]]:
        return [(p.x(), p.y()) for p in self.canvas.points]

    @points.setter
    def points(self, value):
        self.canvas.points = [QPoint(x, y) for x, y in value]
        self.redraw()

    def coords_to_temp(self, x, y) -> tuple[float, float]:
        latitude = (
            180
            * (x - self.canvas.padding)
            / (self.canvas.width() - 2 * self.canvas.padding)
            - 90
        )
        temperature = self.max_temp - (y - self.canvas.padding) * (
            self.max_temp - self.min_temp
        ) / (self.canvas.height() - 2 * self.canvas.padding)
        return latitude, temperature

    def temp_to_coords(self, latitude, temperature):
        x = (
            int((self.canvas.width() - 2 * self.canvas.padding) * (latitude + 90) / 180)
            + self.canvas.padding
        )
        y = (
            int(
                (self.canvas.height() - 2 * self.canvas.padding)
                * (self.max_temp - temperature)
                / (self.max_temp - self.min_temp)
            )
            + self.canvas.padding
        )
        return x, y

    def set_defaults(self):
        points = []
        for i in range(self.def_num_latitudes):
            latitude = (i * 180 / (self.def_num_latitudes - 1)) - 90
            temperature = (
                -abs(latitude) / 90 * (self.max_temp - self.min_temp) + self.max_temp
            )
            x, y = self.temp_to_coords(latitude, temperature)
            points.append((x, y))
        self.points = points

    def set_values(self, lat_temp_list):
        points = []
        for latitude, temperature in lat_temp_list:
            x, y = self.temp_to_coords(latitude, temperature)
            points.append((x, y))
        self.points = points


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = TempSelector()
    window.set_defaults()
    window.show()
    sys.exit(app.exec_())
