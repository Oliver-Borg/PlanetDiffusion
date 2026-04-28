import sys
import os
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QGridLayout,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QComboBox,
    QCheckBox,
    QPushButton,
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QPoint

from planetAI.src.data.uncertainty_sketch import UncertaintySketcher
from planetAI.src.data.utils import PlanetConfig, hex_to_rgb, np_hex_to_rgb, np_rgb_to_hex, rgb_to_hex


class HoverPreview(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.ToolTip | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setStyleSheet("border: 2px solid #ffffff; background: #1a1a1a;")
        self.setScaledContents(True)

    def set_image(self, path, width=600):
        if not path or not os.path.exists(path):
            return
        pixmap = QPixmap(path)
        ratio = pixmap.height() / pixmap.width()
        self.setPixmap(pixmap)
        self.setFixedSize(width, int(width * ratio))


class DataItemWidget(QFrame):
    def __init__(self, batch_files, sketcher, parent=None):
        super().__init__(parent)
        self.sketcher, self.files = sketcher, batch_files
        self.setFixedSize(190, 220)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)

        self.combined_val = (self.files["l"] * 256**2) + (self.files["t"] * 256) + self.files["d"]
        self.rmse = self.calculate_rmse_loss(self.files["sat_real"], self.files["sat_output"])

        self.preview = None  # Initialized lazily
        self.init_ui()

    def init_ui(self):
        self.layout = QVBoxLayout(self)
        self.meta_layout = QHBoxLayout()

        # Uncertainty Tile
        uncert_mask = self.sketcher.get_uncertainty_sketch(np.array([[self.combined_val]]))
        r, g, b = uncert_mask[0, 0]
        self.uncert_tile = QLabel("UNCERT")
        self.uncert_tile.setStyleSheet(
            f"background-color: rgb({r},{g},{b}); color: white; font-weight: bold; border: 1px solid #444; border-radius: 4px;"
        )
        self.uncert_tile.setFixedSize(80, 45)
        self.uncert_tile.setAlignment(Qt.AlignCenter)

        # RMSE Tile
        h_int = min(int(self.rmse), 255)
        self.loss_tile = QLabel(f"RMSE\n{self.rmse:.2f}")
        self.loss_tile.setStyleSheet(
            f"background-color: rgb({h_int},{255-h_int},50); color: black; font-weight: bold; font-size: 10px; border: 1px solid #444; border-radius: 4px;"
        )
        self.loss_tile.setFixedSize(80, 45)
        self.loss_tile.setAlignment(Qt.AlignCenter)

        self.meta_layout.addWidget(self.uncert_tile)
        self.meta_layout.addWidget(self.loss_tile)
        self.layout.addLayout(self.meta_layout)

        # Placeholders for Lazy Loading
        self.img_labels = {}
        self.img_layout = QHBoxLayout()
        for key in ["sat_output", "sat_real"]:
            lbl = QLabel("...")  # Placeholder
            lbl.setFixedSize(80, 80)
            lbl.setAlignment(Qt.AlignCenter)
            self.img_labels[key] = lbl
            self.img_layout.addWidget(lbl)

        self.layout.addLayout(self.img_layout)
        self.layout.addWidget(QLabel(f"Batch {self.files['batch']}", alignment=Qt.AlignCenter))

    def load_images_lazily(self):
        """Called only when the page containing this widget is displayed."""
        for key, lbl in self.img_labels.items():
            if key in self.files and lbl.text() == "...":
                pix = QPixmap(self.files[key]).scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                lbl.setPixmap(pix)
                lbl.setText("")  # Clear placeholder

    def get_mode(self, im: np.ndarray) -> np.ndarray:
        im = np_rgb_to_hex(im)
        values, counts = np.unique(im, return_counts=True)
        max_count = counts.argmax()
        return np_hex_to_rgb(values[max_count])

    def calculate_rmse_loss(self, path_a, path_b):
        try:
            # TODO Consider improving this loss
            img1 = np.array(Image.open(path_a), dtype=np.uint8)
            img2 = np.array(Image.open(path_b), dtype=np.uint8)

            mode_colour_1 = self.get_mode(img1)
            mode_colour_2 = self.get_mode(img2)
            return np.sqrt(np.mean((mode_colour_1 - mode_colour_2) ** 2))
        except Exception:
            return 0.0

    def enterEvent(self, event):
        if not self.preview:
            self.preview = HoverPreview()
            self.preview.set_image(self.files.get("display"))

        cursor_pos = self.mapToGlobal(QPoint(self.width() + 10, 0))
        screen_geo = QApplication.primaryScreen().availableGeometry()
        if cursor_pos.x() + self.preview.width() > screen_geo.right():
            cursor_pos.setX(self.mapToGlobal(QPoint(0, 0)).x() - self.preview.width() - 10)
        if cursor_pos.y() + self.preview.height() > screen_geo.bottom():
            cursor_pos.setY(screen_geo.bottom() - self.preview.height() - 10)

        self.preview.move(cursor_pos)
        self.preview.show()
        self.preview.raise_()

    def leaveEvent(self, event):
        if self.preview:
            self.preview.hide()


class PlanetExplorer(QWidget):
    def __init__(self, data_dir):
        super().__init__()
        self.setWindowTitle("PlanetAI Paginated Explorer")
        self.resize(1100, 950)
        self.setStyleSheet("background-color: #2b2b2b; color: #ffffff;")

        self.config = PlanetConfig()
        self.sketcher = UncertaintySketcher(self.config)
        self.data_dir = data_dir

        self.all_data_items = []
        self.items_per_page = 20  # 4 rows of 5
        self.current_page = 0

        self.init_ui()
        self.load_data()

    def init_ui(self):
        self.main_layout = QVBoxLayout(self)

        # Top Controls
        ctrl_layout = QHBoxLayout()
        self.sort_box = QComboBox()
        self.sort_box.addItems(["Sort by: Batch ID", "Sort by: RMSE", "Sort by: Uncertainty"])
        self.sort_box.currentIndexChanged.connect(self.refresh_logic)

        self.unique_check = QCheckBox("Unique Colors")
        self.unique_check.stateChanged.connect(self.refresh_logic)

        ctrl_layout.addWidget(self.sort_box)
        ctrl_layout.addWidget(self.unique_check)
        ctrl_layout.addStretch()
        self.main_layout.addLayout(ctrl_layout)

        # Grid Area
        self.grid = QGridLayout()
        self.grid.setSpacing(10)
        self.main_layout.addLayout(self.grid)
        self.main_layout.addStretch()

        # Pagination Controls
        pager_layout = QHBoxLayout()
        self.prev_btn = QPushButton("Previous")
        self.prev_btn.clicked.connect(lambda: self.change_page(-1))
        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(lambda: self.change_page(1))
        self.page_label = QLabel("Page 1 of 1")

        pager_layout.addStretch()
        pager_layout.addWidget(self.prev_btn)
        pager_layout.addWidget(self.page_label)
        pager_layout.addWidget(self.next_btn)
        pager_layout.addStretch()
        self.main_layout.addLayout(pager_layout)

    def load_data(self):
        batches = self.parse_directory()
        self.all_data_items = [
            v for k, v in batches.items() if all(x in v for x in ["sat_real", "sat_output", "display"])
        ]
        self.refresh_logic()

    def refresh_logic(self):
        self.current_page = 0
        self.update_display()

    def change_page(self, delta):
        self.current_page += delta
        self.update_display()

    def update_display(self):
        # 1. Filter and Sort
        items = self.all_data_items
        if self.unique_check.isChecked():
            seen, unique = set(), []
            for it in items:
                val = (it["l"] * 256**2) + (it["t"] * 256) + it["d"]
                if val not in seen:
                    unique.append(it)
                    seen.add(val)
            items = unique

        idx = self.sort_box.currentIndex()
        if idx == 1:  # RMSE
            items = sorted(items, key=lambda x: self.get_cached_rmse(x), reverse=True)
        elif idx == 2:  # Uncertainty
            items = sorted(items, key=lambda x: (x["l"] * 256**2) + (x["t"] * 256) + x["d"])

        # 2. Paginate
        max_pages = max(1, (len(items) + self.items_per_page - 1) // self.items_per_page)
        self.current_page = max(0, min(self.current_page, max_pages - 1))

        start = self.current_page * self.items_per_page
        end = start + self.items_per_page
        page_items = items[start:end]

        # 3. Clear and Rebuild Grid
        for i in reversed(range(self.grid.count())):
            self.grid.itemAt(i).widget().setParent(None)

        cols = 5
        for i, item_data in enumerate(page_items):
            widget = DataItemWidget(item_data, self.sketcher)
            self.grid.addWidget(widget, i // cols, i % cols)
            widget.load_images_lazily()  # LAZY LOADING TRIGGER

        self.page_label.setText(f"Page {self.current_page + 1} of {max_pages}")
        self.prev_btn.setEnabled(self.current_page > 0)
        self.next_btn.setEnabled(self.current_page < max_pages - 1)

    def get_cached_rmse(self, item):
        # Small helper to calculate RMSE for sorting without creating full widgets
        if "rmse" not in item:
            try:
                img1 = np.array(Image.open(item["sat_real"]).convert("L"), dtype=np.float32)
                img2 = np.array(Image.open(item["sat_output"]).convert("L"), dtype=np.float32)
                item["rmse"] = np.sqrt(np.mean((img1 - img2) ** 2))
            except:
                item["rmse"] = 0.0
        return item["rmse"]

    def parse_directory(self):
        batches = {}
        if not os.path.exists(self.data_dir):
            return {}
        for f in os.listdir(self.data_dir):
            if not f.endswith(".png"):
                continue
            parts = f.split("_")
            if len(parts) < 6:
                continue
            batch_id = f"{parts[0]}_{parts[1]}"
            if batch_id not in batches:
                batches[batch_id] = {"l": int(parts[2]), "t": int(parts[3]), "d": int(parts[4]), "batch": parts[1]}
            p = os.path.join(self.data_dir, f)
            if "sat_real" in f:
                batches[batch_id]["sat_real"] = p
            elif "sat_output" in f:
                batches[batch_id]["sat_output"] = p
            elif "dem_output" in f:
                batches[batch_id]["dem_output"] = p
            elif "display" in f:
                batches[batch_id]["display"] = p
        return batches


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # gui = PlanetExplorer(data_dir="/mnt/e/evaluation/artifacts/Old-palette_inference:v0")
    gui = PlanetExplorer(data_dir="/mnt/e/evaluation/artifacts/Control-palette_inference:v0")
    gui.show()
    sys.exit(app.exec_())
