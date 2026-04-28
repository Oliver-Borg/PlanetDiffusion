from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from contextlib import contextmanager


@contextmanager
def wait_cursor():
    try:
        QApplication.setOverrideCursor(Qt.WaitCursor)
        yield
    finally:
        QApplication.restoreOverrideCursor()
