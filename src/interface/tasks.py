from typing import Callable
from PyQt5.QtCore import QThread
import time


class _OnceOffThread(QThread):
    def __init__(self, target, args, kwargs):
        super().__init__()
        self.target = target
        self.args = args
        self.kwargs = kwargs

    def run(self):
        if not self.isInterruptionRequested():
            self.target(*self.args, **self.kwargs)


class _RepeatingThread(QThread):
    def __init__(self, target, args, kwargs, interval, stop_condition):
        super().__init__()
        self.target = target
        self.args = args
        self.kwargs = kwargs
        self.interval = interval
        self.stop_condition = stop_condition

    def run(self):
        while not self.stop_condition() and not self.isInterruptionRequested():
            try:
                self.target(*self.args, **self.kwargs)
                time.sleep(self.interval)
            except Exception as e:
                print(f"Error in repeating task: {e}")
                time.sleep(self.interval)


class OnceOffTask:
    def __init__(self, target: Callable, args: list = [], kwargs: dict = {}):
        self.target = target
        self.args = args
        self.kwargs = kwargs
        self.thread = _OnceOffThread(target, args, kwargs)

    def start(self):
        self.thread.start()

    def stop(self):
        if self.thread.isRunning():
            self.thread.requestInterruption()
            self.thread.wait()
        self.thread = _OnceOffThread(self.target, self.args, self.kwargs)

    def __del__(self):
        self.stop()


class RepeatingTask:
    def __init__(
        self,
        target: Callable,
        args: list = [],
        kwargs: dict = {},
        interval: float = 0.1,
        stop_condition: Callable = lambda: False,
    ):
        self.target = target
        self.args = args
        self.kwargs = kwargs
        self.interval = interval
        self.stop_condition = stop_condition
        self.thread = _RepeatingThread(target, args, kwargs, interval, stop_condition)

    def start(self):
        self.thread.start()

    def stop(self):
        if self.thread.isRunning():
            self.stop_condition = lambda: True
            self.thread.requestInterruption()
            self.thread.wait()
        self.thread = _RepeatingThread(
            self.target, self.args, self.kwargs, self.interval, self.stop_condition
        )

    def __del__(self):
        self.stop()
