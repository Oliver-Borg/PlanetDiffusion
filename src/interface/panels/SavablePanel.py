import os
import json
import numpy as np
from typing import Callable
from ..utils import get_temp_folder


class SaveablePanel:
    NAME = None

    @property
    def save_folder(self):
        return get_temp_folder()

    def get_config_dict(self):
        return {}

    def from_config_dict(self, config_dict):
        pass

    def save_config_dict(self):
        assert self.NAME is not None
        config_dict = self.get_config_dict()
        os.makedirs(self.save_folder, exist_ok=True)
        with open(os.path.join(self.save_folder, f"{self.NAME}.json"), "w") as f:
            json.dump(config_dict, f)

    def load_config_dict(self):
        assert self.NAME is not None
        dictpath = os.path.join(self.save_folder, f"{self.NAME}.json")
        if not os.path.exists(dictpath):
            return
        try:
            with open(dictpath, "r") as f:
                config_dict = json.load(f)
            self.from_config_dict(config_dict)
        except Exception:
            print(f"Invalid config detected at {dictpath} for {self.NAME}. Deleting...")
            os.remove(dictpath)
            return

    def noise_format_func(self) -> Callable[[np.ndarray], np.ndarray]:
        return lambda x: x
