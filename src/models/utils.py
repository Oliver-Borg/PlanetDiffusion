import torch
import os

from ..training.trainer import (
    get_latest_checkpoint,
    MODEL_FILE_NAME
)


def load_model(path) -> torch.nn.Module:
    actual_path = path

    if os.path.isdir(path):
        m_path = os.path.join(path, MODEL_FILE_NAME)
        if os.path.exists(m_path):
            actual_path = m_path
        else:
            latest_checkpoint = get_latest_checkpoint(path)

            if latest_checkpoint is not None:
                actual_path = os.path.join(latest_checkpoint, MODEL_FILE_NAME)
            else:
                raise FileNotFoundError

    return torch.load(actual_path)
