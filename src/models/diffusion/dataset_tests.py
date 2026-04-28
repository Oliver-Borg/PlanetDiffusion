from threading import Thread
import time

from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
from PIL import Image

from planetAI.src.data.utils import PlanetConfig
from .inpaint_inference import (
    JigsawGridInferenceDataset,
    create_batch_list,
    InferenceDataset,
)


class MockScheduler:
    def __init__(self):
        pass

    def add_noise(self, x: torch.tensor, y: torch.tensor, t: torch.tensor):
        return torch.zeros_like(x)


if __name__ == "__main__":
    shape = (1024, 1024)
    generated_mask = np.zeros(shape, dtype=bool)
    current_output = torch.zeros((1, 4, *shape), dtype=torch.float32)
    stacked_sketch = np.ones((*shape, 4), dtype=np.uint8)
    water_mask = np.zeros(shape, dtype=bool)
    water_mask[512:768, 512:768] = True

    dataset = JigsawGridInferenceDataset(
        generated_mask,
        current_output,
        stacked_sketch,
        PlanetConfig(data_dir="./planetAI/data/unused"),
        grid_align=192,
        water_mask=water_mask,
        scheduler=MockScheduler(),
    )

    batch_list = []
    signal_container = [False, False]  # STOP, DONE
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)
    batch_thread = Thread(
        target=create_batch_list,
        args=(batch_list, dataloader, signal_container, False, 0),
    )
    batch_thread.start()
    while not signal_container[1]:
        time.sleep(0.1)
    for batch in batch_list:
        cond_image, metadata, ys, xs, new_mask, new_pixels, this_batch_size = batch
        for i in range(this_batch_size):
            batch_ys, batch_xs = ys[i], xs[i]
            generated_mask[batch_ys, batch_xs] = True
    batch_list.clear()

    assert np.all(generated_mask)
    batch_thread.join()
