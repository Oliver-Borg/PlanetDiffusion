import itertools
import math
import os
import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

from planetAI.src.data.modal_sketch import ModalSketch
from planetAI.src.data.utils import PlanetConfig, image_grid, tensor_to_np
from planetAI.src.data.dataset import NormaliseTransform, _simple_encode, encoder_size
from planetAI.src.data.landcover_utils import used_classes, landcover_mapping


class PaletteDataset(Dataset):
    def __init__(
        self,
        planet_cfg: PlanetConfig,
        tile_size: int = 256,
        do_transforms: bool = True,
        samples_per_combination: int = 1,
    ):
        assert (
            planet_cfg.image_mode == "planet" and planet_cfg.river_upa_mode == "channel"
        ), "Unsupported configuration detected"

        self.planet_cfg = planet_cfg
        self.tile_size = tile_size
        self.do_transforms = do_transforms
        self.samples_per_combination = samples_per_combination

        self.land_colours = [landcover_mapping[class_name].gray_colour for class_name in used_classes]
        self.temp_colours = planet_cfg.temp_colour_list()[1:]
        self.dem_colours = planet_cfg.colour_list()[1:]
        self.combined_classes = list(itertools.product(self.land_colours, self.temp_colours, self.dem_colours))
        self.modal_sketch = ModalSketch(self.planet_cfg)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                NormaliseTransform(),
            ]
        )

    def __len__(self):
        return len(self.combined_classes) * self.samples_per_combination

    def __getitem__(self, idx: int):
        num_classes = len(self.combined_classes)
        land_colour, temp_colour, dem_colour = self.combined_classes[idx // self.samples_per_combination % num_classes]
        return self.get_item(land_colour, temp_colour, dem_colour)

    def get_item(self, land_colour: int, temp_colour: int, dem_colour: int):
        # TODO Add river support to this
        sketch = np.zeros((self.tile_size, self.tile_size, 4), dtype=np.uint8)
        sketch[:, :, 0] = land_colour
        sketch[:, :, 1] = temp_colour
        sketch[:, :, 2] = dem_colour

        modal_colour = self.modal_sketch.get_colour(land_colour, temp_colour)
        modal_img = np.zeros((self.tile_size, self.tile_size, 3), dtype=np.uint8)
        modal_img[:, :] = modal_colour

        if self.planet_cfg.embedding_type == "disabled":
            embedding = np.zeros(encoder_size(self.planet_cfg), dtype=np.float32)
        else:
            try:
                embedding = _simple_encode(
                    sketch[:, :, 0],
                    sketch[:, :, 1],
                    sketch[:, :, 2],
                    self.planet_cfg,
                    self.planet_cfg.delta,
                    is_mars=land_colour == 255,
                )
            except Exception:
                print("Unable to get embedding, using zeros")
                embedding = np.zeros(encoder_size(self.planet_cfg), dtype=np.float32)

        to_return: dict[str, dict | torch.Tensor | np.ndarray] = {
            "metadata": {
                "embedding": embedding,
                "sample_name": f"{land_colour}_{temp_colour}_{dem_colour}"
            }
        }

        target = np.dstack([modal_img, sketch[:, :, 2]])
        condition = sketch

        if self.do_transforms:
            to_return["target_image"] = self.transform(target.astype(np.float32) / 255.0)
            to_return["cond_image"] = self.transform(condition.astype(np.float32) / 255.0)
        else:
            to_return["target_image"] = target
            to_return["cond_image"] = condition

        return to_return


if __name__ == "__main__":
    test_cfg = PlanetConfig(image_mode="planet", river_upa_mode="channel")
    palette_dataset = PaletteDataset(test_cfg)

    print("Length:", len(palette_dataset))

    dataloader = DataLoader(palette_dataset, shuffle=False)

    im_grid = []
    cols = 10

    for i, batch in enumerate(dataloader):
        display_tensor = test_cfg.output_display(
            batch["cond_image"], batch["target_image"], batch["target_image"], show_outputs=False
        )
        c, h, w = display_tensor.shape
        im_grid.append(Image.fromarray(tensor_to_np(display_tensor)).resize((512 * w // h, 512), Image.NEAREST))

    im_grid = image_grid(im_grid, int(math.ceil(len(im_grid) / cols)), cols)

    im_grid.save(os.path.join(test_cfg.test_dir, "palette_dataset.png"))
