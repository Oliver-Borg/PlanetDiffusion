import os
from typing import Any, List, Tuple

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from PIL import Image

from .classes import FilterClass
from ...core.terrain_transforms import RandomTerrainTransform, NormaliseTransform

class CustomImageFolder(Dataset):
    def __init__(
            self,
            folder,
            classes_enum: FilterClass,
            ext,
            data_augmentation=True
    ) -> None:

        self.folder = folder
        self.extensions = ext

        self.classes_enum = classes_enum
        self.num_classes = len(classes_enum)
        self.classes = [x.name for x in classes_enum]
        self.class_to_idx = classes_enum.class_to_idx()
        self.idx_to_class = classes_enum.idx_to_class()

        self.samples, self.class_weights = self.make_dataset()

        tforms = [
            transforms.ToTensor()
        ]
        if data_augmentation:
            tforms.append(RandomTerrainTransform())

        tforms.append(NormaliseTransform())

        self.transform = transforms.Compose(tforms)

    def make_dataset(self) -> List[Tuple[str, int]]:

        class_counts = {}
        instances = []
        for target_class in self.classes_enum:

            target_dir = os.path.join(self.folder, target_class.name)
            if not os.path.isdir(target_dir):
                continue

            curr_instances = []
            fnames = os.listdir(target_dir)
            for f in fnames:
                if f.endswith(self.extensions):
                    path = os.path.join(target_dir, f)
                    item = path, target_class.value
                    curr_instances.append(item)

            class_counts[target_class.name] = len(curr_instances)

            instances.extend(curr_instances)

        total_count = sum(class_counts.values())
        class_counts = [class_counts[x.name] for x in self.classes_enum]

        # https://stackoverflow.com/questions/61414065/pytorch-weight-in-cross-entropy-loss
        # https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
        class_weights = total_count / \
            (len(class_counts) * torch.tensor(class_counts))
        return instances, class_weights

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.transform(Image.open(path))

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)
