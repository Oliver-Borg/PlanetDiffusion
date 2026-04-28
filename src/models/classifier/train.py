
from dataclasses import dataclass, field

import os

import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau


from .efficientnet import efficientnet_v2_s, efficientnet_v2_m
from .dataset import CustomImageFolder
from .classes import TYPE_MAPPINGS
from .filter import ClassifierArguments

from ...training.metrics import ClassificationMetrics
from ...training.trainer import BaseTrainer, TrainingArguments
from ...core.dataclass_argparser import CustomArgumentParser


@dataclass
class ClassifierTrainingArguments(TrainingArguments, ClassifierArguments):
    results_dir: str = field(
        default='{classifiers_dir}/{classifier_type}',
        metadata=TrainingArguments.__dataclass_fields__['results_dir'].metadata
    )

    def __post_init__(self):
        self.results_dir = self.results_dir.format(
            classifiers_dir=self.classifiers_dir,
            classifier_type=self.classifier_type
        )


class ClassifierTrainer(BaseTrainer):

    def extract(self, item):
        # TODO move logic to Base
        inputs = item[0].to(self.device)
        targets = item[1].to(self.device)
        return inputs, targets


def main():

    parser = CustomArgumentParser(
        ClassifierTrainingArguments,
        description='Train a classifier'
    )

    classifier_args,  = parser.parse_args_into_dataclasses()

    c_type, num_channels, img_ext = TYPE_MAPPINGS[classifier_args.classifier_type]

    c_dir = os.path.join(classifier_args.data_dir,
                         classifier_args.classifier_type)
    training_dir = os.path.join(c_dir, classifier_args.training_dir)

    # Create Model
    model = efficientnet_v2_s(
        num_classes=len(c_type),
        img_channels=num_channels
    ).cuda()

    datasets = {}
    for f in ('train', 'valid', 'test'):
        datasets[f] = CustomImageFolder(
            os.path.join(training_dir, f),
            classes_enum=c_type,
            ext=img_ext,
            data_augmentation=(f == 'train')
        )

    # Define optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=classifier_args.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-5,  # https://stackoverflow.com/a/46597531/13989043
    )

    # Define loss function
    criterion = nn.CrossEntropyLoss(
        # TODO does this help?
        weight=datasets['train'].class_weights.cuda()
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=0.8,
        patience=5,
        verbose=True,
        min_lr=1e-5,
    )

    metrics = [
        ClassificationMetrics,
    ]
    trainer = ClassifierTrainer(
        model=model,
        datasets=datasets,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,

        metrics=metrics,

        training_args=classifier_args
    )

    trainer.train()


if __name__ == '__main__':
    main()
