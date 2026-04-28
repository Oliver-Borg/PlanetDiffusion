from dataclasses import dataclass, field
from typing import List, Optional

from ...training.trainer import BaseTrainer
from ...core.terrain_transforms import (
    NormaliseTransform,
    UnnormaliseTransform,
)

@dataclass
class SamplingArguments:
    sample_steps: int = field(
        default=5000,
        metadata={
            'help': 'Run sampling every X steps'
        }
    )
    num_samples: int = field(
        default=32,
        metadata={
            'help': 'Number of samples to generate (must be divisible by batch_size)'
        }
    )
    samples_dir: str = field(
        default='samples',
        metadata={
            'help': 'Where to save sample images'
        }
    )
    normalise_output: bool = field(
        default=False,
        metadata={
            'help': 'Whether to normalise output images after sampling'
        }
    )


class SamplingTrainer(BaseTrainer):
    def __init__(self,
                 sampling_args: Optional[SamplingArguments] = None,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)
        if sampling_args is None:
            sampling_args = SamplingArguments()

        assert sampling_args.num_samples % (
            self.training_args.sampling_batch_size or self.training_args.batch_size
        ) == 0, 'Number of samples must be divisible by batch_size'
        self.sampling_args = sampling_args

        self.unnormalise = UnnormaliseTransform()
        self.normalise = NormaliseTransform()

    def sample(self):
        raise NotImplementedError
