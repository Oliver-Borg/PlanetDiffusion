from abc import abstractmethod
import numpy as np
import torch

from ..core.utils import roll_up, roll_left


def mse(predicted: torch.Tensor, actual: torch.Tensor):
    """Global Mean Squared Error"""
    return ((predicted - actual)**2).mean()


def mae(predicted: torch.Tensor, actual: torch.Tensor):
    """Global Mean Absolute Error"""
    return ((predicted - actual).abs()).mean()


def schwarz_accuracy(sgf):
    """
    A measure of whether a synthesised derivative image
    respects Schwarz's theorem:

    Schwarz's theorem
    Let f : R^2 → R. Assume that the mixed partial derivatives
    f_xy and f_yx exist on a disc D centered at the point (a, b)
    and are continuous at that point. Then f_xy(a, b) = f_yx(a, b)
    """

    dh_dx = sgf[:, :, 0]
    dh_dy = sgf[:, :, 1]

    rolled_left = roll_left(dh_dy)
    rolled_up = roll_up(dh_dx)

    a = dh_dx + rolled_left
    b = dh_dy + rolled_up

    return np.nanmean((a - b) ** 2)


class Metric:
    name = None

    @abstractmethod
    def __call__(self, output, target=None):
        raise NotImplementedError


class SchwarzMetric(Metric):
    name = 'schwarz_accuracy'

    def __call__(self, output, target=None):
        running_accuracy = 0
        for img in output:
            running_accuracy += schwarz_accuracy(img.cpu())
        return running_accuracy/len(output)


METRIC_NAMES = {cls.name: cls for cls in Metric.__subclasses__()}
