import numpy as np
from numpy.core.fromnumeric import amax
from .base import Transformer


class Normalize(Transformer):
    pass


class LinearNormalize(Normalize):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, inp, *args):
        min_ = np.min(inp)
        max_ = np.max(inp)

        return (inp - min_) / (max_ - min_)


class CentralNormalize(Normalize):
    def __init__(self, mean, std) -> None:
        super().__init__()
        if std == 0:
            raise ValueError('std should not be 0')
        self.mean = mean
        self.std = std

    def __call__(self, inp, *args):
        return (inp - self.mean) / self.std
