import numpy as np
from .base import Transformer


class Normalize(Transformer):
    pass


class LinearNormalize(Normalize):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, inp, gt):
        min_ = np.min(inp)
        max_ = np.max(inp)

        return (inp - min_) / (max_ - min_), gt


class CentralNormalize(Normalize):
    def __init__(self, mean, std) -> None:
        super().__init__()
        if std == 0:
            raise ValueError('std should not be 0')
        self.mean = mean
        self.std = std

    def __call__(self, inp, gt):
        return (inp - self.mean) / self.std, gt


class GeneralNormalize(Normalize):
    def __init__(self, opt) -> None:
        super().__init__()
        norm_opt = opt.get('norm', 'LinearNormalize')
        if norm_opt == 'LinearNormalize':
            self.norm = LinearNormalize()
        elif norm_opt == 'CentralNormalize':
            self.norm = CentralNormalize(opt.mean, opt.std)
        else:
            raise RuntimeError(f'Unrecognized norm function ({opt.norm}).')

    def __call__(self, *args, **kwargs):
        return self.norm(*args, **kwargs)
