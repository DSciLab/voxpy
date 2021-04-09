import numpy as np
from .base import Transformer


class Normalize(Transformer):
    pass


class LinearNormalize(Normalize):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def __call__(self, inp, gt):
        min_ = np.min(inp)
        max_ = np.max(inp)

        return (inp - min_) / (max_ - min_), gt


class CentralNormalize(Normalize):
    ESP = 1.0e-8
    def __init__(self, mean=None, std=None) -> None:
        super().__init__()
        assert (mean is None and std is None) or \
               (mean is not None and std is not None), f'mean={mean} and std={std}'
        self.mean = mean
        self.std = std

    def __call__(self, inp, gt):
        if self.mean is None and self.std is None:
            mean = np.mean(inp)
            std = np.std(inp)
        else:
            mean = self.mean
            std = self.std

        return (inp - mean) / (std + self.ESP), gt


class GeneralNormalize(Normalize):
    def __init__(self, opt) -> None:
        super().__init__()
        norm_opt = opt.get('norm', 'LinearNormalize')
        if norm_opt == 'LinearNormalize':
            self.norm = LinearNormalize()
        elif norm_opt == 'CentralNormalize':
            if opt.get('unified_stat', False):
                self.norm = CentralNormalize()
            else:
                self.norm = CentralNormalize(opt.mean, opt.std)
        else:
            raise RuntimeError(f'Unrecognized norm function ({opt.norm}).')

    def __call__(self, *args, **kwargs):
        return self.norm(*args, **kwargs)
