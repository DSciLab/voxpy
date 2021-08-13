import numpy as np
import torch


class Rescale(object):
    pass


class LinearNormRescale255(Rescale):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, inp):
        return inp * 255


class CentralNormRescale255(Rescale):
    def __init__(self, mean, std) -> None:
        super().__init__()
        self.mean = mean
        self.std = std

    def __call__(self, inp):
        inp = inp * self.std + self.mean
        if isinstance(inp, np.ndarray):
            min_ = np.min(inp)
            max_ = np.max(inp)
        else:
            min_ = torch.min(inp)
            max_ = torch.max(inp)

        inp = (inp - min_) / (max_ - min_)

        return inp * 255


class GeneralNormRescale255(Rescale):
    def __init__(self, opt) -> None:
        super().__init__()
        norm_opt = opt.get('norm', 'LinearNormalize')
        if norm_opt == 'LinearNormalize':
            self.rescale = LinearNormRescale255()
        elif 'Central' in norm_opt:
            self.rescale = CentralNormRescale255(opt.mean, opt.std)
        else:
            raise RuntimeError(f'Unrecognized norm function ({opt.norm}).')

    def __call__(self, *args, **kwargs):
        return self.rescale(*args, **kwargs)
