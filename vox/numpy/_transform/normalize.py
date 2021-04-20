import numpy as np
from .base import Transformer


class Normalize(Transformer):
    ESP = 1.0e-8


class LinearNormalize(Normalize):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def __call__(self, inp, gt, *args, **kwargs):
        min_ = np.min(inp)
        max_ = np.max(inp)

        return (inp - min_) / (max_ - min_), gt


class CentralNormalize(Normalize):
    def __init__(self, mean=None, std=None) -> None:
        super().__init__()
        assert (mean is None and std is None) or \
               (mean is not None and std is not None), f'mean={mean} and std={std}'
        self.mean = mean
        self.std = std

    def __call__(self, inp, gt, *args, **kwargs):
        if self.mean is None and self.std is None:
            mean = np.mean(inp)
            std = np.std(inp)
        else:
            mean = self.mean
            std = self.std

        return (inp - mean) / (std + self.ESP), gt


class RunningCentralNormalize(Normalize):
    def __init__(self, beta=0.8) -> None:
        super().__init__()
        self.beta = beta
        self.runing_mean = None
        self.runing_std = None

    def __call__(self, inp, gt):
        mean = np.mean(inp)
        std = np.std(inp)
        if self.runing_mean is None or self.runing_std is None:
            self.runing_mean = mean
            self.runing_std = std
        else:
            self.runing_mean = self.runing_mean * self.beta + mean * (1 - self.beta)
            self.runing_std = self.runing_std * self.beta + std * (1 - self.beta)

        inp = (inp - self.runing_mean) / (self.runing_std + self.ESP)
        return inp, gt


class DomainSpecificCentralNormalize(Normalize):
    def __init__(self, num_domain, beta=0.8) -> None:
        super().__init__()
        self.running_central_normalize_list = [
            RunningCentralNormalize(beta) for _ in range(num_domain)]

    def __call__(self, inp, gt, vendor_id=None):
        if vendor_id is None:
            normalized_inp_list = []
            for normalize_fn in self.running_central_normalize_list:
                normalized_inp, _ = normalize_fn(inp, gt)
                normalized_inp_list.append(normalized_inp)
            return np.mean(normalized_inp_list, axis=0), gt
        else:
            normalize_fn = self.running_central_normalize_list[vendor_id]
            return normalize_fn(inp, gt)


class GeneralNormalize(Normalize):
    def __init__(self, opt) -> None:
        super().__init__()
        norm_opt = opt.get('norm', 'LinearNormalize')
        if norm_opt == 'LinearNormalize':
            self.norm = LinearNormalize()
        elif norm_opt == 'RunningCentralNormalize':
            beta = opt.get('running_normalize_beta', 0.8)
            self.norm = RunningCentralNormalize(beta)
        elif norm_opt == 'DomainSpecificCentralNormalize':
            beta = opt.get('running_normalize_beta', 0.8)
            num_domain = opt.get('num_domain', None)
            assert num_domain is not None, \
                f'num_domain is undefined on DomainSpecific mod.'
            self.norm = DomainSpecificCentralNormalize(num_domain=num_domain,
                                                       beta=beta)
        elif norm_opt == 'CentralNormalize':
            if opt.get('unified_stat', False):
                self.norm = CentralNormalize()
            else:
                self.norm = CentralNormalize(opt.mean, opt.std)
        else:
            raise RuntimeError(f'Unrecognized norm function ({opt.norm}).')

    def __call__(self, *args, **kwargs):
        return self.norm(*args, **kwargs)
