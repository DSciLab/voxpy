import numpy as np
from scipy.ndimage import gaussian_filter
from vox.numpy._transform import Transformer


class GaussianBlur(Transformer):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, inp, mask, sigma=None):
        return gaussian_filter(inp, sigma=sigma), mask


class RandomGaussianBlur(Transformer):
    def __init__(self, r_min, r_max, decay=None) -> None:
        super().__init__()
        assert r_max > r_min, \
            f'r_max <= r_min, r_max={r_max} and r_min={r_min}'
        self.r_max = r_max
        self.r_min = r_min
        self.decay = decay
        self.gaussian_blur = GaussianBlur()

    def update_param(self, verbose=False, *args, **kwargs):
        if self.decay is not None:
            self.r_max *= self.decay
            self.r_min *= self.decay
            if verbose:
                print(f'Update {self.__class__.__name__} parameter to '
                      f'{self.r_min}~{self.r_max}')

    def __call__(self, inp, mask):
        sigma = np.random.rand() * (self.r_max - self.r_min) + self.r_min
        return self.gaussian_blur(inp, mask, sigma)
