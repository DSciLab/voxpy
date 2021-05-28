from typing import Optional, Tuple, Union
import numpy as np
from scipy.ndimage import gaussian_filter
from vox.numpy._transform import Transformer
from .utils import get_range_val


class GaussianBlur(Transformer):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, inp: np.ndarray,
                 mask: Optional[np.ndarray]=None,
                 sigma: Optional[Union[list, tuple, float]]=0
                 ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        sigma = get_range_val(sigma)
        inp = gaussian_filter(inp, sigma=sigma)

        if mask is not None:
            return inp, mask
        else:
            return inp


class RandomGaussianBlur(Transformer):
    def __init__(self, r_min: float,
                 r_max: float,
                 decay: Optional[float]=None) -> None:
        super().__init__()
        assert r_max > r_min, \
            f'r_max <= r_min, r_max={r_max} and r_min={r_min}'
        self.r_max = r_max
        self.r_min = r_min
        self.decay = decay
        self.gaussian_blur = GaussianBlur()

    def __call__(self, inp: np.ndarray,
                 mask: Optional[np.ndarray]=None
                 ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

        sigma = np.random.rand() * (self.r_max - self.r_min) + self.r_min
        return self.gaussian_blur(inp, mask, sigma)
