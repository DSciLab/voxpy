from typing import List, Optional, Tuple, Union
import random
import numpy as np
from scipy.ndimage import gaussian_filter
from ..._transform import Transformer
from ..utils import get_range_val


class GaussianBlur(Transformer):
    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self, inp: np.ndarray,
        mask: Optional[np.ndarray]=None,
        sigma: Optional[Union[List[int], float]]=[0, 0.8]
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        sigma = get_range_val(sigma)
        inp = gaussian_filter(inp, sigma=sigma)

        if mask is not None:
            return inp, mask
        else:
            return inp


class RandomGaussianBlur(Transformer):
    def __init__(
        self,
        r_min: Optional[float]=None,
        r_max: Optional[float]=None
    ) -> None:
        super().__init__()
        assert (r_max is None and r_min is None) or (r_max > r_min),\
            f'r_max <= r_min, r_max={r_max} and r_min={r_min}'
        self.r_max = r_max
        self.r_min = r_min
        self.gaussian_blur = [GaussianBlur(), None]

    def __call__(
        self,
        inp: np.ndarray,
        mask: Optional[np.ndarray]=None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

        gaussian_blur = random.choice(self.gaussian_blur)
        if gaussian_blur is None:
            return inp, mask
        else:
            if self.r_min is not None and self.r_max is not None:
                return gaussian_blur(inp, mask, [self.r_min, self.r_max])
            else:
                return gaussian_blur(inp, mask)
