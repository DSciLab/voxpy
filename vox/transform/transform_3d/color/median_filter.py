from typing import Optional, Tuple, Union
import random
import numpy as np
from scipy.ndimage import median_filter
from ..._transform import Transformer


class MedianFilter(Transformer):
    def __init__(
        self,
        filter_size: Union[int, Tuple[int, int]]=(3, 5),
    ) -> None:
        self.filter_size = filter_size

    def augument_median_filter(
        self,
        inp: np.ndarray
    ) -> np.ndarray:

        for b in range(inp.shape[0]):
            for c in range(inp.shape[3]):
                if isinstance(self.filter_size, int):
                    filter_size = self.filter_size
                else:
                    filter_size = np.random.randint(*self.filter_size)
                inp[b, :, :, c] = median_filter(inp[b, :, :, c], filter_size)
        return inp

    def __call__(
        self,
        inp: np.ndarray,
        mask: np.ndarray,
        scale: Optional[float]=None
    ) -> Union[np.ndarray, np.ndarray]:

        inp = self.augument_median_filter(inp)
        return inp, mask


class RandomMedianFilter(Transformer):
    def __init__(self) -> None:
        super().__init__()
        self.median_filter_fn = [MedianFilter(), None]

    def __call__(
        self,
        inp: np.ndarray,
        mask: np.ndarray
    ) -> Union[np.ndarray, np.ndarray]:
        
        median_filter_fn = random.choice(self.median_filter_fn)
        if median_filter_fn is None:
            return inp, mask
        else:
            return median_filter_fn(inp, mask)
