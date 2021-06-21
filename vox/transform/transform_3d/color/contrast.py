import random
from typing import Optional, Tuple, Union, Callable
import numpy as np
from ..._transform import Transformer


class Contrast(Transformer):
    def __init__(self) -> None:
        super().__init__()

    def contrast(
        self,
        inp: np.ndarray,
        contrast_range: Union[Tuple[float, float],
                              Callable[[], float]]=(0.75, 1.25),
    ) -> np.ndarray:

        if callable(contrast_range):
            factor = contrast_range()
        else:
            if np.random.random() < 0.5 and contrast_range[0] < 1:
                factor = np.random.uniform(contrast_range[0], 1)
            else:
                factor = np.random.uniform(max(contrast_range[0], 1),
                                           contrast_range[1])

        for c in range(inp.shape[0]):
            mn = inp[c].mean()
            minm = inp[c].min()
            maxm = inp[c].max()

            inp[c] = (inp[c] - mn) * factor + mn

            inp[c][inp[c] < minm] = minm
            inp[c][inp[c] > maxm] = maxm
        return inp

    def __call__(
        self,
        inp: np.ndarray,
        mask: np.ndarray,
        scale: Optional[float]=None
    ) -> Union[np.ndarray, np.ndarray]:
        return self.contrast(inp), mask


class RandomContrast(Transformer):
    def __init__(self) -> None:
        super().__init__()
        self.contrast_fn = [Contrast(), None]

    def __call__(
        self,
        inp: np.ndarray,
        mask: np.ndarray
    ) -> Union[np.ndarray, np.ndarray]:
        
        contrast_fn = random.choice(self.contrast_fn)
        if contrast_fn is None:
            return inp, mask
        else:
            return contrast_fn(inp, mask)
