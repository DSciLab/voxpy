from typing import Optional, Tuple, Union
import random
import numpy as np
from scipy.signal import convolve
from ..._transform import Transformer


class Sharpening(Transformer):
    filter_2d = np.array([[0, -1, 0],
                          [-1, 4, -1],
                          [0, -1, 0]])

    filter_3d = np.array([[[0, 0, 0],
                           [0, -1, 0],
                           [0, 0, 0]],
                          [[0, -1, 0],
                           [-1, 6, -1],
                           [0, -1, 0]],
                          [[0, 0, 0],
                           [0, -1, 0],
                           [0, 0, 0]],
                          ])

    def __init__(
        self,
        strength: Union[float, Tuple[float, float]] = 0.2,
    ):
        self.strength = strength

    def augument_sharp(
        self,
        inp: np.ndarray
    ) -> np.ndarray:

        for b in range(inp.shape[0]):
            for c in range(inp.shape[3]):
                mn, mx = inp[b, :, :, c].min(), inp[b, :, :, c].max()
                if isinstance(self.strength, float):
                    strength_here = self.strength
                else:
                    strength_here = np.random.uniform(*self.strength)
                if len(inp.shape) == 4:
                    filter_here = self.filter_2d * strength_here
                    filter_here[1, 1] += 1
                else:
                    filter_here = self.filter_3d * strength_here
                    filter_here[1, 1, 1] += 1
                inp[b, :, :, c] = convolve(inp[b, :, :, c],
                                           filter_here,
                                           mode='same')
                inp[b, :, :, c] = np.clip(inp[b, :, :, c], mn, mx)
        return inp

    def __call__(
        self,
        inp: np.ndarray,
        mask: np.ndarray,
        scale: Optional[float]=None
    ) -> Union[np.ndarray, np.ndarray]:

        inp = self.augument_sharp(inp)
        return inp, mask


class RandomSharpening(Transformer):
    def __init__(self) -> None:
        super().__init__()
        self.sharpening_fn = [Sharpening(), None]

    def __call__(
        self,
        inp: np.ndarray,
        mask: np.ndarray
    ) -> Union[np.ndarray, np.ndarray]:
        sharpening_fn = random.choice(self.sharpening_fn)
        if sharpening_fn is None:
            return inp, mask
        else:
            return sharpening_fn(inp, mask)
