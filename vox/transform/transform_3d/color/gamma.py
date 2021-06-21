from inspect import FrameInfo
from typing import Optional, Tuple, Union
import random
import numpy as np
from ..._transform import Transformer


class Gamma(Transformer):
    def __init__(self) -> None:
        super().__init__()

    def augment_gamma(
        self,
        inp: np.ndarray,
        gamma_range: Optional[Tuple[float, float]]=(0.5, 2.0),
        invert_image: Optional[bool]=False,
        epsilon: Optional[bool]=1e-7,
        retain_stats=False
    ) -> np.ndarray:

        if invert_image:
            inp = - inp

        if retain_stats:
            mn = inp.mean()
            sd = inp.std()

        if np.random.random() < 0.5 and gamma_range[0] < 1:
            gamma = np.random.uniform(gamma_range[0], 1)
        else:
            gamma = np.random.uniform(max(gamma_range[0], 1),
                                      gamma_range[1])

        minm = inp.min()
        rnge = inp.max() - minm
        inp = np.power(((inp - minm) / float(rnge + epsilon)), gamma)\
              * rnge + minm

        if retain_stats:
            inp = inp - inp.mean()
            inp = inp / (inp.std() + 1e-8) * sd
            inp = inp + mn

        if invert_image:
            inp = - inp
        return inp

    def __call__(
        self,
        inp: np.ndarray,
        mask: Optional[np.ndarray]=None,
        scale: Optional[float]=None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

        return self.augment_gamma(inp), mask


class RandomGamma(Transformer):
    def __init__(self) -> None:
        super().__init__()
        self.gamma_fn = [Gamma(), None]

    def __call__(
        self,
        inp: np.ndarray,
        mask: np.ndarray
    ) -> Union[np.ndarray, np.ndarray]:
        gamma_fn = random.choice(self.gamma_fn)
        if gamma_fn is None:
            return inp, mask
        else:
            return gamma_fn(inp, mask)
