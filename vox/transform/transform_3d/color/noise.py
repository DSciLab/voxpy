from typing import Optional, Tuple, Union
import numpy as np
import random
from ..._transform import Transformer


class NoiseBase(Transformer):
    """
        Use me after normalization.
    """
    def __init__(self) -> None:
        super().__init__()

    def augment_noise(
        self,
        inp: np.ndarray,
        noise_variance: Tuple[float, float]=(0, 0.1)
    ) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, inp: np.ndarray,
                 mask: Optional[np.ndarray]=None,
                 scale: Optional[float]=None,
                 ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

        inp = self.augment_noise(inp)
        if mask is not None:
            return inp, mask
        else:
            return inp


class RicianNoise(NoiseBase):
    def augment_noise(
        self,
        inp: np.ndarray,
        noise_variance: Tuple[float, float]=(0, 0.1)
    ) -> np.ndarray:
        variance = random.uniform(noise_variance[0], noise_variance[1])
        inp = np.sqrt(
            (inp + np.random.normal(0.0, variance, size=inp.shape)) ** 2 +
            np.random.normal(0.0, variance, size=inp.shape) ** 2) * np.sign(inp)
        return inp


class GaussianNoise(NoiseBase):
    noise_variance = (0.0, 0.1)

    def augment_noise(
        self,
        inp: np.ndarray,
        noise_variance: Tuple[float, float]=(0, 0.1)
    ) -> np.ndarray:
        if noise_variance[0] == noise_variance[1]:
            variance = noise_variance[0]
        else:
            variance = random.uniform(noise_variance[0], noise_variance[1])
        inp = inp + np.random.normal(0.0, variance, size=inp.shape)
        return inp


class RandomNoise(Transformer):
    """
        Use me after normalization.
    """
    def __init__(self, max_scale: float,
                 decay: Optional[float]=None) -> None:
        super().__init__()
        assert max_scale >= 0.0
        self.max_scale = max_scale
        self.decay = decay
        self.noise = [RicianNoise(), GaussianNoise(), None]

    def __call__(
        self,
        inp: np.ndarray,
        mask: Optional[np.ndarray]=None,
        scale: Optional[float]=None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    
        noise_fn = random.choice(self.noise)
        if noise_fn is None:
            return inp, mask
        else:
            return noise_fn(inp, mask, scale)
