from typing import Optional, Tuple, Union
import numpy as np
import random
from ..._transform import Transformer


class BrightnessBase(Transformer):
    def __init__(self, method: str='random') -> None:
        super().__init__()

    def __call__(self,
        inp: np.ndarray,
        mask: Optional[np.ndarray]=None,
        scale: float=None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

        inp = self.augment_brightness(inp)
        return inp, mask


class BrightnessAdditive(BrightnessBase):
    def augment_brightness(
        self,
        inp: np.ndarray,
        mu:float=0.0,
        sigma:float=0.2,
    ) -> np.ndarray:
        for c in range(inp.shape[0]):
            rnd_nb = np.random.normal(mu, sigma)
            inp[c] += rnd_nb
        return inp


class BrightnessMultiplicative(BrightnessBase):
    def augment_brightness(
        self,
        inp: np.ndarray,
        multiplier_range: Optional[Tuple[float, float]]=(0.5, 2),
    ) -> np.ndarray:

        for c in range(inp.shape[0]):
            multiplier = np.random.uniform(multiplier_range[0],
                                           multiplier_range[1])
            inp[c] *= multiplier
        return inp


class RandomBrightness(Transformer):
    def __init__(self) -> None:
        super().__init__()
        self.brightness_fn = [
            BrightnessAdditive(), BrightnessMultiplicative(), None]
    
    def __call__(
        self,
        inp: np.ndarray,
        mask: np.ndarray
    ) -> Union[np.ndarray, np.ndarray]:
        brightness_fn = random.choice(self.brightness_fn)
        if brightness_fn is None:
            return inp, mask
        else:
            return brightness_fn(inp, mask)
