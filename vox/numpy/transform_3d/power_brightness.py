from typing import Optional, Tuple, Union
import numpy as np
from vox.numpy._transform import Transformer
from .utils import get_range_val


class PowerBrightness(Transformer):
    _RANGE = [0.8, 1.2]

    @staticmethod
    def brightness_adjust(inp: np.ndarray,
                          power: Union[float, list, tuple]
                          ) -> np.ndarray:
        power = get_range_val(power)
        max_inp = np.max(inp)
        min_inp = np.min(inp)
        range_inp = max_inp - min_inp

        inp_1 = (inp - min_inp) / range_inp
        inp_1 = np.power(inp_1, power)
        inp = inp_1 * range_inp
        return inp

    def __call__(self, inp: np.ndarray,
                 mask: Optional[np.ndarray]=None,
                 power: float=0.8
                 ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        inp = self.brightness_adjust(inp, power)
        return inp, mask


class RandomPowerBrightness(Transformer):
    def __init__(self, min_val: float, max_val: float) -> None:
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.brightness_adjust = PowerBrightness()

    def __call__(self, inp: np.ndarray,
                 mask: Optional[np.ndarray]=None
                 ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

        return self.brightness_adjust(
            inp, mask, power=(self.min_val, self.max_val))
