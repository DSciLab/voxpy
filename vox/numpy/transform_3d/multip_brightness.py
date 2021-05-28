from typing import Optional, Tuple, Union
import numpy as np
from vox.numpy._transform import Transformer
from .utils import get_range_val


class MultipBrightness(Transformer):
    @staticmethod
    def brightness_adjust(inp: np.ndarray,
                          multiplier_range: Optional[Tuple[list, tuple]]=(0.5, 2)
                          ) -> np.ndarray:
        value = get_range_val(multiplier_range)
        inp = inp * value
        return inp

    def __call__(self, inp: np.ndarray,
                 mask:Optional[np.ndarray]=None,
                 multiplier_range: Optional[Tuple[list, tuple]]=(0.5, 2)
                 ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

        inp = self.brightness_adjust(inp, multiplier_range)
        if mask is not None:
            return inp, mask
        else:
            return inp


class RandomMultipBrightness(Transformer):
    def __init__(self, min_val: float, max_val: float) -> None:
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.brightness_adjust = MultipBrightness()

    def __call__(self, inp: np.ndarray,
                 mask: Optional[np.ndarray]=None
                 ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        return self.brightness_adjust(
            inp, mask, multiplier_range=(self.min_val, self.max_val))
