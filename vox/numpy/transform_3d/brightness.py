from typing import Optional, Tuple, Union
import numpy as np
import random
from vox.numpy._transform import Transformer

from .power_brightness import PowerBrightness, RandomPowerBrightness
from .sin_brightness import SinBrightness, ArcSinBrightness, RandomSinBrightness
from .multip_brightness import MultipBrightness, RandomMultipBrightness


class Brightness(Transformer):
    def __init__(self, method: str='random') -> None:
        super().__init__()
        self.brightness_dict = {'PowerBrightness': PowerBrightness(),
                                'SinBrightness': SinBrightness(),
                                'ArcSinBrightness': ArcSinBrightness(),
                                'MultipBrightness': MultipBrightness()}
        self.brightness_method_list = list(self.brightness_dict.values())
        self.brightness_key_list = list(self.brightness_dict.keys()) + ['random']
        self.method = method

    def get_brightness_method(self) -> Transformer:
        if self.method not in self.brightness_key_list:
            raise ValueError(f'Unrecognized brightness method ({self.method})')
        if self.method == 'random':
            random.choice(self.brightness_method_list)
        else:
            return self.brightness_dict[self.method]

    def __call__(self, inp: np.ndarray,
                 mask: Optional[np.ndarray]=None,
                 scale: float=0.7
                 ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        brightness = self.get_brightness_method()
        return brightness(inp, mask, scale)
