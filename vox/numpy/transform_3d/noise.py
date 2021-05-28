from typing import Optional, Tuple, Union
import numpy as np
from vox.numpy._transform import Transformer
from .utils import get_range_val


class Noise(Transformer):
    """
        Use me after normalization.
    """
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, inp: np.ndarray,
                 mask: Optional[np.ndarray]=None,
                 std_scale: Optional[Union[float, list, tuple]]=1.0,
                 ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        std = get_range_val(std_scale)
        noise = np.random.normal(0.0, std, size=inp.shape)
        inp = np.clip(inp + noise, a_min=0.0, a_max=None)
        if mask is not None:
            return inp, mask
        else:
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
        self.noise = Noise()

    def __call__(self, inp: np.ndarray,
                 mask: Optional[np.ndarray]=None
                 ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if self.max_scale == 0.0:
            if mask is not None:
                return inp, mask
            else:
                return inp
        else:
            std_scale = np.random.uniform() * self.max_scale
            return self.noise(inp, mask, std_scale)
