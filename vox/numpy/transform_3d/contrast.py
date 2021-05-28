from typing import List, Optional, Tuple, Union
import numpy as np
from vox.numpy._transform import Transformer


class Contrast(Transformer):
    def __init__(self, max_val) -> None:
        super().__init__()
        self.max_val = max_val

    def contrast(self, inp, alpha, base=None):
        raise NotImplementedError

    def __call__(self, inp, mask, alpha, base=None):
        return self.contrast(inp, alpha=alpha, base=base), mask


class HigherContrast(Contrast):
    _RANGE: List[float] = [0.0, 2.3]

    def contrast(self, inp: np.ndarray,
                 alpha: float,
                 base: Optional[float]=None) -> np.ndarray:
        base = base or inp.mean() / 2
        inp_ = base + alpha * inp - alpha * base
        return np.clip(inp_, 0, inp.max())


class LowerContrast(Contrast):
    _RANGE: List[float] = [1.0, 1.3]

    def contrast(self, inp: np.ndarray,
                 alpha: float, *args, **kwargs) -> np.ndarray:
        inp_max = inp.max()
        inp = (inp / inp.mean())**alpha
        inp = (inp / inp.max()) * inp_max
        return np.clip(inp, 0, inp_max)


class RandomContrast(Transformer):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, inp: np.ndarray,
                 mask: Optional[np.ndarray]
                 ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        pass
