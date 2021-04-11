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
    def contrast(self, inp, alpha, base=None):
        alpah = float(alpha)
        base = base or inp.mean() / 2
        inp_ = base + alpah * inp - alpah * base
        return np.clip(inp_, 0, inp.max())


class LowerContrast(Contrast):
    def contrast(self, inp, alpha, *args, **kwargs):
        alpha = float(alpha)
        inp_max = inp.max()
        # base = np.argmax(np.bincount(inp.reshape(-1).astype(np.int)))
        inp = (inp / inp.mean())**alpha
        inp = (inp / inp.max()) * inp_max
        return np.clip(inp, 0, inp_max)


class RandomContrast(Transformer):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, inp, mask):
        pass
