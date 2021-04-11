import numpy as np
from vox.numpy._transform import Transformer


class Noise(Transformer):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, inp, mask, std_scale):
        std = std_scale * np.std(inp)
        noise = np.random.normal(0, std, size=inp.shape)
        return np.clip(inp + noise, a_min=0.0, a_max=None), mask


class RandomNoise(Transformer):
    def __init__(self, max_scale, decay=None) -> None:
        super().__init__()
        assert max_scale >= 0.0
        self.max_scale = max_scale
        self.decay = decay
        self.noise = Noise()

    def update_param(self, verbose=False, *args, **kwargs):
        if self.decay is not None:
            self.max_scale *= self.decay
            if verbose:
                print(f'Update {self.__class__.__name__} parameter to '
                      f'{self.max_scale}')

    def __call__(self, inp, mask):
        if self.max_scale == 0.0:
            return inp, mask
        else:
            std_scale = np.random.rand() * self.max_scale
            return self.noise(inp, mask, std_scale)
