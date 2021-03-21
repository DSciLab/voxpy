import numpy as np
from .base import Transformer


class RandomSampling(Transformer):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __call__(self, inp, mask=None):
        inp_shape = inp.shape
        assert inp.ndim == len(self.shape)

        assert inp_shape[1] >= self.shape[1], f'x over sampling, {inp_shape[1]} > {self.shape[1]}'
        assert inp_shape[2] >= self.shape[2], f'y over sampling, {inp_shape[2]} > {self.shape[2]}'
        assert inp_shape[3] >= self.shape[3], f'z over sampling, {inp_shape[3]} > {self.shape[3]}'

        x_available = (inp_shape[1] - self.shape[1]) // 2 + 1
        y_available = (inp_shape[2] - self.shape[2]) // 2 + 1
        z_available = (inp_shape[3] - self.shape[3]) // 2 + 1

        x_start = np.random.randint(0, x_available)
        y_start = np.random.randint(0, y_available)
        z_start = np.random.randint(0, z_available)

        if mask is not None:
            return inp[:, x_start: x_start + self.shape[1], \
                          y_start: y_start + self.shape[2], \
                          z_start: z_start + self.shape[3]], \
                mask[x_start: x_start + self.shape[1], \
                     y_start: y_start + self.shape[2], \
                     z_start: z_start + self.shape[3]]
        else:
            return inp[:, x_start: x_start + self.shape[1], \
                          y_start: y_start + self.shape[2], \
                          z_start: z_start + self.shape[3]]
