import numpy as np
from .._transform import Transformer


class ZeroPad(Transformer):
    def __init__(self, least_shape):
        super().__init__()
        assert len(least_shape) == 3
        # shape (C, X, Y)
        self.least_shape = least_shape

    def __call__(self, inp, mask=None):
        x_shape = inp.shape
        assert inp.ndim == len(self.least_shape), 'Dim not consistent.'
        assert x_shape[0] == self.least_shape[0], 'Channels not consistent.'
        c = x_shape[0]
        x = x_shape[1]
        y = x_shape[2]

        least_x = self.least_shape[1]
        least_y = self.least_shape[2]

        target_x = max([x, least_x])
        target_y = max([y, least_y])

        diff_x = target_x - x
        diff_y = target_y - y

        pad_x = diff_x // 2
        pad_y = diff_y // 2

        output = np.zeros((c, target_x, target_y))
        output[:, pad_x:pad_x + x, \
                  pad_y:pad_y + y] = inp
        
        if mask is None:
            return output
        else:
            output_mask = np.zeros((target_x, target_y))
            output_mask[pad_x:pad_x + x, \
                        pad_y:pad_y + y] = mask

            return output, output_mask
