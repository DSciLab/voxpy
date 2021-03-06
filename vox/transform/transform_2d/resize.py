import numpy as np
from numpy.core.fromnumeric import size
from scipy.ndimage import affine_transform
from .._transform import Transformer


class Resize(Transformer):
    def __init__(self) -> None:
        super().__init__()

    def transform_matric(self, scale):
        assert len(scale) == 2, f'len(sclae) = {len(scale)} != 2'
        resize_axis_matrix = np.array(
            [[1 / scale[0],     0.,            0.],
             [0.,          1 / scale[1],       0.],
             [0.,               0.,            1.]])

        return resize_axis_matrix

    def __call__(self, inp, mask, scale=None, size=None):
        assert scale is not None or size is not None, \
            'Scale is None and size is None.'
        assert scale is None or size is None, \
            'Ambiguous, scale is not None and size is not None.'

        width = mask.shape[0]
        height = mask.shape[1]

        if scale is not None and not isinstance(scale, (tuple, list)):
            scale = (scale, scale)
        if size is not None and not isinstance(size, (tuple, list)):
            size = (size, size)
        if scale is None:
            scale = (size[0] / width,
                     size[1] / height)
        if size is None:
            size = (int(width * scale[0]),
                    int(height * scale[1]))

        affine_matrix = self.transform_matric(scale)
        if inp.ndim == 2:
            inp = affine_transform(inp, affine_matrix, output_shape=size)
        else:
            inp_ = []
            for i in range(inp.shape[0]):
                inp_.append(affine_transform(inp[i], affine_matrix, output_shape=size))
            inp = np.stack(inp_, axis=0)
        mask = affine_transform(mask, affine_matrix, order=0, output_shape=size)
        return inp, mask.round()


class RandomResize(Transformer):
    def __init__(self, r_min, r_max) -> None:
        super().__init__()
        assert r_max > r_min, \
            f'r_max <= r_min, r_max={r_max} and r_min={r_min}'
        self.r_max = r_max
        self.r_min = r_min
        self.resizer = Resize()

    def __call__(self, inp, mask):
        scale = np.random.rand() * (self.r_max - self.r_min) + self.r_min
        return self.resizer(inp, mask, scale=scale)


class ResizeTo(Transformer):
    def __init__(self, size) -> None:
        super().__init__()
        assert isinstance(size, (tuple, list)) and len(size) == 2
        self.size = size
        self.resizer = Resize()

    def __call__(self, inp, mask):
        return self.resizer(inp, mask, size=self.size)
