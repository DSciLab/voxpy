from typing import List, Optional, Tuple, Union
import numpy as np
from numpy.core.fromnumeric import size
from scipy.ndimage import affine_transform
from ..._transform import Transformer


class Resize(Transformer):
    def __init__(self) -> None:
        super().__init__()

    def transform_matric(self, scale: Union[Tuple[int, int, int],
                                            List[int]]) -> np.ndarray:
        assert len(scale) == 3, f'len(sclae) = {len(scale)} != 3'
        resize_axis_matrix = np.array(
            [[1 / scale[0],     0.,            0.,     0.],
             [0.,          1 / scale[1],       0.,     0.],
             [0.,               0.,      1 / scale[2], 0.],
             [0.,               0.,            0.,     1.]])

        return resize_axis_matrix

    def __call__(self, inp: np.ndarray,
                 mask: np.ndarray,
                 scale: Optional[Union[float, Tuple[float, float, float], List[float]]]=None,
                 size: Optional[Union[int, Tuple[int, int, int], List[int]]]=None
                 ) -> Tuple[np.ndarray, np.ndarray]:
        assert scale is not None or size is not None, \
            'Scale is None and size is None.'
        assert scale is None or size is None, \
            'Ambiguous, scale is not None and size is not None.'

        width = mask.shape[0]
        height = mask.shape[1]
        depth = mask.shape[2]

        if scale is not None and not isinstance(scale, (tuple, list)):
            scale = (scale, scale, scale)
        if size is not None and not isinstance(size, (tuple, list)):
            size = (size, size, size)
        if scale is None:
            scale = (size[0] / width,
                     size[1] / height,
                     size[2] / depth)
        if size is None:
            size = (int(width * scale[0]),
                    int(height * scale[1]),
                    int(depth * scale[2]))

        affine_matrix = self.transform_matric(scale)
        if inp.ndim == 3:
            inp = affine_transform(inp, affine_matrix,
                                   order=1,
                                   output_shape=size)
        else:
            inp_ = []
            for i in range(inp.shape[0]):
                # TODO fix me.
                inp_.append(affine_transform(inp[i], affine_matrix,
                                             order=1,
                                             output_shape=size))
            inp = np.stack(inp_, axis=0)
        
        mask = affine_transform(mask, affine_matrix, order=0,
                                output_shape=size)
        return inp, mask.round()


class ResizeTo(Transformer):
    def __init__(self, shape: Union[Tuple[int, int, int, int], List[int]]) -> None:
        super().__init__()
        self.shape = shape
        self.resize = Resize()

    def __call__(self, inp: np.ndarray,
                 mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.resize(inp, mask, size=self.shape[1:])


class RandomResize(Transformer):
    def __init__(self, r_min: float, r_max: float,
                 decay: Optional[float]=None) -> None:
        super().__init__()
        assert r_max > r_min, \
            f'r_max <= r_min, r_max={r_max} and r_min={r_min}'
        self.r_max = r_max
        self.r_min = r_min
        self.decay = decay
        self.resizer = Resize()

    def __call__(self, inp: np.ndarray,
                 mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        scale = np.random.rand() * (self.r_max - self.r_min) + self.r_min
        return self.resizer(inp, mask, scale=scale)


class MaxSize(Transformer):
    def __init__(self, size: Union[Tuple[int, int, int, int], List[int]]) -> None:
        super().__init__()
        assert isinstance(size, (tuple, list)) and len(size) == 3
        self.size = size
        self.resizer = Resize()

    def __call__(self, inp: np.ndarray,
                 mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mask_shape = mask.shape
        if np.prod(mask_shape) > np.prod(self.size):
            return self.resizer(inp, mask, size=self.size)
        else:
            return inp, mask
