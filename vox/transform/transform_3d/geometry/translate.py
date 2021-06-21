from typing import List, Optional, Tuple, Union
import numpy as np
from scipy.ndimage import affine_transform
from ..._transform import Transformer


class Translate(Transformer):
    OFFSET = 1.0e6
    def __init__(self) -> None:
        super().__init__()

    def transform_matric(self, size: Union[Tuple[int, int, int], List[int]]):
        assert len(size) == 3, f'len(sclae) = {len(size)} != 3'
        translation_axis_matrix = np.array(
            [[1., 0., 0., size[0]],
             [0., 1., 0., size[1]],
             [0., 0., 1., size[2]],
             [0., 0., 0.,      1.]])

        return translation_axis_matrix

    def translate(self, inp: np.ndarray,
                  mask: np.ndarray,
                  scale: Optional[Union[Tuple[float, float, float], List[float]]]=None,
                  size: Optional[Union[Tuple[int, int, int], List[int]]]=None
                  ) -> Tuple[np.ndarray, np.ndarray]:
        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 3, \
                f'The type of scale is not tuple or list, or the length ' + \
                f'is not equal to 3, type(scale)={type(scale)} ' + \
                f'and len(scale)={len(scale)}.'
        if size is not None:
            assert isinstance(size, (tuple, list)) and len(size) == 3, \
                f'The type of size is not tuple or list, or the length ' + \
                f'is not equal to 3, type(size)={type(size)} ' + \
                f'and len(size)={len(size)}.'

        width = inp.shape[0]
        height = inp.shape[1]
        depth = inp.shape[2]

        if size is None:
            size = (width * scale[0],
                    height * scale[1],
                    depth * scale[2])

        affine_matrix = self.transform_matric(size)
        if inp.ndim == 3:
            inp = affine_transform(inp, affine_matrix, order=1)
        else:
            inp_ = []
            for i in range(inp.shape[0]):
                inp_.append(affine_transform(inp[i], affine_matrix, order=1))
            inp = np.stack(inp_, axis=0)
        mask = affine_transform(mask, affine_matrix, order=0)
        return inp, mask.round()


class TranslateX(Translate):
    def __call__(self, inp: np.ndarray,
                 mask: np.ndarray,
                 scale: Optional[Union[Tuple[float, float, float], List[float]]]=None,
                 size: Optional[Union[Tuple[int, int, int], List[int]]]=None
                 ) -> Tuple[np.ndarray, np.ndarray]:
        assert scale is not None or size is not None, \
            'Scale is None and size is None.'
        assert scale is None or size is None, \
            'Ambiguous, scale is not None and size is not None.'
        if scale is not None:
            assert isinstance(scale, (int, float)), \
                f'The type of scale should be int or float.'
            scale = (scale, 0, 0)
        if size is not None:
            assert isinstance(size, (int, float)), \
                f'The type of size should be int or float.'
            size = (size, 0, 0)

        return self.translate(inp, mask, scale, size)


class TranslateY(Translate):
    def __call__(self, inp: np.ndarray,
                 mask: np.ndarray,
                 scale: Optional[Union[Tuple[float, float, float], List[float]]]=None,
                 size: Optional[Union[Tuple[int, int, int], List[int]]]=None
                 ) -> Tuple[np.ndarray, np.ndarray]:
        assert scale is not None or size is not None, \
            'Scale is None and size is None.'
        assert scale is None or size is None, \
            'Ambiguous, scale is not None and size is not None.'
        if scale is not None:
            assert isinstance(scale, (int, float)), \
                f'The type of scale should be int or float.'
            scale = (0, scale, 0)
        if size is not None:
            assert isinstance(size, (int, float)), \
                f'The type of size should be int or float.'
            size = (0, size, 0)

        return self.translate(inp, mask, scale)


class TranslateXYZ(Translate):
    def __call__(self, inp: np.ndarray,
                 mask: np.ndarray,
                 scale: Optional[Union[Tuple[float, float, float], List[float]]]=None,
                 size: Optional[Union[Tuple[int, int, int], List[int]]]=None
                 ) -> Tuple[np.ndarray, np.ndarray]:
        assert scale is not None or size is not None, \
            'Scale is None and size is None.'
        assert scale is None or size is None, \
            'Ambiguous, scale is not None and size is not None.'

        if scale is not None and not isinstance(scale, (tuple, list)):
            scale = (scale, scale, scale)
        if size is not None and not isinstance(size, (tuple, list)):
            size = (size, size, size)

        return self.translate(inp, mask, scale)


class RandomTranslate(Transformer):
    def __init__(self, r_min: float, r_max: float,
                 depth_translate: Optional[bool]=False,
                 decay: Optional[float]=None) -> None:
        super().__init__()
        assert r_max > r_min, \
            f'r_max <= r_min, r_max={r_max} and r_min={r_min}'
        self.r_max = r_max
        self.r_min = r_min
        self.decay = decay
        self.depth_translate = depth_translate
        self.translator = Translate()

    def __call__(self, inp: np.ndarray,
                 mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        scale = np.random.rand() * (self.r_max - self.r_min) + self.r_min
        if not self.depth_translate:
            scale = (scale, scale, 0)

        return self.translator(inp, mask, scale=scale)
