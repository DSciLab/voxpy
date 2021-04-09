import numpy as np
from scipy.ndimage import affine_transform
from vox.numpy._transform import Transformer


class Translate(Transformer):
    OFFSET = 1.0e6
    def __init__(self) -> None:
        super().__init__()

    def transform_matric(self, size):
        assert len(size) == 3, f'len(sclae) = {len(size)} != 3'
        translation_axis_matrix = np.array(
            [[1.,     0.,     0.,     size[0]],
             [0.,     1.,     0.,     size[1]],
             [0.,     0.,     1.,     size[2]],
             [0.,     0.,     0.,           1.]])

        return translation_axis_matrix

    def __call__(self, inp, mask, scale=None, size=None):
        assert scale is not None or size is not None, \
            'Scale is None and size is None.'
        assert scale is None or size is None, \
            'Ambiguous, scale is not None and size is not None.'

        width = inp.shape[0]
        height = inp.shape[1]
        depth = inp.shape[2]

        if scale is not None and not isinstance(scale, (tuple, list)):
            scale = (scale, scale, scale)
        if size is not None and not isinstance(size, (tuple, list)):
            size = (size, size, size)
        if size is None:
            size = (width * scale[0],
                    height * scale[1],
                    depth * scale[2])

        affine_matrix = self.transform_matric(size)
        if inp.ndim == 3:
            inp = affine_transform(inp, affine_matrix)
        else:
            inp_ = []
            for i in range(inp.shape[0]):
                inp_.append(affine_transform(inp[i], affine_matrix))
            inp = np.stack(inp_, axis=0)
        mask = affine_transform(mask, affine_matrix, order=0)
        return inp, mask.round()


class RandomTranslate(Transformer):
    def __init__(self, r_min, r_max, depth_translate=False, decay=None) -> None:
        super().__init__()
        assert r_max > r_min, \
            f'r_max <= r_min, r_max={r_max} and r_min={r_min}'
        self.r_max = r_max
        self.r_min = r_min
        self.decay = decay
        self.depth_translate = depth_translate
        self.translator = Translate()

    def update_param(self, verbose=False, *args, **kwargs):
        if self.decay is not None:
            self.r_max *= self.decay
            self.r_min *= self.decay
            if verbose:
                print(f'Update {self.__class__.__name__} parameter to '
                      f'{self.r_min}~{self.r_max}')

    def __call__(self, inp, mask):
        scale = np.random.rand() * (self.r_max - self.r_min) + self.r_min
        if not self.depth_translate:
            scale = (scale, scale, 0)

        return self.translator(inp, mask, scale=scale)
