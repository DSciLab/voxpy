import torch
from .utils import affine_transform
from vox._transform import Transformer


class Translate(Transformer):
    OFFSET = 1.0e6
    def __init__(self) -> None:
        super().__init__()

    def transform_matric(self, size):
        assert len(size) == 2, f'len(sclae) = {len(size)} != 2'
        translation_axis_matrix = torch.tensor(
            [[1.,     0.,     size[1]],
             [0.,     1.,     size[0]]])

        return translation_axis_matrix

    def __call__(self, inp, mask, scale=None, size=None):
        assert inp.ndim == 4, f'inp.ndim != 4, inp.ndim == {inp.ndim}'
        assert scale is not None or size is not None, \
            'Scale is None and size is None.'
        assert scale is None or size is None, \
            'Ambiguous, scale is not None and size is not None.'

        width = inp.shape[0]
        height = inp.shape[1]

        if scale is not None and not isinstance(scale, (tuple, list)):
            scale = (scale, scale)
        if size is not None and not isinstance(size, (tuple, list)):
            size = (size, size)
        if scale is None:
            scale = (width / size[0],
                     height / size[1])
        scale = (scale[0] * 2, scale[1] * 2)

        affine_matrix = self.transform_matric(scale)

        inp = affine_transform(inp, affine_matrix)
        mask = mask.unsqueeze(1)
        mask = affine_transform(mask, affine_matrix)
        mask = mask.squeeze(1)
        return inp, mask


class RandomTranslate(Transformer):
    def __init__(self, r_min, r_max) -> None:
        super().__init__()
        assert r_max > r_min, \
            f'r_max <= r_min, r_max={r_max} and r_min={r_min}'
        self.r_max = r_max
        self.r_min = r_min
        self.translator = Translate()

    def __call__(self, inp, mask):
        scale = torch.rand(()) * (self.r_max - self.r_min) + self.r_min
        return self.translator(inp, mask, scale=scale)
