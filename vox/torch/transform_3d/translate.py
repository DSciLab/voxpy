import torch
from .utils import affine_transform
from vox._transform import Transformer


class Translate(Transformer):
    OFFSET = 1.0e6
    def __init__(self) -> None:
        super().__init__()

    def transform_matric(self, scale):
        assert len(scale) == 3, f'len(scale) = {len(scale)} != 3'
        translation_axis_matrix = torch.tensor(
            [[1.,     0.,     0.,     scale[2]],
             [0.,     1.,     0.,     scale[1]],
             [0.,     0.,     1.,     scale[0]]])

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
        if scale is None:
            scale = (size[0] / width,
                     size[1] / height,
                     size[2] / depth)
        scale = (scale[0] * 2, scale[1] * 2, scale[2] * 2)

        affine_matrix = self.transform_matric(scale)

        inp = affine_transform(inp, affine_matrix)
        mask = mask.unsqueeze(1)
        mask = affine_transform(mask, affine_matrix)
        mask = mask.squeeze(1)
        return inp, mask


class RandomTranslate(Transformer):
    def __init__(self, r_min, r_max, depth_translate=False) -> None:
        super().__init__()
        assert r_max > r_min, \
            f'r_max <= r_min, r_max={r_max} and r_min={r_min}'
        self.r_max = r_max
        self.r_min = r_min
        self.depth_translate = depth_translate
        self.translator = Translate()

    def __call__(self, inp, mask):
        scale = torch.rand(()) * (self.r_max - self.r_min) + self.r_min
        if not self.depth_translate:
            scale = (scale, scale, 0)

        return self.translator(inp, mask, scale=scale)
