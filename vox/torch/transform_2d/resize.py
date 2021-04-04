import torch
from vox.torch._transform import Transformer
from .utils import affine_transform


class Resize(Transformer):
    def __init__(self) -> None:
        super().__init__()

    def transform_matric(self, scale):
        assert len(scale) == 2, f'len(sclae) = {len(scale)} != 2'
        resize_axis_matrix = torch.tensor(
            [[1 / scale[1],     0.,            0.],
             [0.,          1 / scale[0],       0.]])
        return resize_axis_matrix

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
            scale = (size[0] / width,
                     size[1] / height)

        affine_matrix = self.transform_matric(scale)

        inp = affine_transform(inp, affine_matrix)
        mask = mask.unsqueeze(1)
        mask = affine_transform(mask, affine_matrix)
        mask = mask.squeeze(1)
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
        scale = torch.rand(()) * (self.r_max - self.r_min) + self.r_min
        return self.resizer(inp, mask, scale=scale)
