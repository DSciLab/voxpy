import torch
from .utils import affine_transform
from vox._transform import Transformer


class Rotate(object):
    def __init__(self) -> None:
        super().__init__()

    def transform_matric(self, theta):
        if not isinstance(theta, torch.Tensor):
            theta = torch.tensor(theta)
        rotate_matrix = torch.tensor(
            [[torch.cos(theta),  torch.sin(theta), 0.],
             [-torch.sin(theta), torch.cos(theta),  0.]])
        return rotate_matrix

    def __call__(self, inp, mask, theta):
        assert inp.ndim == 4, \
            f'input dim error inp.ndim={inp.ndim}'

        affine_matrix = self.transform_matric(theta)
        inp = affine_transform(inp, affine_matrix)
        mask = mask.unsqueeze(1)
        mask = affine_transform(mask, affine_matrix)
        mask = mask.squeeze(1)

        return inp, mask


class RandomRotate(Transformer):
    def __init__(self, r_min, r_max) -> None:
        super().__init__()
        assert r_max > r_min, \
            f'r_max <= r_min, r_max={r_max} and r_min={r_min}'
        self.r_max = r_max
        self.r_min = r_min
        self.rotater = Rotate()

    def __call__(self, inp, mask):
        theta = torch.rand(()) * (self.r_max - self.r_min) + self.r_min
        return self.rotater(inp, mask, theta)
