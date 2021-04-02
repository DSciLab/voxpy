import numpy as np
from scipy.ndimage import affine_transform
from vox._transform import Transformer


class Rotate(object):
    def __init__(self) -> None:
        super().__init__()

    def transform_matric(self, theta, width, height):
        move_axis_matrix = np.array(
            [[1.,     0.,   0.,   width / 2.],
             [0.,     1.,   0.,  height / 2.],
             [0.,     0.,   1.,           0.],
             [0.,     0.,   0.,           1.]])

        move_axis_matrix_back = np.array(
            [[1.,     0.,   0.,   -width / 2.],
             [0.,     1.,   0.,  -height / 2.],
             [0.,     0.,   1.,            0.],
             [0.,     0.,   0.,            1.]])

        rotate_matrix = np.array(
            [[np.cos(theta), -np.sin(theta), 0.,  0.],
             [np.sin(theta), np.cos(theta),  0.,  0.],
             [0.,            0.,             1.,  0.],
             [0.,            0.,             0.,  1.]])

        return move_axis_matrix @ rotate_matrix @ move_axis_matrix_back

    def __call__(self, inp, mask, theta):
        assert inp.ndim in (3, 4), \
            f'input dim error inp.ndim={inp.ndim}'
        width = inp.shape[0]
        height = inp.shape[1]

        affine_matrix = self.transform_matric(theta, width, height)

        if inp.ndim == 3:
            inp = affine_transform(inp, affine_matrix)
        else:
            inp_ = []
            for i in range(inp.shape[0]):
                inp_.append(affine_transform(inp[i], affine_matrix))
            inp = np.stack(inp_, axis=0)

        mask = affine_transform(mask, affine_matrix, order=1)
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
        theta = np.random.rand() * (self.r_max - self.r_min) + self.r_min
        return self.rotater(inp, mask, theta)
