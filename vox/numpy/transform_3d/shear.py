import numpy as np
from scipy.ndimage import affine_transform
from vox.numpy._transform import Transformer


class Shear(Transformer):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, inp, mask, theta):
        assert inp.ndim in (3, 4), \
            f'input dim error inp.ndim={inp.ndim}'

        shear_matrix = self.transform_matric(theta)

        if inp.ndim == 3:
            inp = affine_transform(inp, shear_matrix, order=1)
        else:
            inp_ = []
            for i in range(inp.shape[0]):
                inp_.append(affine_transform(inp[i], shear_matrix, order=1))
            inp = np.stack(inp_, axis=0)

        mask = affine_transform(mask, shear_matrix, order=0)
        return inp, mask.round()


class ShearX(Shear):
    @staticmethod
    def transform_matric(theta):
        matrix = np.array(
            [[1., theta, 0., 0.],
             [0., 1.,    0., 0.],
             [0., 0.,    1., 0.],
             [0., 0.,    0., 1.]])
        return matrix


class ShearY(Shear):
    @staticmethod
    def transform_matric(theta):
        matrix = np.array(
            [[1.,    0., 0., 0.],
             [theta, 1., 0., 0.],
             [0.,    0., 1., 0.],
             [0.,    0., 0., 1.]])
        return matrix


class RandomShear(Transformer):
    def __init__(self, r_min, r_max, threhold=0.5, decay=None) -> None:
        super().__init__()
        assert r_max > r_min, \
            f'r_max <= r_min, r_max={r_max} and r_min={r_min}'
        self.r_max = r_max
        self.r_min = r_min
        self.decay = decay
        self.threhold = threhold
        self.shear_x = ShearX()
        self.shear_y = ShearY()

    def update_param(self, verbose=False, *args, **kwargs):
        if self.decay is not None:
            self.r_max *= self.decay
            self.r_min *= self.decay
            if verbose:
                print(f'Update {self.__class__.__name__} parameter to '
                      f'{self.r_min}~{self.r_max}')

    def rand_shear_x(self, inp, mask):
        theta = np.random.rand() * (self.r_max - self.r_min) + self.r_min
        return self.shear_x(inp, mask, theta)

    def rand_shear_y(self, inp, mask):
        theta = np.random.rand() * (self.r_max - self.r_min) + self.r_min
        return self.shear_y(inp, mask, theta)

    def __call__(self, inp, mask):
        rand = np.random.rand(2)
        if rand[0] > self.threhold:
            inp, mask = self.shear_x(inp, mask)

        if rand[1] > self.threhold:
            inp, mask = self.shear_y(inp, mask)

        return inp, mask
