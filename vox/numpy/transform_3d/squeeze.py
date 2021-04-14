import numpy as np
from scipy.ndimage import affine_transform
from vox.numpy._transform import Transformer


class Squeeze(Transformer):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, inp, mask, theta):
        assert inp.ndim in (3, 4), \
            f'input dim error inp.ndim={inp.ndim}'
        x, y, z = tuple(mask.shape)
        squeeze_matrix = self.transform_matric(theta, x, y, z)

        if inp.ndim == 3:
            inp = affine_transform(inp, squeeze_matrix, order=1)
        else:
            inp_ = []
            for i in range(inp.shape[0]):
                inp_.append(affine_transform(inp[i], squeeze_matrix, order=1))
            inp = np.stack(inp_, axis=0)

        mask = affine_transform(mask, squeeze_matrix, order=0)
        return inp, mask.round()


class SqueezeX(Squeeze):
    @staticmethod
    def transform_matric(theta, x, y, z):
        matrix = np.array(
            [[1. / theta, 0., 0., -x * abs(1 - theta) / theta / 2.],
             [0.,         1., 0.,                               0.],
             [0.,         0., 1.,                               0.],
             [0.,         0., 0.,                               1.]])
        return matrix


class SqueezeY(Squeeze):
    @staticmethod
    def transform_matric(theta, x, y, z):
        matrix = np.array(
            [[1.,         0., 0.,                               0.],
             [0., 1. / theta, 0., -y * abs(1 - theta) / theta / 2.],
             [0.,         0., 1.,                               0.],
             [0.,         0., 0.,                               1.]])
        return matrix


class SqueezeZ(Squeeze):
    @staticmethod
    def transform_matric(theta, x, y, z):
        matrix = np.array(
            [[1., 0.,         0.,                               0.],
             [0., 1.,         0.,                               0.],
             [0., 0., 1. / theta, -z * abs(1 - theta) / theta / 2.],
             [0., 0.,         0.,                               1.]])
        return matrix


class RandomSqueeze(Transformer):
    def __init__(self, r_min, r_max, threhold=0.5, decay=None) -> None:
        super().__init__()
        assert r_max > r_min, \
            f'r_max <= r_min, r_max={r_max} and r_min={r_min}'
        self.r_max = r_max
        self.r_min = r_min
        self.decay = decay
        self.threhold = threhold
        self.squeeze_x = SqueezeX()
        self.squeeze_y = SqueezeY()
        self.squeeze_z = SqueezeZ()

    def update_param(self, verbose=False, *args, **kwargs):
        if self.decay is not None:
            self.r_max *= self.decay
            self.r_min *= self.decay
            if verbose:
                print(f'Update {self.__class__.__name__} parameter to '
                      f'{self.r_min}~{self.r_max}')

    def rand_squeeze_x(self, inp, mask):
        theta = np.random.rand() * (self.r_max - self.r_min) + self.r_min
        return self.squeeze_x(inp, mask, theta)

    def rand_squeeze_y(self, inp, mask):
        theta = np.random.rand() * (self.r_max - self.r_min) + self.r_min
        return self.squeeze_y(inp, mask, theta)

    def rand_squeeze_z(self, inp, mask):
        theta = np.random.rand() * (self.r_max - self.r_min) + self.r_min
        return self.squeeze_z(inp, mask, theta)

    def __call__(self, inp, mask):
        rand = np.random.rand(3)
        if rand[0] > self.threhold:
            inp, mask = self.rand_squeeze_x(inp, mask)

        if rand[1] > self.threhold:
            inp, mask = self.rand_squeeze_y(inp, mask)

        if rand[2] > self.threhold:
            inp, mask = self.rand_squeeze_z(inp, mask)

        return inp, mask
