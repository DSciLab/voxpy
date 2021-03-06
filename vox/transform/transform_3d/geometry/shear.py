from typing import Optional, Tuple
import numpy as np
from scipy.ndimage import affine_transform
from ..._transform import Transformer


class Shear(Transformer):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, inp: np.ndarray,
                 mask: np.ndarray,
                 theta: float
                 ) -> Tuple[np.ndarray, np.ndarray]:
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
    def transform_matric(theta: float) -> np.ndarray:
        matrix = np.array(
            [[1., theta, 0., 0.],
             [0., 1.,    0., 0.],
             [0., 0.,    1., 0.],
             [0., 0.,    0., 1.]])
        return matrix


class ShearY(Shear):
    @staticmethod
    def transform_matric(theta: float) -> np.ndarray:
        matrix = np.array(
            [[1.,    0., 0., 0.],
             [theta, 1., 0., 0.],
             [0.,    0., 1., 0.],
             [0.,    0., 0., 1.]])
        return matrix


class RandomShear(Transformer):
    def __init__(self, r_min: float, r_max: float,
                 threhold: float=0.5, decay: Optional[float]=None) -> None:
        super().__init__()
        assert r_max > r_min, \
            f'r_max <= r_min, r_max={r_max} and r_min={r_min}'
        self.r_max = r_max
        self.r_min = r_min
        self.decay = decay
        self.threhold = threhold
        self.shear_x = ShearX()
        self.shear_y = ShearY()

    def rand_shear_x(self, inp: np.ndarray,
                     mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        theta = np.random.rand() * (self.r_max - self.r_min) + self.r_min
        return self.shear_x(inp, mask, theta)

    def rand_shear_y(self, inp: np.ndarray,
                     mask: np.ndarray
                     ) -> Tuple[np.ndarray, np.ndarray]:
        theta = np.random.rand() * (self.r_max - self.r_min) + self.r_min
        return self.shear_y(inp, mask, theta)

    def __call__(self, inp: np.ndarray,
                 mask: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray]:
        rand = np.random.rand(2)
        if rand[0] > self.threhold:
            inp, mask = self.rand_shear_x(inp, mask)

        if rand[1] > self.threhold:
            inp, mask = self.rand_shear_y(inp, mask)

        return inp, mask
