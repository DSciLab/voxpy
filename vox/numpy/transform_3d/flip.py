from typing import Optional, Tuple, Union
import numpy as np
from vox.numpy._transform import Transformer


class FlipX(Transformer):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, inp: np.ndarray,
                 mask: Optional[np.ndarray]=None
                 ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        assert inp.ndim in (3, 4), \
            f'input dim error inp.ndim={inp.ndim}'

        if inp.ndim == 4:
            inp = inp[:, ::-1, :, :]
        else:
            inp = inp[::-1, :, :]

        if mask is not None:
            mask = mask[::-1, :, :]
            return inp.copy(), mask.copy()
        else:
            return inp.copy()


class FlipY(Transformer):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, inp: np.ndarray,
                 mask: Optional[np.ndarray]=None
                 ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        assert inp.ndim in (3, 4), \
            f'input dim error inp.ndim={inp.ndim}'

        if inp.ndim == 4:
            inp = inp[:, :, ::-1, :]
        else:
            inp = inp[:, ::-1, :]

        if mask is not None:
            mask = mask[:, ::-1, :]
            return inp.copy(), mask.copy()
        else:
            return inp.copy()


class FlipZ(Transformer):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, inp: np.ndarray,
                 mask: Optional[np.ndarray]=None
                 ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        assert inp.ndim in (3, 4), \
            f'input dim error inp.ndim={inp.ndim}'

        if inp.ndim == 4:
            inp = inp[:, :, :, ::-1]
        else:
            inp = inp[:, :, ::-1]

        if mask is not None:
            mask = mask[:, :, ::-1]
            return inp.copy(), mask.copy()
        else:
            return inp.copy()


class RandomFlip(Transformer):
    def __init__(self, threhold: Optional[float]=0.5,
                 decay: Optional[float]=None) -> None:
        super().__init__()
        self.decay = decay
        self.threhold = threhold
        self.flip_x = FlipX()
        self.flip_y = FlipY()
        self.flip_z = FlipZ()

    def __call__(self, inp: np.ndarray,
                 mask: Optional[np.ndarray]=None
                 ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        rand = np.random.uniform(0, 1, 3)
        if rand[0] > 0.5:
            if mask is None:
                inp = self.flip_x(inp)
            else:
                inp, mask = self.flip_x(inp, mask)
        if rand[1] > 0.5:
            if mask is None:
                inp = self.flip_y(inp)
            else:
                inp, mask = self.flip_y(inp, mask)
        if rand[2] > 0.5:
            if mask is None:
                inp = self.flip_z(inp)
            else:
                inp, mask = self.flip_z(inp, mask)

        if mask is None:
            return inp
        else:
            return inp, mask
