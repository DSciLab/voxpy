import numpy as np
from vox.numpy._transform import Transformer


class FlipX(Transformer):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, inp, mask):
        assert inp.ndim in (3, 4), \
            f'input dim error inp.ndim={inp.ndim}'

        if inp.ndim == 4:
            inp = inp[:, ::-1, :, :]
        else:
            inp = inp[::-1, :, :]

        mask = mask[::-1, :, :]
        return inp.copy(), mask.copy()


class FlipY(Transformer):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, inp, mask):
        assert inp.ndim in (3, 4), \
            f'input dim error inp.ndim={inp.ndim}'

        if inp.ndim == 4:
            inp = inp[:, :, ::-1, :]
        else:
            inp = inp[:, ::-1, :]

        mask = mask[:, ::-1, :]
        return inp.copy(), mask.copy()


class FlipZ(Transformer):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, inp, mask):
        assert inp.ndim in (3, 4), \
            f'input dim error inp.ndim={inp.ndim}'

        if inp.ndim == 4:
            inp = inp[:, :, :, ::-1]
        else:
            inp = inp[:, :, ::-1]

        mask = mask[:, :, ::-1]
        return inp.copy(), mask.copy()


class RandomFlip(Transformer):
    def __init__(self, threhold=0.5, decay=None) -> None:
        super().__init__()
        self.decay = decay
        self.threhold = threhold
        self.flip_x = FlipX()
        self.flip_y = FlipY()
        self.flip_z = FlipZ()

    def update_param(self, verbose=False, *args, **kwargs):
        if self.decay is not None:
            self.threhold += (1 - self.threhold) * (1 - self.decay)
            if verbose:
                print(f'Update {self.__class__.__name__} parameter to '
                      f'{self.threhold}')

    def __call__(self, inp, mask):
        rand = np.random.rand(3)
        if rand[0] > 0.5:
            inp, mask = self.flip_x(inp, mask)
        if rand[1] > 0.5:
            inp, mask = self.flip_y(inp, mask)
        if rand[2] > 0.5:
            inp, mask = self.flip_z(inp, mask)

        return inp, mask
