import torch
from vox.torch._transform import Transformer


class FlipX(Transformer):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, inp, mask):
        assert inp.ndim in (4, 5), \
            f'input dim error inp.ndim={inp.ndim}'

        if inp.ndim == 5:
            inp = torch.flip(inp, (2,))
        else:
            inp = torch.flip(inp, (1,))

        mask = torch.flip(mask, (1,))
        return inp, mask


class FlipY(Transformer):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, inp, mask):
        assert inp.ndim in (4, 5), \
            f'input dim error inp.ndim={inp.ndim}'

        if inp.ndim == 5:
            inp = torch.flip(inp, (3,))
        else:
            inp = torch.flip(inp, (2,))

        mask = torch.flip(mask, (2,))
        return inp, mask


class FlipZ(Transformer):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, inp, mask):
        assert inp.ndim in (4, 5), \
            f'input dim error inp.ndim={inp.ndim}'

        if inp.ndim == 5:
            inp = torch.flip(inp, (4,))
        else:
            inp = torch.flip(inp, (3,))

        mask = torch.flip(mask, (3,))
        return inp, mask


class RandomFlip(Transformer):
    def __init__(self) -> None:
        super().__init__()
        self.flip_x = FlipX()
        self.flip_y = FlipY()
        self.flip_z = FlipZ()

    def __call__(self, inp, mask):
        rand = torch.rand(3)
        if rand[0] > 0.5:
            inp, mask = self.flip_x(inp, mask)
        if rand[1] > 0.5:
            inp, mask = self.flip_y(inp, mask)
        if rand[2] > 0.5:
            inp, mask = self.flip_z(inp, mask)

        return inp, mask
