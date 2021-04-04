import torch
from vox.torch._transform import Transformer


class FixChannels(Transformer):
    def __init__(self) -> None:
        super().__init__()

    def fix_channels(self, inp, mask=False):
        if mask:
            return inp

        if inp.ndim == 3:
            return inp
        elif inp.ndim == 2:
            return torch.unsqueeze(inp, dim=0)
        else:
            raise RuntimeError(
                f'Unrecognized input dim ({inp.ndim}/{inp.shape})')

    def __call__(self, inp, mask=None):
        if mask is None:
            return self.fix_channels(inp)
        else:
            return self.fix_channels(inp), self.fix_channels(mask, mask=True)
