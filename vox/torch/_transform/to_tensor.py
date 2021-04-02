import torch
import numpy as np
from .base import Transformer


class ToTensor(Transformer):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, inp, mask=False):
        if isinstance(inp, torch.Tensor):
            if mask:
                return inp.type(torch.int64)
            else:
                return inp.type(torch.float)
        elif isinstance(inp, np.ndarray):
            if mask:
                return torch.from_numpy(inp).type(torch.int64)
            else:
                return torch.from_numpy(inp).type(torch.float)
        else:
            raise RuntimeError(
                f'Unrecognized data type ({type(inp)}).')
