import torch
import numpy as np
from .base import Transformer


class ToNumpyArray(Transformer):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, inp):
        if isinstance(inp, np.ndarray):
            return inp
        elif isinstance(inp, torch.Tensor):
            if not inp.device.type == 'cpu':
                inp = inp.cpu()
            return inp.numpy()
        else:
            raise RuntimeError(
                f'Unrecognized data type ({type(inp)})')
