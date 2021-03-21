import numpy as np
import torch


def numpy_to_tensor(inp):
    if isinstance(inp, torch.Tensor):
        return inp
    elif isinstance(inp, np.ndarray):
        return torch.from_numpy(inp)
    else:
        raise ValueError(
            f'Unrecognized data type ({type(inp)})')


def tensor_to_numpy(inp):
    if isinstance(inp, torch.Tensor):
        return inp.numpy()
    elif isinstance(inp, np.ndarray):
        return inp
    else:
        raise ValueError(
            f'Unrecognized data type ({type(inp)})')
