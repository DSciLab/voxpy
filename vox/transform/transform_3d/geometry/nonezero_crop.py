from typing import Tuple
import numpy as np


def crop_nonzero(inp: np.ndarray,
                 mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert inp.ndim == 4, \
        f'Expect inp number of dim to be 4, ' + \
        f'but inp.ndim={inp.ndim}'
    _, x, y, z = np.where(inp > 0)

    x_max = np.max(x) + 1
    x_min = np.min(x)

    y_max = np.max(y) + 1
    y_min = np.min(y)

    z_max = np.max(z) + 1
    z_min = np.min(z)

    inp = inp[:, x_min:x_max, y_min:y_max, z_min:z_max]
    mask = mask[x_min:x_max, y_min:y_max, z_min:z_max]
    return inp, mask
