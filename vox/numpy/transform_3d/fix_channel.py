from typing import Optional, Tuple, Union
import numpy as np
from vox.numpy._transform import Transformer


class FixChannels(Transformer):
    def __init__(self) -> None:
        super().__init__()

    def fix_channels(self, inp: np.ndarray,
                     is_mask: Optional[bool]=False) -> np.ndarray:
        if is_mask:
            return inp

        if inp.ndim == 4:
            return inp
        elif inp.ndim == 3:
            return np.expand_dims(inp, axis=0)
        else:
            raise RuntimeError(
                f'Unrecognized input dim ({inp.ndim}/{inp.shape})')

    def __call__(self, inp: np.ndarray,
                 mask: Optional[np.ndarray]=None
                 ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if mask is None:
            return self.fix_channels(inp)
        else:
            return self.fix_channels(inp), mask
