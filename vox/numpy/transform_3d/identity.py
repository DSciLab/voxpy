from typing import Optional, Tuple, Union
import numpy as np
from vox.numpy._transform import Transformer


class Identity(Transformer):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, inp: np.ndarray,
                 mask: Optional[np.ndarray]=None
                 ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if mask is not None:
            return inp, mask
        else:
            return inp
