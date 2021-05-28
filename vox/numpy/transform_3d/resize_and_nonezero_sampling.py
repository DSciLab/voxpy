 from typing import List, Tuple, Union
import numpy as np
from vox.numpy._transform import Transformer
from .resize  import Resize
from .nonezero_sampling import NoneZeroSampling
from .random_sampling import RandomSampling
from .nonezero_crop import crop_nonzero


class ResizeAndNoneZeroSampling(Transformer):
    def __init__(self, least_shape: Union[Tuple[int, int, int, int], List[int]],
                 sampling_shape: Union[Tuple[int, int, int, int], List[int]]) -> None:
        super().__init__()
        self.least_shape = least_shape
        self.sampling_shape = sampling_shape
        self.resize = Resize()
        self.sampler = NoneZeroSampling(self.sampling_shape)
        self.random_sampler = RandomSampling(self.sampling_shape)

    def __call__(self, inp: np.ndarray,
                 mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        inp, mask = crop_nonzero(inp, mask)

        inp_shape = inp.shape
        # c = inp_shape[0]
        x = inp_shape[1]
        y = inp_shape[2]
        z = inp_shape[3]

        resize_x = max(x, self.least_shape[1])
        resize_y = max(y, self.least_shape[2])
        resize_z = max(z, self.least_shape[3])
        resize_shape = (resize_x, resize_y, resize_z)

        inp, mask = self.resize(inp, mask, size=resize_shape)
        
        try:
            inp, mask = self.sampler(inp, mask)
        except:
            inp, mask = self.random_sampler(inp, mask)
        
        return inp, mask
