from typing import List, Optional, Tuple, Union
from cfg import Opts
import numpy as np
from ..._transform import Transformer
from .pad_and_nonezero_sampling import PadAndNoneZeroSampling, \
                                       PadAndNoneZeroSamplingXY
from .pad_and_random_sampling import PadAndRandomSampling


class PadAndGeneralSampling(Transformer):
    def __init__(self, opt: Opts) -> None:
        super().__init__()
        sampler = opt.get('sampler', 'random')
        self.least_shape: Union[Tuple[int, int, int, int], List[int]] = opt.least_shape
        self.sampling_shape: Union[Tuple[int, int, int, int], List[int]] = opt.input_shape

        self.rand_sampler = PadAndRandomSampling(self.least_shape,
                                                 self.sampling_shape)
        if sampler == 'random':
            self.sampler = self.rand_sampler
        elif sampler == 'nonezero':
            self.sampler = PadAndNoneZeroSampling(self.least_shape,
                                                  self.sampling_shape)
        elif sampler == 'nonezero_xy':
            self.sampler = PadAndNoneZeroSamplingXY(self.least_shape,
                                                  self.sampling_shape)
        else:
            raise RuntimeError(f'Unrecognized sampler.')

    def __call__(self, inp: np.ndarray,
                 mask: Optional[np.ndarray]=None
                ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        try:
            return self.sampler(inp, mask)
        except ValueError:
            return self.rand_sampler(inp, mask)
