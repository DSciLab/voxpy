from typing import List, Tuple, Union
from ..._transform import PadAndNoneZeroSampling as _PadAndNoneZeroSampling
from .padding import ReflectPad
from .nonezero_sampling import NoneZeroSampling, NoneZeroSamplingXY


class PadAndNoneZeroSampling(_PadAndNoneZeroSampling):
    def __init__(self, least_shape: Union[Tuple[int, int, int, int], List[int]],
                 sampling_shape: Union[Tuple[int, int, int, int], List[int]]) -> None:
        super().__init__(least_shape, sampling_shape,
                         ReflectPad, NoneZeroSampling)


class PadAndNoneZeroSamplingXY(_PadAndNoneZeroSampling):
    def __init__(self, least_shape: Union[Tuple[int, int, int, int], List[int]],
                 sampling_shape: Union[Tuple[int, int, int, int], List[int]]) -> None:
        super().__init__(least_shape, sampling_shape,
                         ReflectPad, NoneZeroSamplingXY)
