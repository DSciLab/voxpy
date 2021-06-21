from typing import Union, Tuple, List
from ..._transform import PadAndRandomSampling as _PadAndRandomSampling
from .padding import ReflectPad
from .random_sampling import RandomSampling


class PadAndRandomSampling(_PadAndRandomSampling):
    def __init__(self, least_shape: Union[Tuple[int, int, int, int], List[int]],
                 sampling_shape: Union[Tuple[int, int, int, int], List[int]]) -> None:
        super().__init__(least_shape, sampling_shape,
                         ReflectPad, RandomSampling)
