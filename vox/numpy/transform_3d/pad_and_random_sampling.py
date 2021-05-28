from typing import Union, Tuple, List
from vox.numpy._transform import PadAndRandomSampling as _PadAndRandomSampling
from .padding import ZeroPad, ReflectPad, NearPad
from .random_sampling import RandomSampling


class PadAndRandomSampling(_PadAndRandomSampling):
    def __init__(self, least_shape: Union[Tuple[int, int, int, int], List[int]],
                 sampling_shape: Union[Tuple[int, int, int, int], List[int]]) -> None:
        super().__init__(least_shape, sampling_shape,
                         ReflectPad, RandomSampling)
