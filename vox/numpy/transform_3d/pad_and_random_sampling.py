from vox.numpy._transform import PadAndRandomSampling as _PadAndRandomSampling
from .padding import ZeroPad, ReflectPad, NearPad
from .random_sampling import RandomSampling


class PadAndRandomSampling(_PadAndRandomSampling):
    def __init__(self, least_shape, sampling_shape) -> None:
        super().__init__(least_shape, sampling_shape, ReflectPad, RandomSampling)
