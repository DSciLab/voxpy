from vox._transform import PadAndNoneZeroSampling as _PadAndNoneZeroSampling
from .zero_padding import ZeroPad
from .nonezero_sampling import NoneZeroSampling


class PadAndNoneZeroSampling(_PadAndNoneZeroSampling):
    def __init__(self, least_shape, sampling_shape) -> None:
        super().__init__(least_shape, sampling_shape, ZeroPad, NoneZeroSampling)
