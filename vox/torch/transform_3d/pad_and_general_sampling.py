from vox.torch._transform import Transformer
from .pad_and_nonezero_sampling import PadAndNoneZeroSampling
from .pad_and_random_sampling import PadAndRandomSampling


class PadAndGeneralSampling(Transformer):
    def __init__(self, opt) -> None:
        super().__init__()
        sampler = opt.get('sampler', 'random')
        self.least_shape = opt.least_shape
        self.sampling_shape = opt.input_shape

        if sampler == 'random':
            self.sampler = PadAndRandomSampling(self.least_shape, self.sampling_shape)
        elif sampler == 'nonezero':
            self.sampler = PadAndNoneZeroSampling(self.least_shape, self.sampling_shape)
        else:
            raise RuntimeError(f'Unrecognized sampler.')

    def __call__(self, inp, mask=None):
        return self.sampler(inp, mask)
