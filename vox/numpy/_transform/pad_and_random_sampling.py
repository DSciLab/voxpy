from .base import Transformer


class PadAndRandomSampling(Transformer):
    def __init__(self, least_shape, sampling_shape,
                 pad_cls, random_sampling_cls) -> None:
        super().__init__()
        assert isinstance(least_shape, (tuple, list))
        assert isinstance(sampling_shape, (tuple, list))

        self.least_shape = least_shape
        self.sampling_shape = sampling_shape
        self.pad = pad_cls(least_shape)
        self.random_sampling = random_sampling_cls(sampling_shape)

    def __call__(self, inp, mask=None):
        inp, mask = self.pad(inp, mask)
        inp, mask = self.random_sampling(inp, mask)

        if mask is not None:
            return inp, mask
        else:
            return inp
