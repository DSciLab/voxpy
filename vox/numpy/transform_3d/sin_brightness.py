import numpy as np
from vox.numpy._transform import Transformer


class Brightness(Transformer):
    def __call__(self, inp, mask, scale):
        assert scale >= 0.0 and scale <= 1.0
        inp = self.brightness_adjust(inp, scale)
        return inp, mask


class SinBrightness(Brightness):
    @staticmethod
    def brightness_adjust(inp, scale):
        max_inp = np.max(inp)
        min_inp = np.min(inp)
        range_inp = max_inp - min_inp

        inp_1 = (inp - min_inp) / range_inp
        inp_1 = np.sin(inp_1) * scale + inp_1 * (1.0 - scale)
        inp = inp_1 * range_inp
        return inp


class ArcSinBrightness(Brightness):
    @staticmethod
    def brightness_adjust(inp, scale):
        max_inp = np.max(inp)
        min_inp = np.min(inp)
        range_inp = max_inp - min_inp

        inp_1 = (inp - min_inp) / range_inp
        inp_1 = np.arcsin(inp_1) * scale + inp_1 * (1.0 - scale)
        inp = inp_1 * range_inp
        return inp


class RandomSinBrightness(Transformer):
    def __init__(self, min_val, max_val, threhold=1.0) -> None:
        super().__init__()
        threhold = float(threhold)
        assert threhold >= 0.0 and threhold <= 1.0, \
            f'threhold should be in [0, 1], but threold={threhold}.'
        assert max_val > min_val, \
            f'Expect max_val > min_val, but max_val={max_val} ' +\
            f'and min_val={min_val}'
        assert max_val <= 1.0, \
            f'Expect max_val <= 1.0, but max_val={max_val} > 1.0'
        assert min_val >= 0.0, \
            f'Expect min_val >= 0.0, but min_val={min_val} < 0.0'

        self.threhold = threhold
        self.max_val = max_val
        self.min_val = min_val
        self.sin_brightness_adjust = SinBrightness()
        self.arcsin_brightness_adjust = ArcSinBrightness()

    def __call__(self, inp, mask):
        val = np.random.rand()
        if val > self.threhold:
            # do nothing
            return inp, mask

        scale = np.random.rand() * (self.max_val - self.min_val) \
                + self.min_val
        if val > self.threhold / 2:
            return self.sin_brightness_adjust(inp, mask, scale)
        else:
            return self.arcsin_brightness_adjust(inp, mask, scale)
