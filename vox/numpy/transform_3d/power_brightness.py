import numpy as np
from vox.numpy._transform import Transformer


class PowerBrightness(Transformer):
    @staticmethod
    def brightness_adjust(inp, power):
        max_inp = np.max(inp)
        min_inp = np.min(inp)
        range_inp = max_inp - min_inp

        inp_1 = (inp - min_inp) / range_inp
        inp_1 = np.power(inp_1, power)
        inp = inp_1 * range_inp
        return inp

    def __call__(self, inp, mask, power):
        inp = self.brightness_adjust(inp, power)
        return inp, mask


class RandomPowerBrightness(Transformer):
    def __init__(self, min_val, max_val) -> None:
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.brightness_adjust = PowerBrightness()

    def __call__(self, inp, mask):
        power = np.random.rand() * (self.max_val - self.min_val) \
                + self.min_val
        return self.brightness_adjust(inp, mask, power=power)
