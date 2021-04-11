import numpy as np
from numpy.lib.function_base import append
from scipy.ndimage import gaussian_filter
from vox.numpy._transform import Transformer


class Sharp(Transformer):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def sharp(inp, alpha, denoise_sigma, sharp_sigma):
        blurred_inp = gaussian_filter(inp, sigma=denoise_sigma)
        filter_blurred_inp = gaussian_filter(blurred_inp, sigma=sharp_sigma)
        sharpened = inp + alpha * (blurred_inp - filter_blurred_inp)
        sharpened = (sharpened - sharpened.min()) /\
                    (sharpened.max() - sharpened.min()) * sharpened.max()
        return sharpened

    def __call__(self, inp, mask, alpha=1, denoise_sigma=0.1, sharp_sigma=0.4):
        return self.sharp(inp, alpha=alpha, denoise_sigma=denoise_sigma,
                          sharp_sigma=sharp_sigma), mask


class RandomSharp(Transformer):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, inp, mask):
        pass
