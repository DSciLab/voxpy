import numpy as np
from vox.numpy._transform import Transformer


class HistEqual(Transformer):
    def __init__(self, max_val) -> None:
        super().__init__()
        self.max_val = max_val

    def equalize_hist(self, inp, nbins):
        inp_flat = inp.reshape(-1)
        hist, bins = np.histogram(inp_flat, nbins)
        cdf = np.cumsum(hist)
        cdf = self.max_val * cdf / cdf[-1]
        bins = bins[1:]

        output = np.interp(inp_flat, bins, cdf)
        return output.reshape(inp.shape)

    def __call__(self, inp, mask, alpha=0.5, nbins=None):
        nbins = int(nbins or self.max_val)
        hist_out = self.equalize_hist(inp, nbins=nbins)
        output = hist_out * alpha  + inp * (1 - alpha)
        return output, mask
