import numpy as np
import random
from vox.numpy._transform import Transformer

from .sharp import Sharp
from .contrast import HigherContrast, LowerContrast
from .shear import ShearX, ShearY
from .rotate import Rotate
from .translate import TranslateX, TranslateY
from .flip import FlipX, FlipY, FlipZ
from .noise import Noise
from .resize import Resize
from .identity import Identity
from .equalization  import HistEqual
from .gaussian_blur import GaussianBlur


__all__ = ['IdentityOps',
           'ResizeOps',
           'HistEqualOps',
           'GaussianBlurOps',
           'NoiseOps',
           'FlipZOps',
           'FlipYOps',
           'FlipXOps',
           'TranslateXOps',
           'TranslateYOps',
           'ShearXOps',
           'ShearYOps',
           'RotateOps',
           'HigherContrastOps',
           'LowerContrastOps',
           'SharpOps']


class TransformerOps(object):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self):
        raise NotImplementedError


class IdentityOps(TransformerOps):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.transformer = Identity()

    def __call__(self, inp, mask, *args, **kwargs):
        return self.transformer(inp, mask)


class ResizeOps(TransformerOps):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.transformer = Resize()

    def __call__(self, inp, mask, scale):
        return self.transformer(inp, mask, scale=scale)


class HistEqualOps(TransformerOps):
    def __init__(self, opt) -> None:
        super().__init__()
        self.transformer = HistEqual(opt.max_val)

    def __call__(self, inp, mask, scale):
        return self.transformer(inp, mask, alpha=scale)


class GaussianBlurOps(TransformerOps):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.transformer = GaussianBlur()

    def __call__(self, inp, mask, scale):
        return self.transformer(inp, mask, sigma=scale)


class NoiseOps(TransformerOps):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.transformer = Noise()

    def __call__(self, inp, mask, scale):
        return self.transformer(inp, mask, std_scale=scale)


class FlipZOps(TransformerOps):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.transformer = FlipZ()

    def __call__(self, inp, mask, *args, **kwargs):
        return self.transformer(inp, mask)


class FlipYOps(TransformerOps):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.transformer = FlipY()

    def __call__(self, inp, mask, *args, **kwargs):
        return self.transformer(inp, mask)


class FlipXOps(TransformerOps):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.transformer = FlipX()

    def __call__(self, inp, mask, *args, **kwargs):
        return self.transformer(inp, mask)


class TranslateXOps(TransformerOps):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.transformer = TranslateX()

    def __call__(self, inp, mask, scale):
        return self.transformer(inp, mask, scale=scale)


class TranslateYOps(TransformerOps):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.transformer = TranslateY()

    def __call__(self, inp, mask, scale):
        return self.transformer(inp, mask, scale=scale)


class RotateOps(TransformerOps):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.transformer = Rotate()

    def __call__(self, inp, mask, scale):
        return self.transformer(inp, mask, theta=scale)


class ShearXOps(TransformerOps):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.transformer = ShearX()

    def __call__(self, inp, mask, scale):
        return self.transformer(inp, mask, theta=scale)


class ShearYOps(TransformerOps):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.transformer = ShearY()

    def __call__(self, inp, mask, scale):
        return self.transformer(inp, mask, theta=scale)


class HigherContrastOps(TransformerOps):
    def __init__(self, opt) -> None:
        super().__init__()
        self.transformer = HigherContrast(opt.max_val)

    def __call__(self, inp, mask, scale):
        return self.transformer(inp, mask, alpha=scale)


class LowerContrastOps(TransformerOps):
    def __init__(self, opt) -> None:
        super().__init__()
        self.transformer = LowerContrast(opt.max_val)

    def __call__(self, inp, mask, scale):
        return self.transformer(inp, mask, alpha=scale)


class SharpOps(TransformerOps):
    def __init__(self, opt) -> None:
        super().__init__()
        denoise_sigma, sharp_sigma = tuple(opt.get('aug_sharp_sigma', [0.1, 0.4]))
        self.denoise_sigma = denoise_sigma
        self.sharp_sigma = sharp_sigma
        self.transformer = Sharp()

    def __call__(self, inp, mask, scale):
        return self.transformer(inp, mask, alpha=scale,
                                denoise_sigma=self.denoise_sigma,
                                sharp_sigma=self.sharp_sigma)


class RandAugment(Transformer):
    def __init__(self, opt) -> None:
        super().__init__()
        self.N = opt.rand_aug_N
        self.M = opt.rand_aug_M
        self.aug_ops = [
            #   OP       minval      maxval
            (IdentityOps(opt), None, None),
            (ResizeOps(opt), 1.0, 0.7),
            (ResizeOps(opt), 1.0, 1.3),
            (HistEqualOps(opt), 0.0, 0.012),
            (GaussianBlurOps(opt), 0, 0.6),
            (NoiseOps(opt), 0, 0.1),
            (FlipZOps(opt), None, None),
            (FlipYOps(opt), None, None),
            (FlipXOps(opt), None, None),
            (TranslateXOps(opt), 0.0, 0.2),
            (TranslateXOps(opt), 0.0, -0.2),
            (TranslateYOps(opt), 0.0, 0.2),
            (TranslateYOps(opt), 0.0, -0.2),
            (ShearXOps(opt), 0.0, 0.3),
            (ShearXOps(opt), 0.0, -0.3),
            (ShearYOps(opt), 0.0, 0.3),
            (ShearYOps(opt), 0.0, -0.3),
            (RotateOps(opt), 0.0, np.pi/6),
            (RotateOps(opt), 0.0, -np.pi/6),
            (HigherContrastOps(opt), 0.0, 2.3),
            (LowerContrastOps(opt), 1.0, 1.3),
            (SharpOps(opt), 0.0, 2.5)]

    def __call__(self, inp, mask):
        ops = random.choices(self.aug_ops, k=self.N)
        for op, minval, maxval in ops:
            if minval is not None and maxval is not None:
                val = float(self.M) * (maxval - minval) + minval
            else:
                val = None
            # print(op.__class__.__name__, '<<',inp.min())
            inp, mask = op(inp, mask, val)
            # print(op.__class__.__name__, '>>', inp.min())
        return inp, mask
