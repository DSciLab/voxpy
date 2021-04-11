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


__all__ = ['IdentityOp',
           'ResizeOp',
           'HistEqualOp',
           'GaussianBlurOp',
           'NoiseOp',
           'FlipZOp',
           'FlipYOp',
           'FlipXOp',
           'TranslateXOp',
           'TranslateYOp',
           'ShearXOp',
           'ShearYOp',
           'RotateOp',
           'HigherContrastOp',
           'LowerContrastOp',
           'SharpOp']


class TransformerOp(object):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self):
        raise NotImplementedError


class IdentityOp(TransformerOp):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.transformer = Identity()

    def __call__(self, inp, mask, *args, **kwargs):
        return self.transformer(inp, mask)


class ResizeOp(TransformerOp):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.transformer = Resize()

    def __call__(self, inp, mask, scale):
        return self.transformer(inp, mask, scale=scale)


class HistEqualOp(TransformerOp):
    def __init__(self, opt) -> None:
        super().__init__()
        self.transformer = HistEqual(opt.max_val)

    def __call__(self, inp, mask, scale):
        return self.transformer(inp, mask, alpha=scale)


class GaussianBlurOp(TransformerOp):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.transformer = GaussianBlur()

    def __call__(self, inp, mask, scale):
        return self.transformer(inp, mask, sigma=scale)


class NoiseOp(TransformerOp):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.transformer = Noise()

    def __call__(self, inp, mask, scale):
        return self.transformer(inp, mask, std_scale=scale)


class FlipZOp(TransformerOp):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.transformer = FlipZ()

    def __call__(self, inp, mask, *args, **kwargs):
        return self.transformer(inp, mask)


class FlipYOp(TransformerOp):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.transformer = FlipY()

    def __call__(self, inp, mask, *args, **kwargs):
        return self.transformer(inp, mask)


class FlipXOp(TransformerOp):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.transformer = FlipX()

    def __call__(self, inp, mask, *args, **kwargs):
        return self.transformer(inp, mask)


class TranslateXOp(TransformerOp):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.transformer = TranslateX()

    def __call__(self, inp, mask, scale):
        return self.transformer(inp, mask, scale=scale)


class TranslateYOp(TransformerOp):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.transformer = TranslateY()

    def __call__(self, inp, mask, scale):
        return self.transformer(inp, mask, scale=scale)


class RotateOp(TransformerOp):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.transformer = Rotate()

    def __call__(self, inp, mask, scale):
        return self.transformer(inp, mask, theta=scale)


class ShearXOp(TransformerOp):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.transformer = ShearX()

    def __call__(self, inp, mask, scale):
        return self.transformer(inp, mask, theta=scale)


class ShearYOp(TransformerOp):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.transformer = ShearY()

    def __call__(self, inp, mask, scale):
        return self.transformer(inp, mask, theta=scale)


class HigherContrastOp(TransformerOp):
    def __init__(self, opt) -> None:
        super().__init__()
        self.transformer = HigherContrast(opt.max_val)

    def __call__(self, inp, mask, scale):
        return self.transformer(inp, mask, alpha=scale)


class LowerContrastOp(TransformerOp):
    def __init__(self, opt) -> None:
        super().__init__()
        self.transformer = LowerContrast(opt.max_val)

    def __call__(self, inp, mask, scale):
        return self.transformer(inp, mask, alpha=scale)


class SharpOp(TransformerOp):
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
            (IdentityOp(opt), None, None),
            (ResizeOp(opt), 1.0, 0.7),
            (ResizeOp(opt), 1.0, 1.3),
            (HistEqualOp(opt), 0.0, 0.012),
            (GaussianBlurOp(opt), 0, 0.6),
            (NoiseOp(opt), 0, 0.1),
            (FlipZOp(opt), None, None),
            (FlipYOp(opt), None, None),
            (FlipXOp(opt), None, None),
            (TranslateXOp(opt), 0.0, 0.1),
            (TranslateXOp(opt), 0.0, -0.1),
            (TranslateYOp(opt), 0.0, 0.1),
            (TranslateYOp(opt), 0.0, -0.1),
            (ShearXOp(opt), 0.0, 0.2),
            (ShearXOp(opt), 0.0, -0.2),
            (ShearYOp(opt), 0.0, 0.2),
            (ShearYOp(opt), 0.0, -0.2),
            (RotateOp(opt), 0.0, np.pi/6),
            (RotateOp(opt), 0.0, -np.pi/6),
            (HigherContrastOp(opt), 0.0, 2.3),
            (LowerContrastOp(opt), 1.0, 1.3),
            (SharpOp(opt), 0.0, 2.5)]

    def __call__(self, inp, mask):
        ops = random.sample(self.aug_ops, self.N)
        # print(ops)
        for op, minval, maxval in ops:
            if minval is not None and maxval is not None:
                val = float(self.M) * (maxval - minval) + minval
            else:
                val = None
            # print(op.__class__.__name__, '<<', inp.min(), inp.max())
            inp, mask = op(inp, mask, val)
            # print(op.__class__.__name__, '>>', inp.min(), inp.max())
        return inp, mask
