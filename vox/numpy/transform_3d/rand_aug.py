import numpy as np
import random
from vox.numpy._transform import Transformer

from .sharp import Sharp
from .contrast import HigherContrast, LowerContrast
from .shear import ShearX, ShearY
from .rotate import Rotate
from .translate import TranslateX, TranslateY
from .flip import FlipX, FlipY, FlipZ
from .squeeze import SqueezeX, SqueezeY, SqueezeZ
from .noise import Noise
from .resize import Resize
from .identity import Identity
from .equalization  import HistEqual
from .gaussian_blur import GaussianBlur
from .power_brightness import PowerBrightness
from .sin_brightness import SinBrightness, ArcSinBrightness


__all__ = ['IdentityOp',
           'HistEqualOp',
           'GaussianBlurOp',
           'PowerBrightnessOp',
           'SinBrightnessOp',
           'ArcSinBrightnessOp',
           'NoiseOp',
           'SharpOp',
           'HigherContrastOp',
           'LowerContrastOp',
           'ResizeOp',
           'FlipZOp',
           'FlipYOp',
           'FlipXOp',
           'TranslateXOp',
           'TranslateYOp',
           'ShearXOp',
           'ShearYOp',
           'SqueezeXOp',
           'SqueezeYOp',
           'SqueezeZOp',
           'RotateOp']


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


class SqueezeXOp(TransformerOp):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.transformer = SqueezeX()

    def __call__(self, inp, mask, scale):
        return self.transformer(inp, mask, scale)


class SqueezeYOp(TransformerOp):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.transformer = SqueezeY()

    def __call__(self, inp, mask, scale):
        return self.transformer(inp, mask, scale)


class SqueezeZOp(TransformerOp):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.transformer = SqueezeZ()

    def __call__(self, inp, mask, scale):
        return self.transformer(inp, mask, scale)


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


class PowerBrightnessOp(TransformerOp):
    def __init__(self, opt) -> None:
        super().__init__()
        self.transformer = PowerBrightness()

    def __call__(self, inp, mask, scale):
        return self.transformer(inp, mask, power=scale)


class SinBrightnessOp(TransformerOp):
    def __init__(self, opt) -> None:
        super().__init__()
        self.transformer = SinBrightness()

    def __call__(self, inp, mask, scale):
        return self.transformer(inp, mask, scale=scale)


class ArcSinBrightnessOp(TransformerOp):
    def __init__(self, opt) -> None:
        super().__init__()
        self.transformer = ArcSinBrightness()

    def __call__(self, inp, mask, scale):
        return self.transformer(inp, mask, scale=scale)


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

        assert isinstance(self.N, (tuple, list)) and len(self.N) == 2, \
            f'The type of rand_aug_N should be list or tuple, ' +\
            f'and the length of rand_aug_N should be equal to 2, ' +\
            f'type(rand_aug_N)={type(self.N)} len(rand_aug_N)=' +\
            f'{len(self.N) if isinstance(self.N, (tuple, list)) else None}'

        self.geometry_aug_ops = [
            #   OP       minval      maxval
            (IdentityOp(opt), None, None),
            (HistEqualOp(opt), 0.0, 0.012),
            (GaussianBlurOp(opt), 0.0, 0.6),
            (NoiseOp(opt), 0.0, 0.1),
            (SharpOp(opt), 0.0, 2.5),

            (PowerBrightnessOp(opt), 1.0, 0.7),
            (PowerBrightnessOp(opt), 1.0, 1.3),
            # (SinBrightnessOp(opt), 0.96, 1.0),
            # (ArcSinBrightnessOp(opt), 0.96, 1.0),

            (HigherContrastOp(opt), 0.0, 2.3),
            (LowerContrastOp(opt), 1.0, 1.3)]

        self.color_aug_ops = [
            #   OP       minval      maxval
            (ResizeOp(opt), 1.0, 0.7),
            (ResizeOp(opt), 1.0, 1.3),

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

            (SqueezeXOp(opt), 1.0, 0.8),
            (SqueezeXOp(opt), 1.0, 1.2),
            (SqueezeYOp(opt), 1.0, 0.8),
            (SqueezeYOp(opt), 1.0, 1.2),
            (SqueezeZOp(opt), 1.0, 0.8),
            (SqueezeZOp(opt), 1.0, 1.2),

            (RotateOp(opt), 0.0, np.pi/8),
            (RotateOp(opt), 0.0, -np.pi/8)]

    def __call__(self, inp, mask):
        color_ops = random.sample(self.color_aug_ops, self.N[0])
        geometry_ops = random.sample(self.geometry_aug_ops, self.N[1])
        ops = color_ops + geometry_ops

        for op, minval, maxval in ops:
            if minval is not None and maxval is not None:
                val = float(self.M) * (maxval - minval) + minval
            else:
                val = None
            # print(op.__class__.__name__, '<<', inp.min(), inp.max())
            inp, mask = op(inp, mask, val)
            # print(op.__class__.__name__, '>>', inp.min(), inp.max())
        return inp, mask
