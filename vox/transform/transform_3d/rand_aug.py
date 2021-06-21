from typing import Optional, Tuple, Union
from cfg import Opts
import numpy as np
import random
from .._transform import Transformer

from .color.gaussian_blur import GaussianBlur
from .color.noise import RicianNoise, GaussianNoise
from .color.sharp import Sharpening
from .color.contrast import Contrast
from .color.gamma import Gamma
from .color.median_filter import MedianFilter
from .color.brightness import BrightnessAdditive, BrightnessMultiplicative

from .geometry.identity import Identity
from .geometry.resize import Resize
from .geometry.shear import ShearX, ShearY
from .geometry.rotate import Rotate, Rotate90
from .geometry.translate import TranslateX, TranslateY
from .geometry.flip import FlipX, FlipY, FlipZ
from .geometry.squeeze import SqueezeX, SqueezeY, SqueezeZ


__all__ = [
    # Geometry Transform Operation
    'IdentityOp',
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
    'RotateOp',
    'Rotate90Op',
    # Color Transform Operation
    'GaussianBlurOp',
    'RicianNoiseOp',
    'GaussianNoiseOp',
    'ContrastOp',
    'AdditiveBrightnessOp',
    'MultiplicativeBrightnessOp',
    'SharpOp'
]


class TransformerOp(object):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self):
        raise NotImplementedError


##################################
#   Geometry transform Operation
##################################

class IdentityOp(TransformerOp):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.transformer = Identity()

    def __call__(self, inp: np.ndarray,
                 mask: np.ndarray, *args, **kwargs
                 ) -> Tuple[np.ndarray, np.ndarray]:
        return self.transformer(inp, mask)


class ResizeOp(TransformerOp):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.transformer = Resize()

    def __call__(self, inp: np.ndarray, mask: np.ndarray,
                 scale: float) -> Tuple[np.ndarray, np.ndarray]:
        return self.transformer(inp, mask, scale=scale)


class FlipZOp(TransformerOp):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.transformer = FlipZ()

    def __call__(self, inp: np.ndarray, mask: np.ndarray,
                 *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        return self.transformer(inp, mask)


class FlipYOp(TransformerOp):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.transformer = FlipY()

    def __call__(self, inp: np.ndarray, mask: np.ndarray,
                 *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        return self.transformer(inp, mask)


class FlipXOp(TransformerOp):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.transformer = FlipX()

    def __call__(self, inp: np.ndarray, mask: np.ndarray,
                 *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        return self.transformer(inp, mask)


class SqueezeXOp(TransformerOp):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.transformer = SqueezeX()

    def __call__(self, inp: np.ndarray, mask: np.ndarray,
                 scale: float) -> Tuple[np.ndarray, np.ndarray]:
        return self.transformer(inp, mask, scale)


class SqueezeYOp(TransformerOp):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.transformer = SqueezeY()

    def __call__(self, inp: np.ndarray, mask: np.ndarray,
                 scale: float) -> Tuple[np.ndarray, np.ndarray]:
        return self.transformer(inp, mask, scale)


class SqueezeZOp(TransformerOp):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.transformer = SqueezeZ()

    def __call__(self, inp: np.ndarray, mask: np.ndarray,
                 scale: float) -> Tuple[np.ndarray, np.ndarray]:
        return self.transformer(inp, mask, scale)


class TranslateXOp(TransformerOp):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.transformer = TranslateX()

    def __call__(self, inp: np.ndarray, mask: np.ndarray,
                 scale: float) -> Tuple[np.ndarray, np.ndarray]:
        return self.transformer(inp, mask, scale=scale)


class TranslateYOp(TransformerOp):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.transformer = TranslateY()

    def __call__(self, inp: np.ndarray, mask: np.ndarray,
                 scale: float) -> Tuple[np.ndarray, np.ndarray]:
        return self.transformer(inp, mask, scale=scale)


class RotateOp(TransformerOp):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.transformer = Rotate()

    def __call__(self, inp: np.ndarray, mask: np.ndarray,
                 scale: float) -> Tuple[np.ndarray, np.ndarray]:
        return self.transformer(inp, mask, theta=scale)


class Rotate90Op(TransformerOp):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.transformer = Rotate90()

    def __call__(self, inp: np.ndarray, mask: np.ndarray,
                 *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        return self.transformer(inp, mask)


class ShearXOp(TransformerOp):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.transformer = ShearX()

    def __call__(self, inp: np.ndarray, mask: np.ndarray,
                 scale: float) -> Tuple[np.ndarray, np.ndarray]:
        return self.transformer(inp, mask, theta=scale)


class ShearYOp(TransformerOp):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.transformer = ShearY()

    def __call__(self, inp: np.ndarray, mask: np.ndarray,
                 scale: float) -> Tuple[np.ndarray, np.ndarray]:
        return self.transformer(inp, mask, theta=scale)


##################################
#   Color transform Operation
##################################

class GaussianBlurOp(TransformerOp):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.transformer = GaussianBlur()

    def __call__(self, inp: np.ndarray, mask: np.ndarray,
                 scale: float) -> Tuple[np.ndarray, np.ndarray]:
        return self.transformer(inp, mask, sigma=scale)


class RicianNoiseOp(TransformerOp):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.transformer = RicianNoise()

    def __call__(self, inp: np.ndarray, mask: np.ndarray,
                 scale: float) -> Tuple[np.ndarray, np.ndarray]:
        return self.transformer(inp, mask)


class GaussianNoiseOp(TransformerOp):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.transformer = GaussianNoise()

    def __call__(self, inp: np.ndarray, mask: np.ndarray,
                 scale: float) -> Tuple[np.ndarray, np.ndarray]:
        return self.transformer(inp, mask)


class ContrastOp(TransformerOp):
    def __init__(self, opt: Opts) -> None:
        super().__init__()
        self.transformer = Contrast()

    def __call__(self, inp: np.ndarray, mask: np.ndarray,
                 scale: float) -> Tuple[np.ndarray, np.ndarray]:
        return self.transformer(inp, mask)



class AdditiveBrightnessOp(TransformerOp):
    def __init__(self, opt: Optional[Opts]=None) -> None:
        super().__init__()
        self.transformer = BrightnessAdditive()

    def __call__(self, inp: np.ndarray, mask: np.ndarray,
                 scale: float) -> Tuple[np.ndarray, np.ndarray]:
        return self.transformer(inp, mask)


class MultiplicativeBrightnessOp(TransformerOp):
    def __init__(self, opt: Optional[Opts]=None) -> None:
        super().__init__()
        self.transformer = BrightnessMultiplicative()

    def __call__(self, inp: np.ndarray, mask: np.ndarray,
                 scale: float) -> Tuple[np.ndarray, np.ndarray]:
        return self.transformer(inp, mask)


class SharpOp(TransformerOp):
    def __init__(self, opt: Opts) -> None:
        super().__init__()
        self.transformer = Sharpening()

    def __call__(self, inp: np.ndarray, mask: np.ndarray,
                 scale: float) -> Tuple[np.ndarray, np.ndarray]:
        return self.transformer(inp, mask)


class ConflictOp(TransformerOp):
    def __init__(self, *ops: TransformerOp) -> None:
        super().__init__()
        for op in ops:
            assert isinstance(op[0], TransformerOp)
        self.ops = ops

    def __call__(self, inp: np.ndarray, mask: np.ndarray,
                 M: float, rand: bool) -> Tuple[np.ndarray, np.ndarray]:
        op, minval, maxval = random.choices(self.ops)[0]
        val = get_aug_value(maxval, minval, M, rand)
        return op(inp, mask, val)

    def __str__(self) -> str:
        string = '<ConflictOp ['
        for op in self.ops:
            string += str(op) + ', '
        string = string[:-2]
        string += ']>'
        return string

    __repr__ = __str__


class RandAugment(Transformer):
    def __init__(self, opt: Opts) -> None:
        super().__init__()
        self.N = opt.rand_aug_N
        self.M = opt.rand_aug_M
        self.rand = opt.get('rand_aug_val', True)

        assert isinstance(self.N, (tuple, list)) and len(self.N) == 2, \
            f'The type of rand_aug_N should be list or tuple, ' +\
            f'and the length of rand_aug_N should be equal to 2, ' +\
            f'type(rand_aug_N)={type(self.N)} len(rand_aug_N)=' +\
            f'{len(self.N) if isinstance(self.N, (tuple, list)) else None}'

        self.color_aug_ops = [
            #   OP       minval      maxval
            (ContrastOp(opt), 0.0, 0.1),

            (ConflictOp((GaussianBlurOp(opt), 0.0, 1.0),
                        (SharpOp(opt), None, None)),
                        None, None),

            (ConflictOp((GaussianNoiseOp(opt), None, None),
                        (RicianNoiseOp(opt), None, None)),
                        None, None),

            (ConflictOp((AdditiveBrightnessOp(opt), None, None),
                        (MultiplicativeBrightnessOp(opt), None, None)),
                        None, None)
        ]

        self.geometry_aug_ops = [
            #   OP       minval      maxval
            (IdentityOp(opt), None, None),
            (ConflictOp((ResizeOp(opt), 1.0, 0.7),
                        (ResizeOp(opt), 1.0, 1.3)),
                        None, None),

            (ConflictOp((TranslateXOp(opt), 0.0, 0.1),
                        (TranslateXOp(opt), 0.0, -0.1),
                        (TranslateYOp(opt), 0.0, 0.1),
                        (TranslateYOp(opt), 0.0, -0.1)),
                        None, None),

            (ConflictOp((ShearXOp(opt), 0.0, 0.2),
                        (ShearXOp(opt), 0.0, -0.2),
                        (ShearYOp(opt), 0.0, 0.2),
                        (ShearYOp(opt), 0.0, -0.2)),
                        None, None),

            (ConflictOp((SqueezeXOp(opt), 1.0, 0.8),
                        (SqueezeXOp(opt), 1.0, 1.2),
                        (SqueezeYOp(opt), 1.0, 0.8),
                        (SqueezeYOp(opt), 1.0, 1.2)),
                        None, None),

            (ConflictOp((RotateOp(opt), 0.0, np.pi/8),
                        (RotateOp(opt), 0.0, -np.pi/8)),
                        None, None),

            (FlipZOp(opt), None, None),
            (FlipYOp(opt), None, None),
            (FlipXOp(opt), None, None),

            (Rotate90Op(opt), None, None)
        ]

    def __call__(self, inp: np.ndarray,
                 mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        geometry_ops = random.sample(self.geometry_aug_ops, self.N[0])
        color_ops = random.sample(self.color_aug_ops, self.N[1])
        ops = color_ops + geometry_ops

        for op, minval, maxval in ops:
            # print(op)
            # print('inp.min()', inp.min())
            # print('inp.max()', inp.max())
            # print('inp.mean()', inp.mean())
            # print('inp.shape', inp.shape)
            # print('=====')
            if isinstance(op, ConflictOp):
                try:
                    inp, mask = op(inp, mask, M=self.M, rand=self.rand)
                except Exception as e:
                    print(op)
                    raise e
            else:
                val = get_aug_value(maxval, minval, self.M, self.rand)
                try:
                    inp, mask = op(inp, mask, val)
                except Exception as e:
                    print(op)
                    raise e
        return inp, mask


def get_aug_value(maxval: Union[float, None], minval: Union[float, None],
                  M: float, rand: Optional[bool]=True) -> Union[float, None]:
    if minval is not None and maxval is not None:
        if not rand:
            val = M * (maxval - minval) + minval
        else:
            val = np.random.uniform(low=minval, high=maxval)
    else:
        val = None
    
    return val
