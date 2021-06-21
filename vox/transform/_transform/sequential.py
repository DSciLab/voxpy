from .base import Transformer
from .to_tensor import ToTensor
from .to_numpy import ToNumpyArray
from .normalize import Normalize


class Sequantial(Transformer):
    def __init__(self, lst):
        super().__init__()
        self.transform_lst = lst

        # check transform
        for transform in self.transform_lst:
            if not isinstance(transform, Transformer):
                raise ValueError(
                    f'Want a instance of <Transform>,'
                    f' {transform} not a instance of Transform.')

    def append(self, transform):
        if not isinstance(transform, Transformer):
            raise ValueError(
                f'Want a instance of <Transform>,'
                f' {transform} not a instance of Transform.')
        self.transform_lst.append(transform)

    def update_param(self, *args, **kwargs):
        for transformer in self.transform_lst:
            try:
                transformer.update_param(*args, **kwargs)
            except NotImplementedError:
                pass

    def __call__(self, inp, mask=None, vendor_id=None):
        for transform in self.transform_lst:
            if isinstance(transform, (ToNumpyArray, ToTensor)):
                inp = transform(inp)
                if mask is not None:
                    mask = transform(mask, mask=True)
            else:
                try:
                    if isinstance(transform, Normalize):
                        inp, mask = transform(inp, mask, vendor_id=vendor_id)
                    else:
                        inp, mask = transform(inp, mask)
                except ValueError as e:
                    print(transform.__class__.__name__)
                    raise e

        if mask is not None:
            return inp, mask
        else:
            return inp
