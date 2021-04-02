from .base import Transformer
from .to_tensor import ToTensor
from .to_numpy import ToNumpyArray


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

    def __call__(self, inp, mask=None):
        for transform in self.transform_lst:
            if isinstance(transform, (ToNumpyArray, ToTensor)):
                inp = transform(inp)
                if mask is not None:
                    mask = transform(mask, mask=True)
            else:
                try:
                    inp, mask = transform(inp, mask)
                except ValueError as e:
                    print(transform.__class__.__name__)
                    raise e

        if mask is not None:
            return inp, mask
        else:
            return inp
