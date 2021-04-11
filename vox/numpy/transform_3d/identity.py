import numpy as np
from vox.numpy._transform import Transformer


class Identity(Transformer):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, inp, mask):
        return inp, mask
