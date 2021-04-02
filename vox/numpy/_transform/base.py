class Transformer(object):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, inp):
        raise NotImplementedError
