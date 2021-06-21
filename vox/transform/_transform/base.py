class Transformer(object):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, inp):
        raise NotImplementedError

    def update_param(self, *args, **kwargs):
        raise NotImplementedError
