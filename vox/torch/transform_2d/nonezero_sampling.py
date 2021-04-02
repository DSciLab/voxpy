import torch
from vox._transform import Transformer


def find_none_zero_bbox_single_channel(mask):
    x, y = torch.where(mask > 0)
    x_low = torch.min(x)
    x_high = torch.max(x)

    y_low = torch.min(y)
    y_high = torch.max(y)

    return ((x_low, x_high), (y_low, y_high))


def find_none_zero_bbox(mask):
    assert mask.ndim == 2
    return find_none_zero_bbox_single_channel(mask)


class NoneZeroSampling(Transformer):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape     # C * X * Y

    def __call__(self, inp, mask):
        bbox = find_none_zero_bbox(mask)   # X * Y
        inp_shape = inp.shape  # C * X * Y

        x_start_available = (max(0, bbox[0][0] - self.shape[1]), \
                             min(inp_shape[1] - self.shape[1], bbox[0][1]))
        y_start_available = (max(0, bbox[1][0] - self.shape[2]), \
                             min(inp_shape[2] - self.shape[2], bbox[1][1]))

        x_start = torch.randint(*x_start_available, ())
        y_start = torch.randint(*y_start_available, ())

        return inp[:, x_start: x_start + self.shape[1], \
                      y_start: y_start + self.shape[2]], \
               mask[x_start: x_start + self.shape[1], \
                    y_start: y_start + self.shape[2]]
