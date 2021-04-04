import torch
from vox.torch._transform import Transformer


def find_none_zero_bbox_single_channel(mask):
    x, y, z = torch.where(mask > 0)
    x_low = torch.min(x)
    x_high = torch.max(x)

    y_low = torch.min(y)
    y_high = torch.max(y)

    z_low = torch.min(z)
    z_high = torch.max(z)

    return ((x_low, x_high), (y_low, y_high), (z_low, z_high))


def merge_bbox(*bboxs):
    final_bbox = None

    for bbox in bboxs:
        if final_bbox is None:
            final_bbox = bbox
        else:
            final_bbox[0][0] = min(final_bbox[0][0], bbox[0][0])
            final_bbox[0][1] = max(final_bbox[0][1], bbox[0][1])

            final_bbox[1][0] = min(final_bbox[1][0], bbox[1][0])
            final_bbox[1][1] = max(final_bbox[1][1], bbox[1][1])

            final_bbox[2][0] = min(final_bbox[2][0], bbox[2][0])
            final_bbox[2][1] = max(final_bbox[2][1], bbox[2][1])
    return final_bbox


def find_none_zero_bbox(mask):
    return find_none_zero_bbox_single_channel(mask)
    # assert mask.ndim == 4

    # C = mask.shape[0]
    # bboxs = []
    # for c in range(C):
    #     mask_c = mask[c, :, :, :]
    #     bboxs.append(find_none_zero_bbox_single_channel(mask_c))
    # return merge_bbox(*bboxs)


class NoneZeroSampling(Transformer):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape     # C * X * Y * Z

    def __call__(self, inp, mask):
        bbox = find_none_zero_bbox(mask)   # X * Y * Z
        inp_shape = inp.shape  # C * X * Y * Z

        x_start_available = (max(0, bbox[0][0] - self.shape[1]), \
                             min(inp_shape[1] - self.shape[1], bbox[0][1]) + 1)
        y_start_available = (max(0, bbox[1][0] - self.shape[2]), \
                             min(inp_shape[2] - self.shape[2], bbox[1][1]) + 1)
        z_start_available = (max(0, bbox[2][0] - self.shape[3]), \
                             min(inp_shape[3] - self.shape[3], bbox[2][1]) + 1)

        x_start = torch.randint(*x_start_available, ())
        y_start = torch.randint(*y_start_available, ())
        z_start = torch.randint(*z_start_available, ())

        return inp[:, x_start: x_start + self.shape[1], \
                      y_start: y_start + self.shape[2], \
                      z_start: z_start + self.shape[3]], \
               mask[x_start: x_start + self.shape[1], \
                    y_start: y_start + self.shape[2], \
                    z_start: z_start + self.shape[3]]
