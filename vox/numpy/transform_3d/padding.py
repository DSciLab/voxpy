import numpy as np
from numpy.core.fromnumeric import repeat
from vox.numpy._transform import Transformer


def crop_nonzero(inp):
    assert inp.ndim == 4, \
        f'Expect inp number of dim to be 4, ' + \
        f'but inp.ndim={inp.ndim}'
    _, x, y, z = np.where(inp > 0)

    x_max = np.max(x) + 1
    x_min = np.min(x)

    y_max = np.max(y) + 1
    y_min = np.min(y)

    z_max = np.max(z) + 1
    z_min = np.min(z)

    return inp[:, x_min:x_max, y_min:y_max, z_min:z_max]


class _Pad(Transformer):
    def __init__(self, least_shape):
        super().__init__()
        assert len(least_shape) == 4
        # shape (C, X, Y, Z)
        self.least_shape = least_shape

    def get_pad_size(self, inp):
        inp_shape = inp.shape
        assert inp.ndim == len(self.least_shape), \
            'Dim not consistent.'
        assert inp_shape[0] == self.least_shape[0], \
            'Channels not consistent.'
        x = inp_shape[1]
        y = inp_shape[2]
        z = inp_shape[3]

        least_x = self.least_shape[1]
        least_y = self.least_shape[2]
        least_z = self.least_shape[3]

        target_x = max([x, least_x])
        target_y = max([y, least_y])
        target_z = max([z, least_z])

        diff_x = target_x - x
        diff_y = target_y - y
        diff_z = target_z - z

        pad_x = diff_x // 2
        pad_y = diff_y // 2
        pad_z = diff_z // 2

        return target_x, target_y, target_z, pad_x, pad_y, pad_z


class ZeroPad(_Pad):
    def __call__(self, inp, mask=None):
        inp_shape = inp.shape
        c = inp_shape[0]
        x = inp_shape[1]
        y = inp_shape[2]
        z = inp_shape[3]

        target_x, target_y, target_z, pad_x, pad_y, pad_z \
            = self.get_pad_size(inp)
        output = np.zeros((c, target_x, target_y, target_z))
        output[:, pad_x:pad_x + x, \
                  pad_y:pad_y + y, \
                  pad_z:pad_z + z] = inp
        
        if mask is None:
            return output
        else:
            output_mask = np.zeros((target_x, target_y, target_z))
            output_mask[pad_x:pad_x + x, \
                        pad_y:pad_y + y, \
                        pad_z:pad_z + z] = mask

            return output, output_mask


class NearPadX(_Pad):
    def __call__(self, inp, mask=None):
        inp = crop_nonzero(inp)
        inp_shape = inp.shape
        x = inp_shape[1]

        target_x, _, _, pad_x_left, _, _ = self.get_pad_size(inp)

        pad_x_right = target_x - pad_x_left - x
        repeat_x_list = [1] * x
        repeat_x_list[0] = pad_x_left + 1
        repeat_x_list[-1] = pad_x_right + 1

        output = np.repeat(inp, repeat_x_list, axis=1)

        if mask is None:
            return output
        else:
            output_mask = np.repeat(mask, repeat_x_list, axis=1)
            return output, output_mask


class NearPadY(_Pad):
    def __call__(self, inp, mask=None):
        inp = crop_nonzero(inp)
        inp_shape = inp.shape
        y = inp_shape[2]

        _, target_y, _, _, pad_y_left, _ = self.get_pad_size(inp)

        pad_y_right = target_y - pad_y_left - y
        repeat_y_list = [1] * y
        repeat_y_list[0] = pad_y_left + 1
        repeat_y_list[-1] = pad_y_right + 1

        output = np.repeat(inp, repeat_y_list, axis=2)

        if mask is None:
            return output
        else:
            output_mask = np.repeat(mask, repeat_y_list, axis=2)
            return output, output_mask


class NearPadZ(_Pad):
    def __call__(self, inp, mask=None):
        inp = crop_nonzero(inp)
        inp_shape = inp.shape
        z = inp_shape[3]

        _, _, target_z, _, _, pad_z_left = self.get_pad_size(inp)

        pad_z_right = target_z - pad_z_left - z
        repeat_z_list = [1] * z
        repeat_z_list[0] = pad_z_left + 1
        repeat_z_list[-1] = pad_z_right + 1

        output = np.repeat(inp, repeat_z_list, axis=3)

        if mask is None:
            return output
        else:
            output_mask = np.repeat(mask, repeat_z_list, axis=3)
            return output, output_mask


class ReflectPadX(_Pad):
    def __call__(self, inp, mask=None):
        inp = crop_nonzero(inp)
        inp_shape = inp.shape
        x = inp_shape[1]

        target_x, _, _, pad_x_left, _, _ = self.get_pad_size(inp)
        pad_x_right = target_x - pad_x_left - x

        left_part_inp = inp[:, :pad_x_left, :, :]
        left_part_inp = left_part_inp[:, ::-1, :, :]
        right_part_inp = inp[:, -pad_x_right:, :, :]
        right_part_inp = right_part_inp[:, ::-1, :, :]

        output = np.concatenate([left_part_inp, inp, right_part_inp], axis=1)

        if mask is None:
            return output
        else:
            left_part_mask = mask[:, :pad_x_left, :, :]
            left_part_mask = left_part_mask[:, ::-1, :, :]
            right_part_mask = mask[:, -pad_x_right:, :, :]
            right_part_mask = right_part_mask[:, ::-1, :, :]

            output_mask = np.concatenate([left_part_mask,
                                          mask,
                                          right_part_mask], axis=1)
            return output, output_mask


class ReflectPadY(_Pad):
    def __call__(self, inp, mask=None):
        inp = crop_nonzero(inp)
        inp_shape = inp.shape
        y = inp_shape[2]

        _, target_y, _, _, pad_y_left, _ = self.get_pad_size(inp)
        pad_y_right = target_y - pad_y_left - y

        left_part_inp = inp[:, :, :pad_y_left, :]
        left_part_inp = left_part_inp[:, :, ::-1, :]
        right_part_inp = inp[:, :, -pad_y_right:, :]
        right_part_inp = right_part_inp[:, :, ::-1, :]

        output = np.concatenate([left_part_inp, inp, right_part_inp], axis=2)

        if mask is None:
            return output
        else:
            left_part_mask = mask[:, :, :pad_y_left, :]
            left_part_mask = left_part_mask[:, :, ::-1, :]
            right_part_mask = mask[:, :, -pad_y_right:, :]
            right_part_mask = right_part_mask[:, :, ::-1, :]

            output_mask = np.concatenate([left_part_mask,
                                          mask,
                                          right_part_mask], axis=2)
            return output, output_mask


class ReflectPadZ(_Pad):
    def __call__(self, inp, mask=None):
        inp = crop_nonzero(inp)
        inp_shape = inp.shape
        z = inp_shape[3]

        _, _, target_z, _, _, pad_z_left = self.get_pad_size(inp)
        pad_z_right = target_z - pad_z_left - z

        left_part_inp = inp[:, :, :, :pad_z_left]
        left_part_inp = left_part_inp[:, :, :, ::-1]
        right_part_inp = inp[:, :, :, -pad_z_right:]
        right_part_inp = right_part_inp[:, :, :, ::-1]

        output = np.concatenate([left_part_inp, inp, right_part_inp], axis=3)

        if mask is None:
            return output
        else:
            left_part_mask = mask[:, :, :, :pad_z_left]
            left_part_mask = left_part_mask[:, :, :, ::-1]
            right_part_mask = mask[:, :, :, -pad_z_right:]
            right_part_mask = right_part_mask[:, :, :, ::-1]

            output_mask = np.concatenate([left_part_mask,
                                          mask,
                                          right_part_mask], axis=3)
            return output, output_mask
