import numpy as np
from numpy.core.fromnumeric import size
from .utils import tensor_to_numpy


def grid_view(vox, layout, flatten_axis=-1, channel=0, margin=0):
    assert isinstance(layout, (list, tuple)), \
        f'Type error, layout should be a two element tuple, but {type(layout)} got.'
    assert isinstance(flatten_axis, int), \
        f'Type error, flatten_axis should be a int, but {type(flatten_axis)} got.'
    assert isinstance(channel, int), \
        f'Type error, channel should be a int,  but {type(channel)} got.'
    assert isinstance(margin, (int, tuple, list)), \
        f'Type error, margin should be a int, list or tuple, but {type(channel)} got.'

    if isinstance(margin, int):
        margin = (margin, margin)

    vox = tensor_to_numpy(vox)
    assert vox.ndim == 3 or vox.ndim == 4, f'please check vox dim. vox dim={vox.ndim}'

    if vox.ndim == 4:
        vox = vox[channel, :, :, :]

    return make_grid(vox, layout, flatten_axis, margin)


def grid_view_for_all_channel(vox, layout, flatten_axis=-1, ignore_channels=[0], margin=0):
    assert isinstance(ignore_channels, (int, tuple, list)), \
        f'Type error, the type of ignore_channels should be one of int, tuple or list, ' + \
        f'but {type(ignore_channels)} got.'
    assert vox.ndim == 4, f'vox should have 4 dim (C, X, Y, Z), but vox.ndim={vox.ndim} got.'
    assert isinstance(margin, (int, tuple, list)), \
        'Type error, the type of margin should be one of int, tuple or list, ' + \
        f'but {type(margin)} got.'

    num_channels = vox.shape[0]
    num_show_channels = num_channels - len(ignore_channels)
    output = None
    k = 0

    for i in range(num_channels):
        if i in ignore_channels:
            continue

        sub_grid = grid_view(vox, layout, flatten_axis, channel=i, margin=margin)          
        if output is None:
            size_of_sub_grid = sub_grid.shape
            output = np.zeros((num_show_channels * size_of_sub_grid[0] + margin[0] * 2,
                               size_of_sub_grid[1] + margin[1] * 2))

        output[(k + 1) * margin[0] + k * size_of_sub_grid[0]: \
                (k + 1) * margin[0] + (k + 1) * size_of_sub_grid[0],
               margin[1]: margin[1] + size_of_sub_grid[1]] = sub_grid

        k += 1

    return output.astype(np.int)


def make_grid(vox, layout, flatten_axis=-1, margin=0):
    """
    vox: A 3D numpy array
    """
    assert isinstance(vox, np.ndarray), \
        f'Type error, wanted np.ndarray, but {type(vox)} got.'
    assert vox.ndim == 3, \
        f'vox should be a 3D numpy array, but {vox.ndim}D array got.'
    if flatten_axis not in [0, 1, 2, -1, -2, -3]:
        raise ValueError(
            f'flatten_axis should be 0, 1, 2, -1, -2 or -3, but {flatten_axis} got.')
    if isinstance(margin, int):
        margin = (margin, margin)

    num_sub_figure = np.cumprod(layout)[-1]
    if flatten_axis < 0:
        flatten_axis = 3 + flatten_axis

    if flatten_axis == 1:
        vox = np.transpose(vox, (1, 0, 2))
    elif flatten_axis == 2:
        vox = np.transpose(vox, (2, 1, 0))
    else:
        # do nothing
        pass

    size = vox.shape[1:]
    output = np.zeros((layout[0] * size[0] + margin[0] * (layout[0] + 1),\
                       layout[1] * size[1] + margin[1] * (layout[1] + 1)))

    k = 0
    skip_frame = min(vox.shape[0] // num_sub_figure, 1)

    for i in range(layout[0]):
        for j in range(layout[1]):
            output[i * size[0] + margin[0] * (i + 1): (i + 1) * size[0] + margin[0] * (i + 1), \
                   j * size[1] + margin[1] * (j + 1): (j + 1) * size[1] + margin[1] * (j + 1)] = vox[k]
            k += skip_frame
            if k >= vox.shape[0]:
                break
    return output
