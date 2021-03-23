import numpy as np
from vox.utils import one_hot
from .grid_view import grid_view
from .seg_mask import seg_mask_cmp, seg_mask_cmp_for_all_classes


def mask_cover(vox_grid, vox_mask_grid, show_class,
               is_one_hoted_mask=False, 
               num_classes=None,
               color=(255, 0, 0),
               color_adjust=100):
    if is_one_hoted_mask:
        one_hot_vox_mask_grid = vox_mask_grid
        num_classes = one_hot_vox_mask_grid.shape[0]
    else:
        if num_classes is None:
            raise ValueError('When is_one_hoted_mask is False,'
                             ' num_classes should be specific.')
        one_hot_vox_mask_grid = one_hot(vox_mask_grid,
                                        num_classes=num_classes,
                                        batch=False)

    assert show_class < num_classes, \
        f'show_class should be less than num_classes, ' +\
        f'but {show_class} >= {num_classes}'

    assert vox_grid.shape[-2:] == one_hot_vox_mask_grid.shape[-2:], \
        f'vox_grid and vox_mask_grid spatial shape are not consistent.'

    assert isinstance(color, (tuple, list)), \
        'color should be a RGB value list or tuple'

    assert len(color) == 3, \
        f'color length should be 3, but len(color)=={len(color)}'

    show_mask = one_hot_vox_mask_grid[show_class]
    color_matrix = np.expand_dims(np.array(color) - color_adjust, axis=(1, 2))

    rgb_mask = show_mask * color_matrix
    vox_grid = np.expand_dims(vox_grid, axis=0)
    covered_vox = vox_grid + rgb_mask

    return np.clip(covered_vox.astype(np.int), 0, 255)


def cmp_cover(vox_grid, seg_grid, mask_grid, show_class, color_adjust=80):
    cmp_rgb_mask = seg_mask_cmp(seg_grid, mask_grid, show_class)
    cmp_rgb_mask[2] = -1 * np.clip(cmp_rgb_mask[0] + cmp_rgb_mask[1], 0, color_adjust)
    vox_grid = np.expand_dims(vox_grid, axis=0)

    covered_vox = vox_grid + cmp_rgb_mask
    return np.clip(covered_vox, 0, 255.0).astype(np.int)


def mask_cover_for_all_classes(vox_grid, vox_mask_grid, show_class,
                               is_one_hoted_mask=False, 
                               num_classes=None,
                               color=(255, 0, 0),
                               color_adjust=100,
                               margin=0,
                               ignore_class=[0]):
    assert isinstance(ignore_class, (int, tuple, list)), \
        f'Type error, ignore_class should be int, ' + \
        f'tuple or list, but {type(ignore_class)} got.'
    
    assert isinstance(margin, (int, tuple, list)), \
        f'Typel error, the type of should be one of int, ' + \
        'list or tuple, but {type(margin)} got.'

    if isinstance(ignore_class, int):
        ignore_class = [ignore_class]

    if isinstance(margin, int):
        margin = (margin, margin)
    else:
        if len(margin) != 2:
            raise ValueError(
                f'The length of margin should be equal to 2, '
                'but margin={margin} got.')

    if is_one_hoted_mask:
        one_hot_vox_mask_grid = vox_mask_grid
        num_classes = one_hot_vox_mask_grid.shape[0]
    else:
        if num_classes is None:
            raise ValueError('When is_one_hoted_mask is False,'
                             ' num_classes should be specific.')
        one_hot_vox_mask_grid = one_hot(vox_mask_grid,
                                        num_classes=num_classes,
                                        batch=False)

    num_classes_to_show = num_classes - len(ignore_class)
    output = None
    k = 0

    for show_class in range(num_classes):
        if show_class in ignore_class:
            continue

        curr_grid = mask_cover(vox_grid, vox_mask_grid, show_class,
                               is_one_hoted_mask=True, 
                               num_classes=num_classes,
                               color=color,
                               color_adjust=color_adjust)
        if output is None:
            sub_grid_size = curr_grid.shape
            output = np.zeros(
                (3, sub_grid_size[0] * num_classes_to_show + margin[0] * (num_classes_to_show + 1),
                sub_grid_size[1] + 2 * margin[1]))

            output[:, margin[0] * (k + 1) + k * sub_grid_size[0]: \
                        (k + 1) * sub_grid_size[0] + margin[0] * (k + 1),
                      margin[1]: sub_grid_size[1] + margin[1]] = curr_grid

    return output.astype(np.int)


def cmp_cover_for_all_classes(vox, seg_vox, mask_vox, layout,
                              color_adjust=80,
                              flatten_axis=-1,
                              margin=0,
                              vox_channel=0,
                              ignore_class=[0]):
    assert seg_vox.shape == mask_vox.shape, \
        f'The shape of seg_vox and the shape of mask_vox doesn\'t match, ' + \
        f'seg_vox.shape={seg_vox.shape} and mask_vox.shape={mask_vox.shape}'
    assert vox.ndim == 4, f'The dim of vox should be equal to 4, vox.ndim={vox.ndim}'
    assert isinstance(margin, (int, tuple, list)), \
        f'Type error, the type of margin is expected to be one of int, tuple or list, ' + \
        f'but {type(margin)}, got.'

    if isinstance(margin, int):
        margin = [margin, margin]

    num_classes = seg_vox.shape[0]
    num_classes_to_show = num_classes - len(ignore_class)

    cmp_rgb_mask = seg_mask_cmp_for_all_classes(seg_vox, mask_vox, layout,
                                                ignore_class,
                                                flatten_axis=flatten_axis,
                                                margin=margin)
    cmp_rgb_mask[2] = -1 * np.clip(cmp_rgb_mask[0] + cmp_rgb_mask[1], 0, color_adjust)
    vox_grid = grid_view(vox, layout, flatten_axis=flatten_axis,
                         channel=vox_channel, margin=margin)
    # repeat vox_grid
    # TODO optimize me
    assert vox_grid.ndim == 2, \
        f'The shape of vox_grid is expected to be equal to 2, ' + \
        f'but vox_grid.ndim={vox_grid.ndim} got.'
    repeated_vox_grid = np.zeros((margin[0] * (num_classes_to_show + 1) + \
                                    num_classes_to_show * vox_grid.shape[0],
                                  margin[1] * 2 + vox_grid.shape[1]))
    for i in range(num_classes_to_show):
        repeated_vox_grid[margin[0] * (i + 1) + i * vox_grid.shape[0]: \
                            margin[0] * (i + 1) + (i + 1) * vox_grid.shape[0],
                          margin[1]: margin[1] + vox_grid.shape[1]] = vox_grid
    repeated_vox_grid = np.expand_dims(repeated_vox_grid, axis=0)

    assert repeated_vox_grid.shape[-2:] == cmp_rgb_mask.shape[-2:], \
        f'The spatial shape of repeated_vox_grid and cmp_rgb_mask doesn\'t match, ' + \
        f'repeated_vox_grid.shape={repeated_vox_grid.shape} and ' + \
        f'cmp_rgb_mask.shape={cmp_rgb_mask.shape}'
    covered_vox = repeated_vox_grid + cmp_rgb_mask
    return np.clip(covered_vox, 0, 255.0).astype(np.int)
