import numpy as np
from vox.utils import one_hot, threhold_seg
from .grid_view import grid_view_for_all_channel


def seg_mask_cmp(seg_grid, mask_grid, show_class, one_hoted=True, num_classes=None):
    assert one_hoted is True or num_classes is not None, \
        f'When one_hoted is False, num_classes should be specific.'

    if not one_hoted:
        # mask_shape: (W, H)
        # seg_shape: (W, H)
        if mask_grid.ndim != 3:
            mask_grid = one_hot(mask_grid, num_classes=num_classes)
        if seg_grid.ndim != 3:
            seg_grid = one_hot(seg_grid, num_classes=num_classes)
        # one hoted mask and seg
        # mask_shape: (C, W, H)
        # seg_shape: (C, W, H)

    assert seg_grid.shape == mask_grid.shape, \
        f'The seg shape and the mask shape should be consistent, ' +\
        f'but seg.shape ({seg_grid.shape}) != mask.shape ({mask_grid.shape})'

    assert seg_grid.ndim == 3, \
        f'input shape should be (C, W, H), but seg_grid.ndim={seg_grid.ndim}'

    assert show_class < seg_grid.shape[0], \
        f'show class should be less than num_classes, ' +\
        f'but show_class={show_class} and num_classes={seg_grid.shape[0]}'

    seg_selected_class = np.expand_dims(seg_grid[show_class], axis=0)
    mask_selected_class = np.expand_dims(mask_grid[show_class], axis=0)

    red = np.array([[[255.0]], [[0.0]], [[0.0]]])
    green = np.array([[[0.0]], [[255.0]], [[0.0]]])

    red_seg = seg_selected_class * red
    green_mask = mask_selected_class * green

    output = red_seg + green_mask
    return output.astype(np.int)


def seg_mask_cmp_for_all_classes(vox_seg, vox_mask, layout,
                                 ignore_class=[0],
                                 flatten_axis=-1,
                                 margin=0):
    '''
        vox_seg should be one_hoted
        vox_mask should be one_hoted
    '''

    assert vox_mask.shape == vox_seg.shape, \
        f'vox_seg and vox_mask should have same shape, ' + \
        f'vox_seg.shape={vox_seg.shape} and vox_mask.shape={vox_mask.shape}.'

    assert isinstance(ignore_class, (int, tuple, list)), \
        f'Type error, ignore_class should be int, ' + \
        f'tuple or list, but {type(ignore_class)} got.'

    assert vox_mask.ndim == 4, \
        f'The shape of input vox should have 4 dim (C, X, Y, Z), ' + \
        f'but dim={vox_mask.ndim} got.'

    if isinstance(ignore_class, int):
        ignore_class = [ignore_class]

    seg_grid = grid_view_for_all_channel(vox_seg, layout, flatten_axis,
                                         ignore_channels=ignore_class,
                                         margin=margin)
    mask_grid = grid_view_for_all_channel(vox_mask, layout, flatten_axis,
                                         ignore_channels=ignore_class,
                                         margin=margin)

    red = np.array([[[255.0]], [[0.0]], [[0.0]]])
    green = np.array([[[0.0]], [[255.0]], [[0.0]]])

    red_seg = seg_grid * red
    green_mask = mask_grid * green

    output = red_seg + green_mask
    return output.astype(np.int)
