import numpy as np


def one_hot(vox_label, num_classes, batch=False):
    vox_shape = vox_label.shape
    vox_label = vox_label.astype(np.int)
    if not batch:
        one_hot_mask = np.zeros((num_classes, *vox_shape))
        vox_label = np.expand_dims(vox_label, axis=0)
        np.put_along_axis(one_hot_mask, vox_label, 1, axis=0)
    else:
        batch_size = vox_label.shape[0]
        one_hot_mask = np.zeros((batch_size, num_classes, *vox_shape[1:]))
        vox_label = np.expand_dims(vox_label, axis=1)
        np.put_along_axis(one_hot_mask, vox_label, 1, axis=1)
    return one_hot_mask


def de_one_hot(one_hot_mask, axis=0):
    return np.argmax(one_hot_mask, axis=axis)
