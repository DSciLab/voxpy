import numpy as np
from .one_hot import one_hot
from .rescale import LinearNormRescale255, \
                     CentralNormRescale255, \
                     GeneralNormRescale255


def threhold_seg(inp, th=0.5):
    inp_ = np.copy(inp)
    inp_[inp_>0.5] = 1.0
    inp_[inp_<=0.5] = 0.0
    return inp_
