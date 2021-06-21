from typing import Optional, Union
import numpy as np


def get_range_val(value: Union[list, tuple, float],
                  rand_type: Optional[str]='uniform') -> float:
    if isinstance(value, (list, tuple)):
        if rand_type == 'uniform':
            value = np.random.uniform(value[0], value[1])
        elif rand_type == 'normal':
            value = np.random.normal(value[0], value[1])
        else:
            raise ValueError(
                f'Unrecognized rand_type ({rand_type})')
        return value
    else:
        return value
