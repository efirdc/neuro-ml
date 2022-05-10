import sys
import os
from typing import Dict, Any, Sequence, Tuple

import h5py
import numpy as np
import torch
from numpy.typing import ArrayLike


def is_sequence(x):
    if isinstance(x, np.ndarray) or isinstance(x, torch.Tensor):
        return True
    if isinstance(x, Sequence) and not isinstance(x, str):
        return True
    return False


def index_unsorted(x: ArrayLike, indices: ArrayLike):
    """
    Returns x[indices].
    Use if x is an array type where indices must be strictly increasing (i.e. an h5py.Dataset).
    This will sort indices and then remove duplicates before indexing,
    The resulted array is then "unsorted" and "repeated" to respect the original indices.
    """
    unique_indices, inverse_indices = np.unique(indices, return_inverse=True)
    return x[unique_indices][inverse_indices]


def require_dataset(group: h5py.Group, key: str, data: ArrayLike):
    if key in group:
        group[key][:] = data
    else:
        group[key] = data


def merge_dicts(source: Dict, dest: Dict):
    for k, v in source.items():
        if isinstance(v, Dict):
            if k not in dest:
                dest[k] = {}
            merge_dicts(source[k], dest[k])
        else:
            if k in dest:
                raise ValueError()
            else:
                dest[k] = v


def nested_insert(d: Dict, keys: Sequence[Any], value: Any):
    for k in keys[:-1]:
        if k not in d:
            d[k] = {}
        assert isinstance(d[k], Dict)
        d = d[k]

    final_key = keys[-1]
    if final_key not in d:
        d[final_key] = value

    elif isinstance(d[final_key], Dict):
        assert isinstance(value, Dict)
        merge_dicts(value, d[final_key])

    else:
        d[final_key] = value


def nested_select(d: Dict, keys: Sequence[Any]):
    key = keys[0]
    if key is None:
        key = list(d.keys())
    if is_sequence(key):
        return [nested_select(d[k], keys[1:]) for k in key]
    elif len(keys) > 1:
        return nested_select(d[key], keys[1:])
    else:
        return d[key]


def get_data_iterator(loader):
    while True:
        for batch in loader:
            yield batch


def reconstruct_volume(
        data: torch.Tensor,
        volume_shape: Tuple[int, int, int],
        indices: torch.Tensor,
        fill_value: Any = 0.
):
    volume = torch.full(volume_shape, fill_value, dtype=data.dtype)
    volume[indices[:, 0], indices[:, 1], indices[:, 2]] = data
    return volume


def product(seq: Sequence):
    out = 1
    for elem in seq:
        out *= elem
    return out


class DisablePrints:
    """
    Context for disabling print statements.
    Useful when you want an overly-verbose package to be quiet
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
