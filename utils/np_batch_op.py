import scipy
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor


def _getindex(arr, i):
    if type(arr) == scipy.sparse.csc_matrix:
        return arr
    else:
        return arr[i] if arr.shape[0] > 1 else arr[0]

def _worker(i):
    f = _worker.f
    arrays = _worker.arrays
    return f(*[_getindex(arr, i) for arr in arrays])

def np_batch_op(f, *arrays):
    get_bs = lambda arr: 1 if type(arr) == scipy.sparse.csc_matrix else arr.shape[0]
    bs = max([get_bs(arr) for arr in arrays])
    _worker.f = f
    _worker.arrays = arrays
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(_worker, range(bs)))
        
    ret = np.concatenate([np.expand_dims(arr, 0) for arr in results], 0)
    return ret
