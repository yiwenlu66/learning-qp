import scipy
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor


def _getindex(arr, i):
    """
    Retrieves the ith element of an array, or the entire array if it's a scipy sparse matrix.

    Parameters:
    arr (np.ndarray or scipy.sparse.csc_matrix): The array or sparse matrix.
    i (int): Index of the element to retrieve.

    Returns:
    np.ndarray or scipy.sparse.csc_matrix: The ith element of the array or the entire array if it's a sparse matrix.
    """
    if type(arr) == scipy.sparse.csc_matrix:
        return arr
    else:
        return arr[i] if arr.shape[0] > 1 else arr[0]

def _worker(i):
    """
    Worker function to apply the function 'f' on slices of arrays for parallel processing.

    Parameters:
    i (int): The index representing which slice of the arrays to process.

    Returns:
    tuple: A tuple of results returned by the function 'f'.
    """
    f = _worker.f
    arrays = _worker.arrays
    results = f(*[_getindex(arr, i) for arr in arrays])
    return results if isinstance(results, tuple) else (results,)

def np_batch_op(f, *arrays, max_workers=int(os.environ.get("MAX_CPU_WORKERS", 8))):
    """
    Applies a function in a batch operation on multiple arrays, possibly in parallel, handling multiple return values.
    If the function 'f' returns a single value, the function returns a single concatenated value instead of a tuple.

    Parameters:
    f (callable): The function to apply. Can return multiple values.
    arrays (list of np.ndarray or scipy.sparse.csc_matrix): Arrays on which the function is to be applied.

    Returns:
    np.ndarray or tuple: A concatenated array if 'f' returns a single value, otherwise a tuple of concatenated arrays.
    """
    get_bs = lambda arr: 1 if type(arr) == scipy.sparse.csc_matrix else arr.shape[0]
    bs = max([get_bs(arr) for arr in arrays])
    _worker.f = f
    _worker.arrays = arrays

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        all_results = list(executor.map(_worker, range(bs)))

    processed_results = []
    for i in range(len(all_results[0])):
        results = [result[i] for result in all_results]
        if isinstance(results[0], np.ndarray):
            processed_result = np.concatenate([np.expand_dims(arr, 0) for arr in results], 0)
        else:
            processed_result = np.array(results)
        processed_results.append(processed_result)

    # Return a single value if there's only one result, otherwise return a tuple
    return processed_results[0] if len(processed_results) == 1 else tuple(processed_results)
