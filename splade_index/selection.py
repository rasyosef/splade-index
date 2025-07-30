import numpy as np

try:
    import torch
except ImportError:
    PYTORCH_IS_AVAILABLE = False
else:
    PYTORCH_IS_AVAILABLE = True
    # if Torch is available, we need to initialize it with a dummy scores and capture
    # any output to avoid it from saying that gpu is not available
    _ = torch.topk(torch.as_tensor([0] * 5), 1)


def _topk_numpy(query_scores, k, sorted):
    # https://stackoverflow.com/questions/65038206/how-to-get-indices-of-top-k-values-from-a-numpy-array
    # np.argpartition is faster than np.argsort, but do not return the values in order
    partitioned_ind = np.argpartition(query_scores, -k)
    # Since lit's a single query, we can take the last k elements
    partitioned_ind = partitioned_ind.take(indices=range(-k, 0))
    # We use the newly selected indices to find the score of the top-k values
    partitioned_scores = np.take(query_scores, partitioned_ind)

    if sorted:
        # Since our top-k indices are not correctly ordered, we can sort them with argsort
        # only if sorted=True (otherwise we keep it in an arbitrary order)
        sorted_trunc_ind = np.flip(np.argsort(partitioned_scores))

        # We again use np.take_along_axis as we have an array of indices that we use to
        # decide which values to select
        ind = partitioned_ind[sorted_trunc_ind]
        query_scores = partitioned_scores[sorted_trunc_ind]

    else:
        ind = partitioned_ind
        query_scores = partitioned_scores

    return query_scores, ind


def _topk_torch(query_scores, k):
    topk_scores, topk_indices = torch.topk(torch.as_tensor(query_scores), k)
    topk_scores = topk_scores.numpy()
    topk_indices = topk_indices.numpy()

    return topk_scores, topk_indices


def topk(query_scores, k, backend="auto", sorted=True):
    """
    This function is used to retrieve the top-k results for a single query. It will only work
    on a 1-dimensional array of scores.
    """
    if backend == "auto":
        # if torch is available, use it to speed up selection, otherwise use numpy
        backend = "pytorch" if PYTORCH_IS_AVAILABLE else "numpy"

    if backend not in ["numpy", "pytorch"]:
        raise ValueError("Invalid backend. Please choose from 'numpy' or 'pytorch'.")
    elif backend == "pytorch":
        if not PYTORCH_IS_AVAILABLE:
            raise ImportError(
                "PyTorch is not available. Please install PyTorch with `pip install torch` to use this backend."
            )
        return _topk_torch(query_scores, k)
    else:
        return _topk_numpy(query_scores, k, sorted)
