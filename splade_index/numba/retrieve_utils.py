import os
from numba import njit, prange
import numpy as np
from typing import List, Tuple, Any
import logging

from .. import utils
from .selection import _numba_sorted_top_k


@njit()
def _compute_relevance_from_scores_jit_ready(
    data: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    num_docs: int,
    query_tokens_ids: np.ndarray,
    query_tokens_weights: np.ndarray,
    dtype: np.dtype,
) -> np.ndarray:
    """
    This internal static function calculates the relevance scores for a given query,
    by using the SPLADE scores that have been precomputed in the index.
    This version is ready for JIT compilation with numba, but is slow if not compiled.
    """
    indptr_starts = indptr[query_tokens_ids]
    indptr_ends = indptr[query_tokens_ids + 1]

    scores = np.zeros(num_docs, dtype=dtype)
    for i in range(len(query_tokens_ids)):
        start, end = indptr_starts[i], indptr_ends[i]
        # The following code is slower with numpy, but faster after JIT compilation
        for j in range(start, end):
            scores[indices[j]] += data[j] * query_tokens_weights[i]

    return scores


@njit(parallel=True)
def _retrieve_internal_jitted_parallel(
    query_tokens_ids_flat: np.ndarray,
    query_tokens_weights_flat: np.ndarray,
    query_pointers: np.ndarray,
    k: int,
    sorted: bool,
    dtype: np.dtype,
    int_dtype: np.dtype,
    data: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    num_docs: int,
):
    N = len(query_pointers) - 1

    topk_scores = np.zeros((N, k), dtype=dtype)
    topk_indices = np.zeros((N, k), dtype=int_dtype)

    for i in prange(N):
        query_tokens_single = query_tokens_ids_flat[
            query_pointers[i] : query_pointers[i + 1]
        ]

        query_tokens_weights_single = query_tokens_weights_flat[
            query_pointers[i] : query_pointers[i + 1]
        ]

        # query_tokens_single = np.asarray(query_tokens_single, dtype=int_dtype)
        scores_single = _compute_relevance_from_scores_jit_ready(
            query_tokens_ids=query_tokens_single,
            query_tokens_weights=query_tokens_weights_single,
            data=data,
            indptr=indptr,
            indices=indices,
            num_docs=num_docs,
            dtype=dtype,
        )

        topk_scores_sing, topk_indices_sing = _numba_sorted_top_k(
            scores_single, k=k, sorted=sorted
        )
        topk_scores[i] = topk_scores_sing
        topk_indices[i] = topk_indices_sing

    return topk_scores, topk_indices


def _retrieve_numba_functional(
    query_tokens_ids,
    query_tokens_weights,
    scores,
    corpus: List[Any] = None,
    k: int = 10,
    sorted: bool = True,
    return_as: str = "tuple",
    show_progress: bool = True,
    leave_progress: bool = False,
    n_threads: int = 0,
    chunksize: int = None,
    backend_selection="numba",
    dtype="float32",
    int_dtype="int32",
):
    from numba import get_num_threads, set_num_threads, njit

    if backend_selection != "numba":
        error_msg = "The `numba` backend must be selected when retrieving using the numba backend. Please choose a different backend or change the backend_selection parameter to numba."
        raise ValueError(error_msg)

    if chunksize != None:
        # warn the user that the chunksize parameter is ignored
        logging.warning(
            "The `chunksize` parameter is ignored in the `retrieve` function when using the `numba` backend."
            "The function will automatically determine the best chunksize."
        )

    allowed_return_as = ["tuple", "documents"]

    if return_as not in allowed_return_as:
        raise ValueError("`return_as` must be either 'tuple' or 'documents'")
    else:
        pass

    if n_threads == -1:
        n_threads = os.cpu_count()
    elif n_threads == 0:
        n_threads = 1

    # get og thread count
    og_n_threads = get_num_threads()
    set_num_threads(n_threads)

    # convert query_tokens_ids from list of list to a flat 1-d np.ndarray with
    # pointers to the start of each query to be used to find the boundaries of each query
    query_pointers = np.cumsum(
        [0] + [len(q) for q in query_tokens_ids], dtype=int_dtype
    )
    query_tokens_ids_flat = np.concatenate(query_tokens_ids).astype(int_dtype)
    query_tokens_weights_flat = np.concatenate(query_tokens_weights).astype(dtype)

    retrieved_scores, retrieved_indices = _retrieve_internal_jitted_parallel(
        query_pointers=query_pointers,
        query_tokens_ids_flat=query_tokens_ids_flat,
        query_tokens_weights_flat=query_tokens_weights_flat,
        k=k,
        sorted=sorted,
        dtype=np.dtype(dtype),
        int_dtype=np.dtype(int_dtype),
        data=scores["data"],
        indptr=scores["indptr"],
        indices=scores["indices"],
        num_docs=scores["num_docs"],
    )

    # reset the number of threads
    set_num_threads(og_n_threads)

    retrieved_docs = retrieved_indices

    if return_as == "tuple":
        return retrieved_docs, retrieved_scores
    elif return_as == "documents":
        return retrieved_docs
    else:
        raise ValueError("`return_as` must be either 'tuple' or 'documents'")
