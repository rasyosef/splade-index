from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from functools import partial

import os
import logging
from pathlib import Path
import json
from typing import Any, Tuple, Dict, Iterable, List, NamedTuple, Union

import numpy as np

try:
    from .numba import selection as selection_jit
except ImportError:
    selection_jit = None

try:
    from .numba.retrieve_utils import _retrieve_numba_functional
except ImportError:
    _retrieve_numba_functional = None


def _faketqdm(iterable, *args, **kwargs):
    return iterable


if os.environ.get("DISABLE_TQDM", False):
    tqdm = _faketqdm
    # if can't import tqdm, use a fake tqdm
else:
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = _faketqdm


from . import selection
from .version import __version__

logger = logging.getLogger("bm25s")
logger.setLevel(logging.DEBUG)


class Results(NamedTuple):
    """
    NamedTuple with two fields: documents and scores. The `documents` field contains the
    retrieved documents or indices, while the `scores` field contains the scores of the
    retrieved documents or indices.
    """

    documents: np.ndarray
    scores: np.ndarray

    def __len__(self):
        return len(self.documents)

    @classmethod
    def merge(cls, results: List["Results"]) -> "Results":
        """
        Merge a list of Results objects into a single Results object.
        """
        documents = np.concatenate([r.documents for r in results], axis=0)
        scores = np.concatenate([r.scores for r in results], axis=0)
        return cls(documents=documents, scores=scores)


def get_unique_tokens(
    corpus_tokens, show_progress=True, leave_progress=False, desc="Create Vocab"
):
    unique_tokens = set()
    for doc_tokens in tqdm(
        corpus_tokens, desc=desc, disable=not show_progress, leave=leave_progress
    ):
        unique_tokens.update(doc_tokens)
    return unique_tokens


def is_list_of_list_of_type(obj, type_=int):
    if not isinstance(obj, list):
        return False

    if len(obj) == 0:
        return False

    first_elem = obj[0]
    if not isinstance(first_elem, list):
        return False

    if len(first_elem) == 0:
        return False

    first_token = first_elem[0]
    if not isinstance(first_token, type_):
        return False

    return True


class SPLADE:
    def __init__(
        self,
        dtype="float32",
        int_dtype="int32",
        backend="numpy",
    ):
        """
        SPLADE initialization.

        Parameters
        ----------
        dtype : str
            The data type of the BM25 scores.

        int_dtype : str
            The data type of the indices in the BM25 scores.

        backend : str
            The backend used during retrieval. By default, it uses the numpy backend, which
            only requires numpy and scipy as dependencies. You can also select `backend="numba"`
            to use the numba backend, which requires the numba library. If you select `backend="auto"`,
            the function will use the numba backend if it is available, otherwise it will use the numpy
            backend.
        """
        self.dtype = dtype
        self.int_dtype = int_dtype
        self.corpus = None
        self.model = None
        self._original_version = __version__

        if backend == "auto":
            self.backend = "numba" if selection_jit is not None else "numpy"
        else:
            self.backend = backend

    @staticmethod
    def _compute_relevance_from_scores(
        data: np.ndarray,
        indptr: np.ndarray,
        indices: np.ndarray,
        num_docs: int,
        query_token_ids: np.ndarray,
        query_token_weights: np.ndarray,
        dtype: np.dtype,
    ) -> np.ndarray:
        """
        This internal static function calculates the relevance scores for a given query,

        Parameters
        ----------
        data (np.ndarray)
            Data array of the BM25 index.
        indptr (np.ndarray)
            Index pointer array of the BM25 index.
        indices (np.ndarray)
            Indices array of the BM25 index.
        num_docs (int)
            Number of documents in the BM25 index.
        query_token_ids (np.ndarray)
            Array of token IDs to score.
        query_token_weights (np.ndarray)
            Array of token weights.
        dtype (np.dtype)
            Data type for score calculation.

        Returns
        -------
        np.ndarray
            Array of SPLADE relevance scores for a given query.

        Note
        ----
        This function was optimized by the baguetter library. The original implementation can be found at:
        https://github.com/mixedbread-ai/baguetter/blob/main/baguetter/indices/sparse/models/bm25/index.py
        """

        indptr_starts = indptr[query_token_ids]
        indptr_ends = indptr[query_token_ids + 1]

        scores = np.zeros(num_docs, dtype=dtype)
        for i in range(len(query_token_ids)):
            start, end = indptr_starts[i], indptr_ends[i]
            np.add.at(
                scores, indices[start:end], data[start:end] * query_token_weights[i]
            )

            # # The following code is slower with numpy, but faster after JIT compilation
            # for j in range(start, end):
            #     scores[indices[j]] += data[j]

        return scores

    def index(
        self,
        model,
        documents: List[str],
        show_progress=True,
        leave_progress=False,
    ):
        """
        Given a `corpus` of documents, create the SPLADE index.

        show_progress : bool
            If True, a progress bar will be shown. If False, no progress bar will be shown.

        leave_progress : bool
            If True, the progress bars will remain after the function completes.
        """
        import scipy.sparse as sp

        document_embeddings = model.encode_document(documents).coalesce()

        doc_ids = document_embeddings.indices()[0].numpy()
        token_ids = document_embeddings.indices()[1].numpy()
        token_weights = document_embeddings.values().numpy()

        vocab_dict = model.tokenizer.get_vocab()
        num_docs = len(documents)
        n_vocab = len(vocab_dict)

        score_matrix = sp.csc_matrix(
            (token_weights, (doc_ids, token_ids)),
            shape=(num_docs, n_vocab),
            dtype=self.dtype,
        )

        scores = {
            "data": score_matrix.data,
            "indices": score_matrix.indices,
            "indptr": score_matrix.indptr,
            "num_docs": num_docs,
        }

        self.scores = scores
        self.vocab_dict = vocab_dict
        self.model = model

        # we create unique token IDs from the vocab_dict for faster lookup
        self.unique_token_ids_set = set(token_ids)

    def get_scores(
        self, query_token_ids_single: List[int], query_token_weights_single: List[float]
    ) -> np.ndarray:

        data = self.scores["data"]
        indices = self.scores["indices"]
        indptr = self.scores["indptr"]
        num_docs = self.scores["num_docs"]

        dtype = np.dtype(self.dtype)
        int_dtype = np.dtype(self.int_dtype)

        query_tokens_ids_filtered_idx = [
            idx
            for idx, token_id in enumerate(query_token_ids_single)
            if token_id in self.unique_token_ids_set
        ]

        query_token_ids = query_token_ids_single[query_tokens_ids_filtered_idx]
        query_token_weights = query_token_weights_single[query_tokens_ids_filtered_idx]

        scores = self._compute_relevance_from_scores(
            data=data,
            indptr=indptr,
            indices=indices,
            num_docs=num_docs,
            query_token_ids=query_token_ids,
            query_token_weights=query_token_weights,
            dtype=dtype,
        )

        return scores

    def _get_top_k_results(
        self,
        query_token_ids_single: List[int],
        query_token_weights_single: List[float],
        k: int = 1000,
        backend="auto",
        sorted: bool = False,
    ):
        """
        This function is used to retrieve the top-k results for a single query.
        Since it's a hidden function, the user should not call it directly and
        may change in the future. Please use the `retrieve` function instead.
        """
        if len(query_token_ids_single) == 0:
            logger.info(
                msg="The query is empty. This will result in a zero score for all documents."
            )
            scores_q = np.zeros(self.scores["num_docs"], dtype=self.dtype)
        else:
            scores_q = self.get_scores(
                query_token_ids_single, query_token_weights_single
            )

        if backend.startswith("numba"):
            if selection_jit is None:
                raise ImportError(
                    "Numba is not installed. Please install numba to use the numba backend."
                )
            topk_scores, topk_indices = selection_jit.topk(
                scores_q, k=k, sorted=sorted, backend=backend
            )
        else:
            topk_scores, topk_indices = selection.topk(
                scores_q, k=k, sorted=sorted, backend=backend
            )

        return topk_scores, topk_indices

    def retrieve(
        self,
        queries: List[str],
        k: int = 10,
        sorted: bool = True,
        return_as: str = "tuple",
        show_progress: bool = True,
        leave_progress: bool = False,
        n_threads: int = 0,
        chunksize: int = 50,
        backend_selection: str = "auto",
    ):
        """
        Retrieve the top-k documents for each query (tokenized).

        Parameters
        ----------
        query_tokens : List[str]
            List of queries.

        k : int
            Number of documents to retrieve for each query.

        batch_size : int
            Number of queries to process in each batch. Internally, the function will
            process the queries in batches to speed up the computation.

        sorted : bool
            If True, the function will sort the results by score before returning them.

        return_as : str
            If return_as="tuple", a named tuple with two fields will be returned:
            `documents` and `scores`, which can be accessed as `result.documents` and
            `result.scores`, or by unpacking, e.g. `documents, scores = retrieve(...)`.
            If return_as="documents", only the retrieved documents (or indices if `corpus`
            is not provided) will be returned.

        show_progress : bool
            If True, a progress bar will be shown. If False, no progress bar will be shown.

        leave_progress : bool
            If True, the progress bars will remain after the function completes.

        n_threads : int
            Number of jobs to run in parallel. If -1, it will use all available CPUs.
            If 0, it will run the jobs sequentially, without using multiprocessing.

        chunksize : int
            Number of batches to process in each job in the multiprocessing pool.

        backend_selection : str
            The backend to use for the top-k retrieval. Choose from "auto", "numpy", "jax".
            If "auto", it will use JAX if it is available, otherwise it will use numpy.

        weight_mask : np.ndarray
            A weight mask to filter the documents. If provided, the scores for the masked
            documents will be set to 0 to avoid returning them in the results.

        Returns
        -------
        Results or np.ndarray
            If `return_as="tuple"`, a named tuple with two fields will be returned: `documents` and `scores`.
            If `return_as="documents"`, only the retrieved documents (or indices if `corpus` is not provided) will be returned.

        Raises
        ------
        ValueError
            If the `query_tokens` is not a list of list of tokens (str) or a tuple of two lists: the first list is the list of unique token IDs, and the second list is the list of token IDs for each document.

        ImportError
            If the numba backend is selected but numba is not installed.
        """
        num_docs = self.scores["num_docs"]
        if k > num_docs:
            raise ValueError(
                f"k of {k} is larger than the number of available scores"
                f", which is {num_docs} (corpus size should be larger than top-k)."
                f" Please set with a smaller k or increase the size of corpus."
            )
        allowed_return_as = ["tuple", "documents"]

        if return_as not in allowed_return_as:
            raise ValueError("`return_as` must be either 'tuple' or 'documents'")
        else:
            pass

        if n_threads == -1:
            n_threads = os.cpu_count()

        # Embed Queries
        query_embeddings = [
            self.model.encode_query(query).coalesce() for query in queries
        ]

        query_token_ids = [
            query_emb.indices()[0].numpy() for query_emb in query_embeddings
        ]
        query_token_weights = [
            query_emb.values().numpy() for query_emb in query_embeddings
        ]

        tqdm_kwargs = {
            "total": len(query_token_ids),
            "desc": "BM25S Retrieve",
            "leave": leave_progress,
            "disable": not show_progress,
        }
        topk_fn = partial(
            self._get_top_k_results,
            k=k,
            sorted=sorted,
            backend=backend_selection,
        )

        if n_threads == 0:
            # Use a simple map function to retrieve the results
            out = tqdm(
                map(topk_fn, query_token_ids, query_token_weights),
                **tqdm_kwargs,
            )
        else:
            # Use concurrent.futures.ProcessPoolExecutor to parallelize the computation
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                process_map = executor.map(
                    topk_fn,
                    query_token_ids,
                    query_token_weights,
                    chunksize=chunksize,
                )
                out = list(tqdm(process_map, **tqdm_kwargs))

        scores, indices = zip(*out)
        scores, indices = np.array(scores), np.array(indices)

        retrieved_docs = indices

        if return_as == "tuple":
            return Results(documents=retrieved_docs, scores=scores)
        elif return_as == "documents":
            return retrieved_docs
        else:
            raise ValueError("`return_as` must be either 'tuple' or 'documents'")
