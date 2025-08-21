import warnings

from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from functools import partial

import os
import logging
from pathlib import Path
import json
from typing import Any, Tuple, Dict, Iterable, List, NamedTuple, Union, Literal

import numpy as np

from .utils import json_functions as json_functions

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

    doc_ids: np.ndarray
    documents: np.ndarray
    scores: np.ndarray

    def __len__(self):
        return len(self.documents)

    @classmethod
    def merge(cls, results: List["Results"]) -> "Results":
        """
        Merge a list of Results objects into a single Results object.
        """
        doc_ids = np.concatenate([r.doc_ids for r in results], axis=0)
        documents = np.concatenate([r.documents for r in results], axis=0)
        scores = np.concatenate([r.scores for r in results], axis=0)
        return cls(doc_ids=doc_ids, documents=documents, scores=scores)


def is_list_of_type(obj, type_=str):
    if not isinstance(obj, list):
        return False

    if len(obj) == 0:
        return False

    first_elem = obj[0]
    if not isinstance(first_elem, type_):
        return False

    return True


class SPLADE:
    def __init__(
        self,
        dtype="float32",
        int_dtype="int32",
        backend: Literal["auto", "numpy", "numba"] = "numpy",
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
            The backend used during retrieval. By default, it is set to "numpy". You can also select `backend="numba"`
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
        batch_size: int = 32,
        chunk_size: int = 128,
        show_progress=True,
        leave_progress=False,
        compile_numba_code=False,
    ):
        """
        Given a `corpus` of documents, create the SPLADE index.

        model: SparseEncoder
            A sentence-transformers SPLADE model

        documents: List[str]
            A list of documents to be indexed

        batch_size: int
            The batch size used for the computation. Defaults to 32.

        chunk_size: int
            The chunk size used for the computation. Defaults to 128.

        show_progress : bool
            If True, a progress bar will be shown. If False, no progress bar will be shown.

        leave_progress : bool
            If True, the progress bars will remain after the function completes.

        compile_numba_code : bool
            If True, and if the backend is set to `numba`, this will initiate jit-compilation after
            indexing is complete, so that the first query will have not be affected by latency overhead
            caused by the compilation of numba code.

        """
        import scipy.sparse as sp

        document_embeddings = model.encode_document(
            documents,
            batch_size=batch_size,
            chunk_size=chunk_size,
            save_to_cpu=True,
            show_progress_bar=show_progress,
        ).coalesce()

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
        self.corpus = np.array(documents)

        # we create unique token IDs from the vocab_dict for faster lookup
        self.unique_token_ids_set = set(token_ids)

        if self.backend == "numba" and compile_numba_code:
            # to initiate jit-compilation
            _ = self.retrieve(
                ["dummy query"],
                k=1,
                batch_size=8,
                return_as="doc_ids",
                show_progress=False,
            )

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
        batch_size: int = 32,
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
        queries : List[str]
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
            If `return_as="doc_ids"`, only the ids of the retrieved documents will be returned.
            If return_as="documents", only the retrieved documents will be returned.

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
            The backend to use for the top-k retrieval. Choose from "auto", "numpy", "pytorch".
            If "auto", it will use PyTorch if it is available, otherwise it will use numpy.

        weight_mask : np.ndarray
            A weight mask to filter the documents. If provided, the scores for the masked
            documents will be set to 0 to avoid returning them in the results.

        Returns
        -------
        Results or np.ndarray
            If `return_as="tuple"`, a named tuple with two fields will be returned: `documents` and `scores`.
            If `return_as="doc_ids"`, only the ids of the retrieved documents will be returned.
            If `return_as="documents"`, only the retrieved documents will be returned.

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
        allowed_return_as = ["tuple", "doc_ids", "documents"]

        if return_as not in allowed_return_as:
            raise ValueError(
                "`return_as` must be either 'tuple', 'doc_ids' or 'documents'"
            )
        else:
            pass

        if n_threads == -1:
            n_threads = os.cpu_count()

        if self.backend == "numba" and backend_selection not in ("numba", "auto"):
            warning_msg = (
                "backend is set to `numba`, but backend_selection is neither `numba` or `auto`. "
                "In order to retrieve using the numba backend, please change the backend_selection parameter to `numba`."
            )
            warnings.warn(warning_msg, UserWarning)

        backend_selection = (
            "numba"
            if backend_selection == "auto" and self.backend == "numba"
            else backend_selection
        )

        # Embed Queries
        query_embeddings = self.model.encode_query(
            queries,
            batch_size=batch_size,
            convert_to_tensor=False,
            save_to_cpu=True,
            show_progress_bar=show_progress,
        )

        query_token_ids = [
            query_emb.coalesce().indices()[0].numpy() for query_emb in query_embeddings
        ]
        query_token_weights = [
            query_emb.coalesce().values().numpy() for query_emb in query_embeddings
        ]

        if backend_selection == "numba":
            if _retrieve_numba_functional is None:
                raise ImportError(
                    "Numba is not installed. Please install numba wiith `pip install numba` to use the numba backend."
                )

            res = _retrieve_numba_functional(
                query_tokens_ids=query_token_ids,
                query_tokens_weights=query_token_weights,
                scores=self.scores,
                k=k,
                sorted=sorted,
                return_as=return_as,
                show_progress=show_progress,
                leave_progress=leave_progress,
                n_threads=n_threads,
                chunksize=None,  # chunksize is ignored in the numba backend
                backend_selection=backend_selection,  # backend_selection is ignored in the numba backend
                dtype=self.dtype,
                int_dtype=self.int_dtype,
            )

            if return_as == "tuple":
                return Results(
                    doc_ids=res[0], documents=self.corpus[res[0]], scores=res[1]
                )
            elif return_as == "documents":
                return self.corpus[res]
            else:
                return res

        tqdm_kwargs = {
            "total": len(query_token_ids),
            "desc": "SPLADE Index Retrieve",
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
            return Results(
                doc_ids=retrieved_docs,
                documents=self.corpus[retrieved_docs],
                scores=scores,
            )
        elif return_as == "documents":
            return self.corpus[retrieved_docs]
        elif return_as == "doc_ids":
            return retrieved_docs
        else:
            raise ValueError(
                "`return_as` must be either 'tuple', 'doc_ids' or 'documents'"
            )

    def save(
        self,
        save_dir,
        data_name="data.csc.index.npy",
        indices_name="indices.csc.index.npy",
        indptr_name="indptr.csc.index.npy",
        vocab_name="vocab.index.json",
        params_name="params.index.json",
        corpus_name="corpus.npy",
        allow_pickle=False,
    ):
        """
        Save the BM25S index to the `save_dir` directory. This will save the scores array,
        the indices array, the indptr array, the vocab dictionary, and the parameters.

        Parameters
        ----------
        save_dir : str
            The directory where the BM25S index will be saved.

        corpus_name : str
            The name of the file that will contain the corpus.

        data_name : str
            The name of the file that will contain the data array.

        indices_name : str
            The name of the file that will contain the indices array.

        indptr_name : str
            The name of the file that will contain the indptr array.

        vocab_name : str
            The name of the file that will contain the vocab dictionary.

        params_name : str
            The name of the file that will contain the parameters.

        allow_pickle : bool
            If True, the arrays will be saved using pickle. If False, the arrays will be saved
            in a more efficient format, but they will not be readable by older versions of numpy.
        """
        # Save the self.vocab_dict and self.score_matrix to the save_dir
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save the scores arrays
        data_path = save_dir / data_name
        indices_path = save_dir / indices_name
        indptr_path = save_dir / indptr_name

        np.save(data_path, self.scores["data"], allow_pickle=allow_pickle)
        np.save(indices_path, self.scores["indices"], allow_pickle=allow_pickle)
        np.save(indptr_path, self.scores["indptr"], allow_pickle=allow_pickle)

        # Save the vocab dictionary
        vocab_path = save_dir / vocab_name

        with open(vocab_path, "wt", encoding="utf-8") as f:
            f.write(json_functions.dumps(self.vocab_dict, ensure_ascii=False))

        # Save the parameters
        params_path = save_dir / params_name
        params = dict(
            dtype=self.dtype,
            int_dtype=self.int_dtype,
            num_docs=self.scores["num_docs"],
            version=__version__,
            backend=self.backend,
        )
        with open(params_path, "w") as f:
            json.dump(params, f, indent=4)

        corpus = self.corpus
        corpus_path = save_dir / corpus_name

        if corpus is not None:
            np.save(corpus_path, corpus, allow_pickle=allow_pickle)

    def load_scores(
        self,
        save_dir,
        data_name="data.csc.index.npy",
        indices_name="indices.csc.index.npy",
        indptr_name="indptr.csc.index.npy",
        num_docs=None,
        mmap=False,
        allow_pickle=False,
    ):
        """
        Load the scores arrays from the BM25 index. This is useful if you want to load
        the scores arrays separately from the vocab dictionary and the parameters.

        This is called internally by the `load` method, so you do not need to call it directly.

        Parameters
        ----------
        data_name : str
            The name of the file that contains the data array.

        indices_name : str
            The name of the file that contains the indices array.

        indptr_name : str
            The name of the file that contains the indptr array.

        mmap : bool
            Whether to use Memory-map for the np.load function. If false, the arrays will be loaded into memory.
            If True, the arrays will be memory-mapped, using 'r' mode. This is useful for very large arrays that
            do not fit into memory.

        allow_pickle : bool
            If True, the arrays will be loaded using pickle. If False, the arrays will be loaded
            in a more efficient format, but they will not be readable by older versions of numpy.
        """
        save_dir = Path(save_dir)

        data_path = save_dir / data_name
        indices_path = save_dir / indices_name
        indptr_path = save_dir / indptr_name

        mmap_mode = "r" if mmap else None
        data = np.load(data_path, allow_pickle=allow_pickle, mmap_mode=mmap_mode)
        indices = np.load(indices_path, allow_pickle=allow_pickle, mmap_mode=mmap_mode)
        indptr = np.load(indptr_path, allow_pickle=allow_pickle, mmap_mode=mmap_mode)

        scores = {}
        scores["data"] = data
        scores["indices"] = indices
        scores["indptr"] = indptr
        scores["num_docs"] = num_docs

        self.scores = scores

    @classmethod
    def load(
        cls,
        save_dir,
        model,
        data_name="data.csc.index.npy",
        indices_name="indices.csc.index.npy",
        indptr_name="indptr.csc.index.npy",
        vocab_name="vocab.index.json",
        params_name="params.index.json",
        corpus_name="corpus.npy",
        load_corpus=True,
        mmap=False,
        allow_pickle=False,
        load_vocab=True,
    ):
        """
        Load a BM25S index that was saved using the `save` method.
        This returns a BM25S object with the saved parameters and scores,
        which can be directly used for retrieval.

        Parameters
        ----------
        save_dir : str
            The directory where the BM25S index was saved.

        model: SparseEncoder
            A sentence-transformers SPLADE model (The same one used to create the index)

        data_name : str
            The name of the file that contains the data array.

        indices_name : str
            The name of the file that contains the indices array.

        indptr_name : str
            The name of the file that contains the indptr array.

        vocab_name : str
            The name of the file that contains the vocab dictionary.

        params_name : str
            The name of the file that contains the parameters.

        corpus_name : str
            The name of the file that contains the corpus.

        load_corpus : bool
            If True, the corpus will be loaded from the `corpus_name` file.

        mmap : bool
            Whether to use Memory-map for the np.load function. If false, the arrays will be loaded into memory.
            If True, the arrays will be memory-mapped, using 'r' mode. This is useful for very large arrays that
            do not fit into memory.

        allow_pickle : bool
            If True, the arrays will be loaded using pickle. If False, the arrays will be loaded
            in a more efficient format, but they will not be readable by older versions of numpy.

        load_vocab : bool
            If True, the vocab dictionary will be loaded from the `vocab_name` file. If False, the vocab dictionary
            will not be loaded, and the `vocab_dict` attribute of the BM25 object will be set to None.
        """
        if not isinstance(mmap, bool):
            raise ValueError("`mmap` must be a boolean")

        # Load the SPLADE index from the save_dir
        save_dir = Path(save_dir)

        # Load the parameters
        params_path = save_dir / params_name
        with open(params_path, "r") as f:
            params: dict = json_functions.loads(f.read())

        # Load the vocab dictionary
        if load_vocab:
            vocab_path = save_dir / vocab_name
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab_dict: dict = json_functions.loads(f.read())
        else:
            vocab_dict = {}

        original_version = params.pop("version", None)
        num_docs = params.pop("num_docs", None)

        splade_obj = cls(**params)
        splade_obj.vocab_dict = vocab_dict
        splade_obj._original_version = original_version
        splade_obj.unique_token_ids_set = set(splade_obj.vocab_dict.values())

        splade_obj.model = model

        splade_obj.load_scores(
            save_dir=save_dir,
            data_name=data_name,
            indices_name=indices_name,
            indptr_name=indptr_name,
            mmap=mmap,
            num_docs=num_docs,
            allow_pickle=allow_pickle,
        )

        if load_corpus:
            # load the model from the snapshot
            # if a corpus.jsonl file exists, load it
            corpus_path = save_dir / corpus_name
            if os.path.exists(corpus_path):
                mmap_mode = "r" if mmap else None
                corpus = np.load(
                    corpus_path, allow_pickle=allow_pickle, mmap_mode=mmap_mode
                )
                splade_obj.corpus = corpus

        return splade_obj
