import warnings

from concurrent.futures import ThreadPoolExecutor
from functools import partial

import os
import logging
from pathlib import Path
import json
from typing import List, NamedTuple, Literal, Union

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

import shutil
import tempfile

from .hf import README_TEMPLATE, can_save_locally


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

logger = logging.getLogger("splade_index")
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
            The data type of the similarity scores.

        int_dtype : str
            The data type of the indices in the scores.

        backend : str
            The backend used during retrieval. By default, it is set to "numpy". You can also select `backend="numba"`
            to use the numba backend, which requires the numba library. If you select `backend="auto"`,
            the function will use the numba backend if it is available, otherwise it will use the numpy
            backend.
        """
        self.dtype = dtype
        self.int_dtype = int_dtype
        self.corpus = None
        self.document_ids = None
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
        document_ids: Union[List[int], List[str]] = None,
        batch_size: int = 32,
        chunk_size: int = 128,
        show_progress=True,
        leave_progress=False,
        compile_numba_code=True,
    ):
        """
        Given a `corpus` of documents, create the SPLADE index.

        model: SparseEncoder
            A sentence-transformers SPLADE model

        documents: List[str]
            A list of documents to be indexed

        document_ids: Union[List[int], List[str]]
            Optional: The IDs of each or the documents to be indexed in the same order as `documents`. Defaults to None.
            If it's not set, integers 0 up to len(documents)-1 will be used as the document_ids.

        batch_size: int
            The batch size used for the computation. Defaults to 32.

        chunk_size: int
            The chunk size used for the computation. Defaults to 128.

        show_progress : bool
            If True, a progress bar will be shown. If False, no progress bar will be shown.

        leave_progress : bool
            If True, the progress bars will remain after the function completes.

        compile_numba_code : bool
            If True, and if the backend is set to `numba`, this will run a dummy query in order to initiate jit-compilation of
            the numba code after indexing is complete, so that the first query will not be affected by latency overhead
            due to compilation of numba code. It is set to True by default.

        """

        score_matrix = model.encode_document(
            documents,
            batch_size=batch_size,
            chunk_size=chunk_size,
            save_to_cpu=True,
            show_progress_bar=show_progress,
        ).to_sparse_csc()

        data = score_matrix.values().numpy().astype(dtype=self.dtype, copy=False)
        indices = score_matrix.row_indices().numpy().astype(dtype=self.int_dtype)
        indptr = score_matrix.ccol_indices().numpy().astype(dtype=self.int_dtype)

        vocab_dict = model.tokenizer.get_vocab()
        num_docs = len(documents)

        scores = {
            "data": data,
            "indices": indices,
            "indptr": indptr,
            "num_docs": num_docs,
        }

        self.scores = scores
        self.vocab_dict = vocab_dict
        self.model = model
        self.corpus = np.asarray(documents, dtype=object)
        if document_ids is not None:
            self.document_ids = np.asarray(document_ids)
        else:
            self.document_ids = np.arange(len(self.corpus))

        # we create unique token IDs set for faster lookup
        unique_token_ids = [
            i for i in range(len(indptr) - 1) if indptr[i] < indptr[i + 1]
        ]

        self.unique_token_ids_set = set(unique_token_ids)

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
        return_as: Literal["tuple", "documents", "doc_ids"] = "tuple",
        show_progress: bool = True,
        leave_progress: bool = False,
        n_threads: int = 0,
        chunksize: int = 50,
        backend_selection: Literal["auto", "numpy", "pytorch", "numba"] = "auto",
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
            If `return_as="documents"`, only the retrieved documents will be returned.

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
            The backend to use for the top-k retrieval. Choose from "auto", "numpy", "pytorch", "numba".
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
                    doc_ids=self.document_ids[res[0]],
                    documents=self.corpus[res[0]],
                    scores=res[1],
                )
            elif return_as == "documents":
                return self.corpus[res]
            elif return_as == "doc_ids":
                return self.document_ids[res]
            else:
                raise ValueError(
                    "`return_as` must be either 'tuple', 'doc_ids' or 'documents'"
                )

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

        retrieved_indices = indices

        if return_as == "tuple":
            return Results(
                doc_ids=self.document_ids[retrieved_indices],
                documents=self.corpus[retrieved_indices],
                scores=scores,
            )
        elif return_as == "documents":
            return self.corpus[retrieved_indices]
        elif return_as == "doc_ids":
            return retrieved_indices
        else:
            raise ValueError(
                "`return_as` must be either 'tuple', 'doc_ids' or 'documents'"
            )

    def save(
        self,
        save_dir,
        csc_index_name="csc.index.npz",
        corpus_name="corpus.jsonl",
        vocab_name="vocab.index.json",
        params_name="params.index.json",
    ):
        """
        Save the SPLADE index to the `save_dir` directory. This will save the scores array,
        the indices array, the indptr array, the vocab dictionary, and the parameters.

        Parameters
        ----------
        save_dir : str
            The directory where the SPLADE index will be saved.

        csc_index_name : str
            The name of the file that contains the csc index arrays (data, indices, indptr).

        corpus_name : str
            The name of the file that will contain the corpus.

        vocab_name : str
            The name of the file that will contain the vocab dictionary.

        params_name : str
            The name of the file that will contain the parameters.
        """
        # Save the self.vocab_dict and self.score_matrix to the save_dir
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save the scores arrays
        csc_index_path = save_dir / csc_index_name

        np.savez_compressed(
            csc_index_path,
            data=self.scores["data"],
            indices=self.scores["indices"],
            indptr=self.scores["indptr"],
            document_ids=self.document_ids,
        )

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
        document_ids = self.document_ids
        corpus_path = save_dir / corpus_name

        if corpus is not None:
            with open(corpus_path, "wt", encoding="utf-8") as f:
                for doc_id, doc in zip(document_ids, corpus):
                    doc = {"id": doc_id, "text": doc}
                    try:
                        doc_str = json_functions.dumps(doc, ensure_ascii=False)
                    except Exception as e:
                        logging.warning(
                            f"Error saving document with doc_id {doc_id}: {e}"
                        )
                    else:
                        f.write(doc_str + "\n")
            # also save corpus.mmindex
            mmidx = utils.corpus.find_newline_positions(save_dir / corpus_name)
            utils.corpus.save_mmindex(mmidx, path=save_dir / corpus_name)

    def load_scores(
        self,
        save_dir,
        csc_index_name="csc.index.npz",
        num_docs=None,
        mmap=False,
    ):
        """
        Load the scores arrays from the BM25 index. This is useful if you want to load
        the scores arrays separately from the vocab dictionary and the parameters.

        This is called internally by the `load` method, so you do not need to call it directly.

        Parameters
        ----------
        csc_index_name : str
            The name of the file that contains the csc index arrays (data, indices, indptr, document_ids).

        mmap : bool
            Whether to use Memory-map for the np.load function. If false, the arrays will be loaded into memory.
            If True, the arrays will be memory-mapped, using 'r' mode. This is useful for very large arrays that
            do not fit into memory.
        """
        save_dir = Path(save_dir)
        csc_index_path = save_dir / csc_index_name
        mmap_mode = "r" if mmap else None
        csc_index = None

        if mmap:
            from zipfile import ZipFile

            temp_dir = save_dir / "temp"

            with ZipFile(csc_index_path, "r") as f:
                f.extract("data.npy", temp_dir)
                f.extract("indices.npy", temp_dir)
                f.extract("indptr.npy", temp_dir)
                f.extract("document_ids.npy", temp_dir)

            csc_index = {
                "data": np.load(temp_dir / "data.npy", mmap_mode=mmap_mode),
                "indices": np.load(temp_dir / "indices.npy", mmap_mode=mmap_mode),
                "indptr": np.load(temp_dir / "indptr.npy", mmap_mode=mmap_mode),
                "document_ids": np.load(
                    temp_dir / "document_ids.npy", mmap_mode=mmap_mode
                ),
            }
        else:
            csc_index = np.load(csc_index_path)

        scores = {}
        scores["data"] = csc_index["data"]
        scores["indices"] = csc_index["indices"]
        scores["indptr"] = csc_index["indptr"]
        scores["num_docs"] = num_docs

        unique_token_ids = [
            i
            for i in range(len(scores["indptr"]) - 1)
            if scores["indptr"][i] < scores["indptr"][i + 1]
        ]

        self.unique_token_ids_set = set(unique_token_ids)
        self.scores = scores
        self.document_ids = csc_index["document_ids"]

    @classmethod
    def load(
        cls,
        save_dir,
        model,
        csc_index_name="csc.index.npz",
        vocab_name="vocab.index.json",
        params_name="params.index.json",
        corpus_name="corpus.jsonl",
        load_corpus=True,
        mmap=False,
        load_vocab=True,
    ):
        """
        Load a SPLADE index that was saved using the `save` method.
        This returns a SPLADE object with the saved parameters and scores,
        which can be directly used for retrieval.

        Parameters
        ----------
        save_dir : str
            The directory where the BM25S index was saved.

        model: SparseEncoder
            A sentence-transformers SPLADE model (The same one used to create the index)

        csc_index_name : str
            The name of the file that contains the csc index arrays (data, indices, indptr, document_ids).

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

        load_vocab : bool
            If True, the vocab dictionary will be loaded from the `vocab_name` file. If False, the vocab dictionary
            will not be loaded, and the `vocab_dict` attribute of the SPLADE object will be set to None.
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

        splade_obj.model = model

        splade_obj.load_scores(
            save_dir=save_dir,
            csc_index_name=csc_index_name,
            mmap=mmap,
            num_docs=num_docs,
        )

        if load_corpus:
            # load the model from the snapshot
            # if a corpus.json file exists, load it
            corpus_path = save_dir / corpus_name
            if os.path.exists(corpus_path):
                # mmap_mode = "r" if mmap else None
                if mmap:
                    corpus = utils.corpus.JsonlCorpus(corpus_path)
                    splade_obj.corpus = corpus
                else:
                    corpus = []
                    with open(corpus_path, "r", encoding="utf-8") as f:
                        for line in f:
                            doc = json_functions.loads(line)
                            corpus.append(doc["text"])

                    splade_obj.corpus = np.asarray(corpus, dtype=object)

        return splade_obj

    def save_to_hub(
        self,
        repo_id: str,
        token: str = None,
        local_dir: str = None,
        private=True,
        commit_message: str = "Update Splade Index",
        overwrite_local: bool = False,
        include_readme: bool = True,
        **kwargs,
    ):
        """
        This function saves the SPLADE index to the Hugging Face Hub.

        Parameters
        ----------

        repo_id: str
            The name of the repository to save the model to.
            the `repo_id` should be in the form of "username/repo_name".

        token: str
            The Hugging Face API token to use.

        local_dir: str
            The directory to save the model to before pushing to the Hub.
            If it is not empty and `overwrite_local` is False, it will fall
            back to saving to a temporary directory.

        private: bool
            Whether the repository should be private or not. Default is True.

        commit_message: str
            The commit message to use when saving the model.

        overwrite_local: bool
            Whether to overwrite the existing local directory if it exists.

        include_readme: bool
            Whether to include a default README file with the model.

        kwargs: dict
            Additional keyword arguments to pass to `HfApi.upload_folder` call.
        """

        try:
            from huggingface_hub import HfApi
        except ImportError:
            raise ImportError(
                "Please install the huggingface_hub package to use the HuggingFace integrations for splade-index. You can install it via `pip install huggingface_hub`."
            )

        api = HfApi(token=token)
        repo_url = api.create_repo(
            repo_id=repo_id,
            token=api.token,
            private=private,
            repo_type="model",
            exist_ok=True,
        )
        repo_id = repo_url.repo_id

        username, repo_name = repo_id.split("/", 1)

        saving_locally = can_save_locally(local_dir, overwrite_local)
        if saving_locally:
            os.makedirs(local_dir, exist_ok=True)
            save_dir = local_dir
        else:
            # save to a temporary directory otherwise
            save_dir = tempfile.mkdtemp()

        self.save(save_dir)
        # if we include the README, write it to the directory
        if include_readme:
            num_docs = self.scores["num_docs"]
            num_tokens = self.scores["data"].shape[0]
            avg_tokens_per_doc = round(num_tokens / num_docs, 2)

            results = README_TEMPLATE.format(
                username=username,
                repo_name=repo_name,
                version=__version__,
                num_docs=num_docs,
                num_tokens=num_tokens,
                avg_tokens_per_doc=avg_tokens_per_doc,
            )

            with open(os.path.join(save_dir, "README.md"), "w") as f:
                f.write(results)

        # push content of the temporary directory to the repo
        api.upload_folder(
            repo_id=repo_id,
            commit_message=commit_message,
            token=api.token,
            folder_path=save_dir,
            repo_type=repo_url.repo_type,
            **kwargs,
        )
        # delete the temporary directory if it was created
        if not saving_locally:
            shutil.rmtree(save_dir)

        return repo_url

    @classmethod
    def load_from_hub(
        cls,
        repo_name: str,
        model,
        revision=None,
        token=None,
        local_dir=None,
        load_corpus=True,
        mmap=False,
    ):
        """
        This function loads the BM25 model from the Hugging Face Hub.

        Parameters
        ----------

        repo_name: str
            The name of the repository to load the model from.

        model: SparseEncoder
            A sentence-transformers SPLADE model (The same one used to create the index)

        revision: str
            The revision of the model to load.

        token: str
            The Hugging Face API token to use.

        local_dir: str
            The local dir where the model will be stored after downloading.

        load_corpus: bool
            Whether to load the corpus of documents saved with the model, if present.

        mmap: bool
            Whether to memory-map the model. Default is False, which loads the index
            (and potentially corpus) into memory.
        """

        try:
            from huggingface_hub import HfApi
        except ImportError:
            raise ImportError(
                "Please install the huggingface_hub package to use the HuggingFace integrations for splade-index. You can install it via `pip install huggingface_hub`."
            )

        api = HfApi(token=token)
        # check if the model exists
        repo_url = api.repo_info(repo_name)
        if repo_url is None:
            raise ValueError(f"Model {repo_name} not found on the Hugging Face Hub.")

        snapshot = api.snapshot_download(
            repo_name, revision=revision, token=token, local_dir=local_dir
        )
        if snapshot is None:
            raise ValueError(f"Model {repo_name} not found on the Hugging Face Hub.")

        return cls.load(
            save_dir=snapshot,
            model=model,
            load_corpus=load_corpus,
            mmap=mmap,
        )
