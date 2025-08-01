# SPLADE-Index⚡

<i>
SPLADE-Index is an ultrafast index for SPLADE sparse retrieval models implemented in pure Python and powered by Scipy sparse matrices. It is built on top of the BM25s library.
</i>
<br/><br/>

SPLADE is a neural retrieval model which learns query/document sparse expansion. Sparse representations benefit from several advantages compared to dense approaches: efficient use of inverted index, explicit lexical match, interpretability... They also seem to be better at generalizing on out-of-domain data (BEIR benchmark).

For more information about SPLADE models, please refer to the following. 
 - [SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking](https://arxiv.org/abs/2107.05720)
 - [List of Pretrained Sparse Encoder (Sparse Embeddings) Models](https://sbert.net/docs/sparse_encoder/pretrained_models.html)
 - [Training and Finetuning Sparse Embedding Models with Sentence Transformers v5](https://huggingface.co/blog/train-sparse-encoder).

## Installation

You can install `splade-index` with pip:

```bash
pip install splade-index
```

Recommended (but optional) dependencies:

```bash
# To speed up the top-k selection process, you can install `jax`
pip install "jax[cpu]"
```

## Quickstart

Here is a simple example of how to use `splade-index`:

```python
from sentence_transformers import SparseEncoder
from splade_index import SPLADE

# Download a SPLADE model from the 🤗 Hub
model = SparseEncoder("rasyosef/splade-tiny")

# Create your corpus here
corpus = [
    "a cat is a feline and likes to purr",
    "a dog is the human's best friend and loves to play",
    "a bird is a beautiful animal that can fly",
    "a fish is a creature that lives in water and swims",
]

# Create the SPLADE retriever and index the corpus
retriever = SPLADE()
retriever.index(model=model, documents=corpus)

# Query the corpus
queries = ["does the fish purr like a cat?"]

# Get top-k results as a tuple of (doc ids, documents, scores). All three are arrays of shape (n_queries, k).
results = retriever.retrieve(queries, k=2)
doc_ids, result_docs, scores = results.doc_ids, results.documents, results.scores

for i in range(doc_ids.shape[1]):
    doc_id, doc, score = doc_ids[0, i], result_docs[0, i], scores[0, i]
    print(f"Rank {i+1} (score: {score:.2f}) (doc_id: {doc_id}): {doc}")

# You can save the index to a directory
retriever.save("animal_index_splade")

# ...and load it when you need it
import splade_index

reloaded_retriever = splade_index.SPLADE.load("animal_index_splade", model=model)
```

## Performance

`splade-index` with a `numba` backend gives `61%` faster query time on average than the [pyseismic-lsr](https://github.com/TusKANNy/seismic) library, which is "an Efficient Inverted Index for Approximate Retrieval", all while `splade-index` does exact retrieval with no approximations involved. The query latency values shown include the query encoding times using the `rasyosef/splade-mini` SPLADE sparse encoder model.  

|Library|Latency per query (in miliseconds)|
|:-|:-|
|`splade-index` (with `numba` backend)|**1.56 ms**|
|`splade-index` (with `jax` backend)|2.53 ms|
|`splade-index` (with `numpy` backend)|2.48 ms|
|`pyseismic-lsr`|2.51 ms|

The tests were conducted using **`100,231`** documents and **`5,000`** queries from the [sentence-transformers/natural-questions](https://huggingface.co/datasets/sentence-transformers/natural-questions) dataset, and an NVIDIA Tesla T4 16GB GPU on Google Colab. 

## Acknowledgement
- `splade-index` was built on top of the [bm25s](https://github.com/xhluca/bm25s) library, and makes use of it's excellent inverted index impementation, originally used by `bm25s` for many variants of the BM25 ranking algorithm. 


<!-- ## Citation

You can refer to the library with this BibTeX:

```bibtex
@misc{PyLate,
  title={SPLADE-Index: A fast Search Index for SPLADE Sparse Retrieval Models},
  author={Yosef Worku Alemneh},
  url={https://github.com/rasyosef/splade-index},
  year={2025}
}
``` -->