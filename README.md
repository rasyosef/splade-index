# SPLADE-Index⚡

<i>
SPLADE-Index is an ultrafast search index for SPLADE sparse retrieval models implemented in pure Python. It is built on top of the BM25s library.
</i>
<br/><br/>

You can use `splade-index` to 

✅ Index and Query up to millions of documents using any SPLADE Sparse Embedding (SparseEncoder) model supported by `sentence-transformers` such as `naver/spalde-v3`.

✅ Save your index locally and load your index from the save files.

✅ Upload your index to huggingface hub and let anyone else download and use it.

✅ Use memory mapping to load large indices with minimal RAM usage and no noticeable change in search latency (Loading a 1 Million document index with mmap uses just 2GB of RAM).

✅ Make use of NVIDIA GPUs and PyTorch for 10x faster search compared to `splade-index`'s CPU based `numba` backend, when your index contains 1 million plus documents.

## SPLADE

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


## Hugging Face Integration

`splade-index` can naturally work with Hugging Face's `huggingface_hub`, allowing you to load and save your index to the model hub.

First, make sure you have a valid [access token for the Hugging Face model hub](https://huggingface.co/settings/tokens). This is needed to save models to the hub, or to load private models. Once you created it, you can add it to your environment variables:

```bash
export HF_TOKEN="hf_..."
```

Now, let's install the `huggingface_hub` library:

```bash
pip install huggingface_hub
```

Let's see how to use `SPLADE.save_to_hub` to save a SPLADE index to the Hugging Face model hub:

```python
import os
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

# Set your username and token
user = "your-username"
token = os.environ["HF_TOKEN"]
repo_id = f"{user}/splade-index-animals"

# Save the index on your huggingface account
retriever.save_to_hub(repo_id, token=token)
# You can also save it publicly with private=False
```

Then, you can use the following code to load a SPLADE index from the Hugging Face model hub:

```python
import os
from sentence_transformers import SparseEncoder
from splade_index import SPLADE

# Download a SPLADE model from the 🤗 Hub
model = SparseEncoder("rasyosef/splade-tiny")

# Set your huggingface username and token
user = "your-username"
token = os.environ["HF_TOKEN"]
repo_id = f"{user}/splade-index-animals"

# Load a SPLADE index from the Hugging Face model hub
retriever = SPLADE.load_from_hub(repo_id, model=model, token=token)

# Query the corpus
queries = ["does the fish purr like a cat?"]

# Get top-k results as a tuple of (doc ids, documents, scores). All three are arrays of shape (n_queries, k).
results = retriever.retrieve(queries, k=2)
doc_ids, result_docs, scores = results.doc_ids, results.documents, results.scores

for i in range(doc_ids.shape[1]):
    doc_id, doc, score = doc_ids[0, i], result_docs[0, i], scores[0, i]
    print(f"Rank {i+1} (score: {score:.2f}) (doc_id: {doc_id}): {doc}")
```

## 10x faster search with SPLADE_GPU

For large indices with 1 million plus documents, you can use `SPLADE_GPU` for 10x higher search throughput (queries/second) relative to `splade-index`'s already fast CPU based `numba` backend. In order to use `SPLADE_GPU`, you need to have an NVIDIA GPU and a pytorch installation with CUDA.

```python
from sentence_transformers import SparseEncoder
from splade_index.pytorch import SPLADE_GPU

# Download a SPLADE model from the 🤗 Hub
model = SparseEncoder("rasyosef/splade-mini", device="cuda")

# Load a SPLADE index from the Hugging Face model hub
repo_id = "rasyosef/msmarco_dev_1M_splade_index"
retriever = SPLADE_GPU.load_from_hub(
    repo_id, 
    model=model, 
    mmap=True, # memory mapping enabled for low RAM usage
    device="cuda"
)

# Query the corpus
queries = ["what is a corporation?", "do owls eat in the day", "average pharmacy tech salary"]

# Get top-k results as a tuple of (doc ids, documents, scores). All three are arrays of shape (n_queries, k).
results = retriever.retrieve(queries, k=5)
doc_ids, result_docs, scores = results.doc_ids, results.documents, results.scores

for i in range(doc_ids.shape[1]):
    doc_id, doc, score = doc_ids[0, i], result_docs[0, i], scores[0, i]
    print(f"Rank {i+1} (score: {score:.2f}) (doc_id: {doc_id}): {doc}")
```

## Performance

`splade-index` with a `numba` backend gives `45%` faster query time on average than the [pyseismic-lsr](https://github.com/TusKANNy/seismic) library, which is "an Efficient Inverted Index for Approximate Retrieval", all while `splade-index` does exact retrieval with no approximations involved. 

The query latency values shown include the query encoding times using the `naver/splade-v3-distilbert` SPLADE sparse encoder model.  

|Library|Latency per query (in miliseconds)|
|:-|:-|
|`splade-index` (with `numba` backend)|**1.77 ms**|
|`splade-index` (with `numpy` backend)|2.44 ms|
|`splade-index` (with `pytorch` backend)|2.61 ms|
|`pyseismic-lsr`|3.24 ms|

The tests were conducted using **`100,231`** documents and **`5,000`** queries from the [sentence-transformers/natural-questions](https://huggingface.co/datasets/sentence-transformers/natural-questions) dataset, and an NVIDIA Tesla T4 16GB GPU on Google Colab. 

## Examples

- [`splade_index_usage_example.ipynb`](examples/splade_index_usage_example.ipynb) to index and query `1,000` documents on a cpu.

- [`indexing_and_querying_100k_docs_with_gpu.ipynb`](examples/indexing_and_querying_100k_docs_with_gpu.ipynb) to index and query a `100,000` documents on a gpu.

### SPLADE Models

You can use SPLADE-Index with any splade model from huggingface hub such as the ones below.

||Size (# Params)|MSMARCO MRR@10|BEIR-13 avg nDCG@10|
|:---|:----|:-------------------|:------------------|
|[naver/splade-v3](https://huggingface.co/naver/splade-v3)|110M|40.2|51.7|
|[naver/splade-v3-distilbert](https://huggingface.co/naver/splade-v3-distilbert)|67.0M|38.7|50.0|
|[rasyosef/splade-small](https://huggingface.co/rasyosef/splade-small)|28.8M|35.4|46.6|
|[rasyosef/splade-mini](https://huggingface.co/rasyosef/splade-mini)|11.2M|34.1|44.5|
|[rasyosef/splade-tiny](https://huggingface.co/rasyosef/splade-tiny)|4.4M|30.9|40.6|

## Acknowledgement
`splade-index` was built on top of the [bm25s](https://github.com/xhluca/bm25s) library, and makes use of its excellent inverted index impementation, originally used by `bm25s` for its many variants of the BM25 ranking algorithm. 

<!-- ## Citation

You can refer to the library with this BibTeX:

```bibtex
@misc{SPLADE-Index,
  title={SPLADE-Index: A Fast Inverted Search Index for SPLADE Sparse Retrieval Models},
  author={Yosef Worku Alemneh},
  url={https://github.com/rasyosef/splade-index},
  year={2025}
} 
``` -->