# SPLADE-Index⚡

<i>SPLADE-Index is an ultrafast index for SPLADE sparse retrieval models implemented in pure Python and powered by Scipy sparse matrices</i>

## Installation

You can install `splade-index` with pip:

```bash
git clone https://github.com/rasyosef/splade-index.git
pip install ./splade-index
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

# Download SPLADE from the 🤗 Hub
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
doc_ids, result_docs, scores = retriever.retrieve(queries, k=2)

for i in range(doc_ids.shape[1]):
    doc_id, doc, score = doc_ids[0, i], result_docs[0, i], scores[0, i]
    print(f"Rank {i+1} (score: {score:.2f}) (doc_id: {doc_id}): {doc}")
```
