import os

README_TEMPLATE = """---
language: en
library_name: splade-index
tags:
- splade
- splade-index
- retrieval
- search
- sparse
---

# Splade-Index

This is an index created with the [`splade-index` library](https://github.com/rasyosef/splade-index) (version `{version}`)

## Installation

You can install the `bm25s` library with `pip`:

```bash
pip install "splade-index=={version}"

# Include extra dependencies like stemmer
pip install "splade-index[full]=={version}"

# For huggingface hub usage
pip install huggingface_hub
```

## Load this Index

You can use the following code to load this SPLADE index from Hugging Face hub:

```python
import os
from sentence_transformers import SparseEncoder
from splade_index import SPLADE

# Download the SPLADE model used to create the index, from the 🤗 Hub
model_id = "the-splade-model-id" # Enter the splade model id
model = SparseEncoder(model_id)

# Set your huggingface token if repo is private
token = os.environ["HF_TOKEN"]

repo_id = "{username}/{repo_name}"

# Load a SPLADE index from the Hugging Face model hub
retriever = SPLADE.load_from_hub(repo_id, model=model, token=token)
```

## Stats

This dataset was created using the following data:

| Statistic | Value |
| --- | --- |
| Number of documents | {num_docs} |
| Number of tokens | {num_tokens} |
| Average tokens per document | {avg_tokens_per_doc} |

"""


def is_dir_empty(local_save_dir):
    """
    Check if a directory is empty or not.

    Parameters
    ----------
    local_save_dir: str
        The directory to check.

    Returns
    -------
    bool
        True if the directory is empty, False otherwise.
    """
    if not os.path.exists(local_save_dir):
        return True
    return len(os.listdir(local_save_dir)) == 0


def can_save_locally(local_save_dir, overwrite_local: bool) -> bool:
    """
    Check if it is possible to save the model to a local directory.

    Parameters
    ----------
    local_save_dir: str
        The directory to save the model to.

    overwrite_local: bool
        Whether to overwrite the existing local directory if it exists.

    Returns
    -------
    bool
        True if it is possible to save the model to the local directory, False otherwise.
    """
    # if local_save_dir is None, we cannot save locally
    if local_save_dir is None:
        return False

    # if the directory is empty, we can save locally
    if is_dir_empty(local_save_dir):
        return True

    # if we are allowed to overwrite the directory, we can save locally
    if overwrite_local:
        return True
