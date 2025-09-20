import os
import unittest
import tempfile
import shutil
import numpy as np
import splade_index
from sentence_transformers import SparseEncoder


class TestSpladeIndexSaveLoad(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        # Download SPLADE from the 🤗 Hub
        model = SparseEncoder("rasyosef/splade-tiny")

        # Create your corpus here
        corpus = np.array(
            [
                "a cat is a feline and likes to purr",
                "a dog is the human's best friend and loves to play",
                "a bird is a beautiful animal that can fly",
                "a fish is a creature that lives in water and swims",
            ]
        )

        document_ids = ["zero", "one", "two", "three"]

        # Create the SPLADE retriever and index the corpus
        retriever = splade_index.SPLADE()  # backend="numba"
        retriever.index(model=model, documents=corpus, document_ids=document_ids)

        # Save the retriever to temp dir
        cls.retriever = retriever
        cls.model = model
        cls.corpus = corpus
        cls.document_ids = document_ids
        cls.temp_save_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        # remove the temp dir with rmtree
        shutil.rmtree(cls.temp_save_dir)

    def test_save(self):
        self.retriever.save(self.temp_save_dir)

        filenames = [
            "csc.index.npz",
            "corpus.jsonl",
            "corpus.mmindex.json",
            "params.index.json",
            "vocab.index.json",
        ]

        for fname in filenames:
            error_msg = f"File {fname} not found in even though it should be saved by the .save() method"
            path_exists = os.path.exists(os.path.join(self.temp_save_dir, fname))
            self.assertTrue(path_exists, error_msg)

    def test_save_and_reload(self):
        self.retriever.save(self.temp_save_dir)
        self.retriever_reloaded = splade_index.SPLADE.load(
            self.temp_save_dir, self.model
        )

        ground_truth_ids = np.array([["zero", "three"]])
        ground_truth_scores = np.array([[15.67, 8.17]], dtype="float32")
        ground_truth_docs = np.array(
            [
                [
                    "a cat is a feline and likes to purr",
                    "a fish is a creature that lives in water and swims",
                ]
            ]
        )

        queries = np.array(["does the fish purr like a cat?"])

        # Get top-k results as a tuple of (doc ids, documents, scores). All three are arrays of shape (n_queries, k).
        results = self.retriever_reloaded.retrieve(queries, k=2)
        doc_ids, result_docs, scores = (
            results.doc_ids,
            results.documents,
            results.scores,
        )

        scores = np.round(scores, decimals=2)

        self.assertTrue(
            np.array_equal(doc_ids, ground_truth_ids),
            f"Expected {ground_truth_ids}, got {doc_ids}",
        )

        self.assertTrue(
            np.array_equal(scores, ground_truth_scores),
            f"Expected {ground_truth_scores}, got {scores}",
        )

        self.assertTrue(
            np.array_equal(result_docs, ground_truth_docs),
            f"Expected {ground_truth_docs}, got {result_docs}",
        )

    def test_save_and_reload_mmap(self):
        self.retriever.save(self.temp_save_dir)
        self.retriever_reloaded = splade_index.SPLADE.load(
            self.temp_save_dir, self.model, mmap=True
        )

        ground_truth_ids = np.array([["zero", "three"]])
        ground_truth_scores = np.array([[15.67, 8.17]], dtype="float32")
        ground_truth_docs = np.array(
            [
                [
                    "a cat is a feline and likes to purr",
                    "a fish is a creature that lives in water and swims",
                ]
            ]
        )

        queries = np.array(["does the fish purr like a cat?"])

        # Get top-k results as a tuple of (doc ids, documents, scores). All three are arrays of shape (n_queries, k).
        results = self.retriever_reloaded.retrieve(queries, k=2)
        doc_ids, result_docs, scores = (
            results.doc_ids,
            results.documents,
            results.scores,
        )

        scores = np.round(scores, decimals=2)

        self.assertTrue(
            np.array_equal(doc_ids, ground_truth_ids),
            f"Expected {ground_truth_ids}, got {doc_ids}",
        )

        self.assertTrue(
            np.array_equal(scores, ground_truth_scores),
            f"Expected {ground_truth_scores}, got {scores}",
        )

        self.assertTrue(
            np.array_equal(result_docs, ground_truth_docs),
            f"Expected {ground_truth_docs}, got {result_docs}",
        )
