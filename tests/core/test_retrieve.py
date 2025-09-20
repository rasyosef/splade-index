import unittest
import numpy as np
import splade_index
from sentence_transformers import SparseEncoder


class TestSpladeIndexRetrieval(unittest.TestCase):
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
        cls.corpus = corpus
        cls.document_ids = document_ids

    def test_retrieve(self):
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
        default_backend = "numpy"

        queries = np.array(["does the fish purr like a cat?"])

        # Get top-k results as a tuple of (doc ids, documents, scores). All three are arrays of shape (n_queries, k).
        results = self.retriever.retrieve(queries, k=2)
        doc_ids, result_docs, scores = (
            results.doc_ids,
            results.documents,
            results.scores,
        )

        scores = np.round(scores, decimals=2)

        self.assertEqual(
            self.retriever.backend,
            default_backend,
            f"Expected {default_backend}, got {self.retriever.backend}",
        )

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

        # return_as = "doc_ids"
        document_ids = self.retriever.retrieve(queries, k=2, return_as="doc_ids")
        self.assertTrue(
            np.array_equal(doc_ids, ground_truth_ids),
            f"Expected {ground_truth_ids}, got {document_ids}",
        )

        # return_as = "documents"
        documents = self.retriever.retrieve(queries, k=2, return_as="documents")
        self.assertTrue(
            np.array_equal(result_docs, ground_truth_docs),
            f"Expected {ground_truth_docs}, got {documents}",
        )

    def test_retrieve_numpy_backend(self):
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
        results = self.retriever.retrieve(queries, k=2, backend_selection="numpy")
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

        # return_as = "doc_ids"
        document_ids = self.retriever.retrieve(queries, k=2, return_as="doc_ids")
        self.assertTrue(
            np.array_equal(doc_ids, ground_truth_ids),
            f"Expected {ground_truth_ids}, got {document_ids}",
        )

        # return_as = "documents"
        documents = self.retriever.retrieve(queries, k=2, return_as="documents")
        self.assertTrue(
            np.array_equal(result_docs, ground_truth_docs),
            f"Expected {ground_truth_docs}, got {documents}",
        )

    def test_retrieve_pytorch_backend(self):
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
        results = self.retriever.retrieve(queries, k=2, backend_selection="pytorch")
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

        # return_as = "doc_ids"
        document_ids = self.retriever.retrieve(queries, k=2, return_as="doc_ids")
        self.assertTrue(
            np.array_equal(doc_ids, ground_truth_ids),
            f"Expected {ground_truth_ids}, got {document_ids}",
        )

        # return_as = "documents"
        documents = self.retriever.retrieve(queries, k=2, return_as="documents")
        self.assertTrue(
            np.array_equal(result_docs, ground_truth_docs),
            f"Expected {ground_truth_docs}, got {documents}",
        )
