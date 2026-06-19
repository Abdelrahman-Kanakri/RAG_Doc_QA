"""Shared pytest fixtures and environment setup for the test suite.

Set the OpenSSL / HF-offline guards BEFORE any app import so that importing
modules that pull in pandas/pyarrow (via ragas) does not segfault on this
uv-managed CPython, and so no test ever reaches out to the network.
"""

import os

os.environ.setdefault("OPENSSL_CONF", "/dev/null")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import types
import pytest
from langchain_core.documents import Document


@pytest.fixture
def make_doc():
    """Factory for building langchain Documents with arbitrary metadata."""
    def _make(content: str = "some passage content", **metadata) -> Document:
        return Document(page_content=content, metadata=metadata)
    return _make


@pytest.fixture
def fake_llm_response():
    """Build a fake chat-model response object exposing `.content` (like an AIMessage)."""
    def _make(content: str):
        return types.SimpleNamespace(content=content)
    return _make


class FakeVectorStore:
    """Minimal stand-in for a Chroma vector store.

    `similarity_search_with_score` returns the canned (Document, distance) pairs;
    `similarity_search` returns just the Documents. Distance is the raw Chroma
    score (lower = more similar); the retrievers convert it via `1 - distance`.
    """

    def __init__(self, results=None):
        # results: list[tuple[Document, float]]
        self._results = results or []

    def similarity_search_with_score(self, query, k=5):
        return list(self._results)[:k]

    def similarity_search(self, query, k=5):
        return [doc for doc, _ in list(self._results)[:k]]


@pytest.fixture
def fake_vector_store():
    """Return the FakeVectorStore class so tests can instantiate with their own results."""
    return FakeVectorStore
