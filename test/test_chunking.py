"""Unit tests for app.ingestion.chunking.split_documents.

Pure logic — no external services. Verifies chunk creation, the enrichment
metadata (chunk_id / chunk_index / document_id / keywords / chunk_headers),
ID determinism + uniqueness, the empty-input guard, and chunk sizing.
"""

import pytest
from langchain_core.documents import Document

from app.ingestion.chunking import split_documents

REQUIRED_METADATA_KEYS = {"chunk_id", "chunk_index", "document_id", "keywords", "chunk_headers"}


def _long_doc(source: str = "/data/docs/lec1.pdf") -> Document:
    # ~1500 chars of space-separated tokens so the recursive splitter makes several chunks.
    return Document(page_content=("neuron " * 300).strip(), metadata={"source": source})


def test_splits_into_multiple_chunks():
    chunks = split_documents([_long_doc()], chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(isinstance(c, Document) for c in chunks)


def test_each_chunk_has_required_metadata():
    chunks = split_documents([_long_doc()], chunk_size=200, chunk_overlap=20)
    for c in chunks:
        assert REQUIRED_METADATA_KEYS <= set(c.metadata), f"missing keys in {c.metadata}"


def test_document_id_is_source_basename():
    chunks = split_documents([_long_doc(source="/some/nested/path/lecture 9+10.pdf")],
                             chunk_size=200, chunk_overlap=20)
    assert all(c.metadata["document_id"] == "lecture 9+10.pdf" for c in chunks)


def test_chunk_index_is_sequential():
    chunks = split_documents([_long_doc()], chunk_size=200, chunk_overlap=20)
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_chunk_ids_are_unique():
    chunks = split_documents([_long_doc()], chunk_size=200, chunk_overlap=20)
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(ids)  # no empty ids


def test_chunk_ids_are_deterministic():
    docs = [_long_doc()]
    ids_a = [c.metadata["chunk_id"] for c in split_documents(docs, chunk_size=200, chunk_overlap=20)]
    ids_b = [c.metadata["chunk_id"] for c in split_documents(docs, chunk_size=200, chunk_overlap=20)]
    assert ids_a == ids_b


def test_chunks_respect_chunk_size():
    chunks = split_documents([_long_doc()], chunk_size=100, chunk_overlap=0)
    # RecursiveCharacterTextSplitter keeps chunks within chunk_size for whitespace-separated text.
    assert all(len(c.page_content) <= 100 for c in chunks)


def test_short_document_yields_single_chunk():
    docs = [Document(page_content="A short passage about neurons.", metadata={"source": "x.md"})]
    chunks = split_documents(docs, chunk_size=200, chunk_overlap=20)
    assert len(chunks) == 1
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["document_id"] == "x.md"


def test_empty_input_raises_value_error():
    with pytest.raises(ValueError):
        split_documents([])


def test_uses_settings_defaults_when_no_sizes_given():
    # Should not raise and should still enrich metadata when relying on settings defaults.
    chunks = split_documents([_long_doc()])
    assert len(chunks) >= 1
    assert REQUIRED_METADATA_KEYS <= set(chunks[0].metadata)
