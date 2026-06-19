"""Unit tests for the retrieval strategies (multi_query_retrieve, hyde_retrieve).

No live LLM or Chroma calls: the module-level LLM / HyDE chain are monkeypatched,
and a FakeVectorStore supplies canned (Document, distance) pairs. Both retrievers
convert Chroma distance to a similarity via `1 - distance` and stamp it onto
`metadata["score"]`; multi_query additionally dedups on chunk_id and filters by
settings.SIMILARITY_THRESHOLD.
"""

import types
import pytest
from langchain_core.documents import Document

from app.core import settings
import app.retrieval.multiquery as mq
import app.retrieval.hyde as hyde_mod
from app.retrieval.multiquery import multi_query_retrieve
from app.retrieval.hyde import hyde_retrieve


def _doc(chunk_id: str, content: str = "content") -> Document:
    return Document(page_content=content, metadata={"chunk_id": chunk_id})


# ─────────────────────────── multi_query_retrieve ───────────────────────────

def test_multiquery_returns_docs_above_threshold(monkeypatch, fake_vector_store):
    # LLM returns a 3-line numbered list of query variants.
    fake_llm = types.SimpleNamespace(
        invoke=lambda prompt: types.SimpleNamespace(content="1. variant one\n2. variant two\n3. variant three")
    )
    monkeypatch.setattr(mq, "llm", fake_llm)

    keep = _doc("keep")   # distance 0.05 -> similarity 0.95 (>= threshold)
    drop = _doc("drop")   # distance 0.95 -> similarity 0.05 (< threshold)
    vs = fake_vector_store([(keep, 0.05), (drop, 0.95)])

    results = multi_query_retrieve("what is a neuron?", vs, n_variants=3, n_results=5)

    ids = [d.metadata["chunk_id"] for d in results]
    assert "keep" in ids
    assert "drop" not in ids


def test_multiquery_attaches_similarity_score(monkeypatch, fake_vector_store):
    fake_llm = types.SimpleNamespace(
        invoke=lambda prompt: types.SimpleNamespace(content="1. a\n2. b\n3. c")
    )
    monkeypatch.setattr(mq, "llm", fake_llm)

    keep = _doc("keep")
    vs = fake_vector_store([(keep, 0.1)])  # similarity = 0.9
    results = multi_query_retrieve("q", vs, n_variants=3, n_results=5)

    assert len(results) == 1
    assert results[0].metadata["score"] == pytest.approx(0.9)
    assert results[0].metadata["score"] >= settings.SIMILARITY_THRESHOLD


def test_multiquery_deduplicates_repeated_chunks(monkeypatch, fake_vector_store):
    # Every variant returns the SAME chunk -> it must appear only once.
    fake_llm = types.SimpleNamespace(
        invoke=lambda prompt: types.SimpleNamespace(content="1. a\n2. b\n3. c")
    )
    monkeypatch.setattr(mq, "llm", fake_llm)

    dup = _doc("dup")
    vs = fake_vector_store([(dup, 0.1)])
    results = multi_query_retrieve("q", vs, n_variants=3, n_results=5)

    assert [d.metadata["chunk_id"] for d in results] == ["dup"]


def test_multiquery_all_below_threshold_returns_empty(monkeypatch, fake_vector_store):
    fake_llm = types.SimpleNamespace(
        invoke=lambda prompt: types.SimpleNamespace(content="1. a\n2. b\n3. c")
    )
    monkeypatch.setattr(mq, "llm", fake_llm)

    vs = fake_vector_store([(_doc("x"), 0.99)])  # similarity 0.01, far below threshold
    results = multi_query_retrieve("q", vs, n_variants=3, n_results=5)
    assert results == []


# ───────────────────────────── hyde_retrieve ─────────────────────────────

def test_hyde_searches_with_hypothetical_passage(monkeypatch, fake_vector_store):
    captured = {}

    class FakeChain:
        def invoke(self, payload):
            captured["payload"] = payload
            return "A hypothetical scientific passage about neurons and synapses."

    monkeypatch.setattr(hyde_mod, "hyde_doc_chain", FakeChain())

    doc = _doc("h1")
    vs = fake_vector_store([(doc, 0.3)])  # similarity 0.7
    results = hyde_retrieve("how do neurons work?", vs, n_results=5)

    # The chain is invoked with the raw query under the "query" key.
    assert captured["payload"] == {"query": "how do neurons work?"}
    assert len(results) == 1
    assert results[0].metadata["score"] == pytest.approx(0.7)


def test_hyde_returns_topk_without_threshold(monkeypatch, fake_vector_store):
    monkeypatch.setattr(
        hyde_mod, "hyde_doc_chain",
        types.SimpleNamespace(invoke=lambda payload: "hypothetical passage")
    )
    # HyDE does NOT apply SIMILARITY_THRESHOLD — even low-similarity docs come back.
    low = _doc("low")
    vs = fake_vector_store([(low, 0.95)])  # similarity 0.05
    results = hyde_retrieve("q", vs, n_results=5)

    assert [d.metadata["chunk_id"] for d in results] == ["low"]
    assert results[0].metadata["score"] == pytest.approx(0.05)


def test_hyde_respects_n_results(monkeypatch, fake_vector_store):
    monkeypatch.setattr(
        hyde_mod, "hyde_doc_chain",
        types.SimpleNamespace(invoke=lambda payload: "hypothetical passage")
    )
    docs = [(_doc(f"c{i}"), 0.2) for i in range(10)]
    vs = fake_vector_store(docs)
    results = hyde_retrieve("q", vs, n_results=3)
    assert len(results) == 3
