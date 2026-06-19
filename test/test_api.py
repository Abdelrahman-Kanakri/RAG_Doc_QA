"""Endpoint tests for the FastAPI app (health / query / ingest).

Uses FastAPI's TestClient. All external work (vector store, retrieval, answer
generation, loaders) is monkeypatched on the `app.api.routes` module, so no
network, no Chroma, no model calls happen.
"""

import types
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.main import app
import app.api.routes as routes

client = TestClient(app)

BASE = "/api/v1"


# ───────────────────────────────── health ─────────────────────────────────

def test_health_connected(monkeypatch):
    monkeypatch.setattr(routes, "init_vectorStore", lambda: object())  # succeeds
    resp = client.get(f"{BASE}/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["vector_store"] == "connected"


def test_health_disconnected(monkeypatch):
    def boom():
        raise RuntimeError("cannot reach chroma")
    monkeypatch.setattr(routes, "init_vectorStore", boom)
    resp = client.get(f"{BASE}/health")
    assert resp.status_code == 200
    assert resp.json()["vector_store"] == "disconnected"


# ───────────────────────────────── query ──────────────────────────────────

def test_query_success(monkeypatch):
    doc = Document(page_content="Neurons transmit signals.", metadata={"chunk_id": "c1"})
    monkeypatch.setattr(routes, "init_vectorStore", lambda: object())
    monkeypatch.setattr(routes, "multi_query_retrieve", lambda q, vs: [doc])
    monkeypatch.setattr(routes, "generate_answer", lambda q, docs: {
        "answer": "Neurons transmit electrical signals.",
        "sources": [{
            "source_name": "lec1.pdf",
            "start_page": "1",
            "chunk_content": "Neurons transmit signals.",
            "chunk_headers": "",
            "chunk_score": 0.91,
        }],
    })

    resp = client.post(f"{BASE}/query", params={"query": "what do neurons do?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Neurons transmit electrical signals."
    assert body["sources"][0]["source_name"] == "lec1.pdf"


def test_query_no_documents_returns_404(monkeypatch):
    monkeypatch.setattr(routes, "init_vectorStore", lambda: object())
    monkeypatch.setattr(routes, "multi_query_retrieve", lambda q, vs: [])
    # generate_answer should never be reached; make it fail loudly if it is.
    monkeypatch.setattr(routes, "generate_answer",
                        lambda q, docs: pytest.fail("generate_answer called despite no docs"))

    resp = client.post(f"{BASE}/query", params={"query": "unanswerable question"})
    assert resp.status_code == 404


# ───────────────────────────────── ingest ─────────────────────────────────

def test_ingest_rejects_unsupported_file_type():
    # 415 is raised before any disk write or vector-store call — no mocking needed.
    resp = client.post(
        f"{BASE}/ingest",
        files={"file": ("notes.txt", b"plain text", "text/plain")},
    )
    assert resp.status_code == 415


def test_ingest_success(monkeypatch, tmp_path):
    fake_chunks = [
        Document(page_content="chunk one", metadata={"document_id": "paper.pdf", "chunk_id": "a"}),
        Document(page_content="chunk two", metadata={"document_id": "paper.pdf", "chunk_id": "b"}),
    ]
    monkeypatch.setattr(routes, "choose_loader", lambda path: [Document(page_content="raw")])
    monkeypatch.setattr(routes, "split_documents", lambda content: fake_chunks)
    monkeypatch.setattr(routes, "init_vectorStore", lambda: object())
    monkeypatch.setattr(routes, "add_chunks", lambda vs, chunks: None)

    resp = client.post(
        f"{BASE}/ingest",
        files={"file": ("paper.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["chunks"] == 2
    assert body["document_id"] == "paper.pdf"

    # The endpoint writes the upload to data/docs/uploads/<name>; clean it up.
    from pathlib import Path
    Path("data/docs/uploads/paper.pdf").unlink(missing_ok=True)


def test_ingest_empty_chunks_returns_422(monkeypatch):
    monkeypatch.setattr(routes, "choose_loader", lambda path: [Document(page_content="raw")])
    monkeypatch.setattr(routes, "split_documents", lambda content: [])  # nothing usable
    monkeypatch.setattr(routes, "init_vectorStore", lambda: object())

    resp = client.post(
        f"{BASE}/ingest",
        files={"file": ("empty.pdf", b"%PDF-1.4", "application/pdf")},
    )
    assert resp.status_code == 422

    from pathlib import Path
    Path("data/docs/uploads/empty.pdf").unlink(missing_ok=True)
