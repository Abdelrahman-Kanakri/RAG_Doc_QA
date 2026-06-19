# Architecture

This document describes the system design, the responsibility of each module,
the key design decisions (with their trade-offs), and the API contract. It
doubles as the record of the four-phase build journey.

## 1. Goal & constraints

A FastAPI service that accepts PDF/Markdown uploads, indexes them, and answers
questions over them with **cited, grounded answers** — no hallucinations, no
out-of-scope answers.

Constraints that shaped the design:

- All API I/O validated with Pydantic v2.
- **No answer without a source citation.**
- Target latency budget: ~3 s per query.
- Fail loudly — a clear error beats a silent wrong answer.

## 2. Data flow

```text
INGESTION
  upload (PDF/MD) ─► loaders.choose_loader ─► chunking.split_documents
                                                   │  (+ chunk_id, document_id, page…)
                                                   ▼
                              vectorstore.add_chunks ─► ChromaDB (Mistral embeddings)

QUERY
  question ─► multiquery.multi_query_retrieve ─► ChromaDB similarity search
                  (LLM expands to N variants)        │  (filter < SIMILARITY_THRESHOLD, dedup)
                                                      ▼
                          generation.generate_answer ─► grounded answer + sources[]
```

`routes.py` is the **orchestrator**: it calls each worker module, catches their
exceptions, logs structured events, and maps failures to HTTP status codes.
Worker modules (loaders, chunking, vectorstore, retrieval, generation) know
nothing about HTTP or logging — they do one job and either return a result or
raise. This "lego" separation is deliberate.

## 3. Modules

### `app/core/`
- **`config.py`** — `pydantic-settings` `Settings` singleton. Required secrets
  have no default → `ValidationError` on startup if missing. `extra="ignore"`
  silently drops unknown `.env` keys.
- **`logging.py`** — `structlog` configured once at module level to emit JSON
  logs. `get_logger(name)` returns a named bound logger.

### `app/ingestion/`
- **`loaders.py`** — `load_pdf` (PyPDFLoader, `mode='page'` → one Document per
  page, preserving `metadata['page']` for citations), `load_markdown`
  (`Path.read_text`), `choose_loader` (dispatch by extension), and
  `load_directory` (recursive batch load for evaluation).
- **`chunking.py`** — `RecursiveCharacterTextSplitter` (1000 chars / 200 overlap
  by default). Enriches each chunk with `chunk_id` (a hash of `content + index`),
  `chunk_index`, `document_id` (the source filename), and placeholder
  `keywords` / `chunk_headers`. Raises if no chunks result (e.g. image-only PDF).

### `app/retrieval/`
- **`vectorstore.py`** — `init_vectorStore` (open/create a persistent Chroma
  collection with Mistral embeddings), `add_chunks` (idempotent insert that skips
  chunks whose `chunk_id` already exists), `view_collection` (count inspector).
- **`multiquery.py`** — `multi_query_retrieve`: one LLM call expands the query
  into N variants, each is searched, results are deduped on `chunk_id` and
  filtered by `SIMILARITY_THRESHOLD` (Chroma returns *distance*; similarity is
  `1 - distance`). The kept similarity is stamped onto `metadata["score"]`.
- **`hyde.py`** — `hyde_retrieve`: an LLM writes a hypothetical *passage* that
  reads like a source document, that passage is embedded, and real chunks
  similar to it are retrieved. Retrieval only — answering stays in the
  generation layer.

### `app/generation/`
- **`answerer.py`** — `generate_answer(query, documents)`: injects the documents
  into a grounding prompt that forbids outside knowledge, calls the Mistral
  `LARGE_MODEL_NAME`, and returns `{"answer", "sources"}` with one citation dict
  per document. Retrieval is the caller's job; this only generates + cites.

### `app/schemas/`
- **`models.py`** — Pydantic response models: `GetHealth`, `ResponseIngest`,
  `Source`, `ResponseQuery`. `Source` is defined before `ResponseQuery` (Python
  reads top-down). Note `start_page: str` and `chunk_headers: str` — they can be
  `"Unknown"` and Chroma flattens metadata to strings.

### `app/api/`
- **`routes.py`** — the three endpoints (see [API contract](#5-api-contract)),
  wired with `response_model=` and structured logging at each step.

### `eval/`
- See [evaluation.md](evaluation.md).

## 4. Key design decisions

### Chunk metadata schema
Every chunk carries the fields needed for retrieval *and* citation — nothing
"just in case":

| Field           | Purpose                                                        |
| --------------- | ------------------------------------------------------------- |
| `chunk_id`      | hash-based unique id; enables idempotent insert               |
| `document_id`   | source filename (a content hash is the production upgrade)     |
| `page`          | page number for citations (from PyPDFLoader)                   |
| `chunk_index`   | ordinal position within the document                          |
| `chunk_headers` | section heading (placeholder; extractable for free)           |
| `keywords`      | comma-separated string (**not a list** — Chroma needs flat values) |
| `score`         | similarity, attached at retrieval time                        |

> **Chroma constraint:** metadata must be flat — no nested dicts or lists. Lists
> are serialised to comma-separated strings.

### Chunking strategy — recursive
Chosen over semantic and proposition chunking because it needs **zero model
calls** at index time and behaves predictably for any document type. Semantic
chunking embeds every sentence to find split points; proposition chunking sends
each chunk through an LLM — both are too costly/slow for ingestion. ~1000 chars
≈ 250 tokens, well within Mistral embed's 8192-token limit; 200-char overlap
preserves boundary context without collapsing chunks into redundancy.

### Embedding model — `mistral-embed`
1024-dim (smaller = faster similarity, less storage than OpenAI's 1536), API-based
(avoids local-CPU query latency). **Switching embedders requires re-indexing** —
vectors from two models live in different geometric spaces even at the same
dimension, so the query embedder must be identical to the ingestion embedder.

### Vector store — ChromaDB, single collection
Sufficient for a single-user service. Multi-tenant upgrade path: add `user_id`
to metadata + a `where` filter on every query (the risk being that one missing
filter leaks data across users).

### Query enhancement — MultiQuery (default) + HyDE (evaluated)
MultiQuery helps most on technical documents, where the vocabulary gap between
how users ask and how papers are written is largest. HyDE was deferred to the
evaluation phase because (a) stacking it with MultiQuery blows the latency
budget, and (b) for *private* documents the LLM has never seen, its hypothetical
answer is drawn from training data and can steer retrieval the wrong way. Both
are measured head-to-head in [evaluation.md](evaluation.md).

### Failure modes & mitigations

| Failure                                   | Mitigation                                            |
| ----------------------------------------- | ---------------------------------------------------- |
| Scanned/image PDF (no text layer)         | empty chunks → **HTTP 422**                          |
| Unsupported format (.xlsx, .docx, …)      | extension check at upload boundary → **HTTP 415**    |
| Duplicate upload                          | `chunk_id` already present → skipped on insert       |
| All retrieved chunks below threshold      | **HTTP 404** "no relevant documents" — never sent to LLM |
| LLM ignores / hallucinates beyond context | grounding system prompt forbids outside knowledge    |
| Embedding API / Chroma down               | `init_vectorStore` raises → `/health` reports `disconnected`; ingest returns 500 |

## 5. API contract

All routes are under the `/api/v1` prefix.

### `GET /health`
```text
200 → { "status": "healthy", "vector_store": "connected" | "disconnected" }
```

### `POST /ingest`  (multipart file upload)
```text
file: UploadFile         # PDF or Markdown only

200 → { "document_id": str, "chunks": int, "message": str }
415 → unsupported file type
422 → no usable chunks (e.g. image-only PDF)
500 → unexpected ingestion error (uploaded file is cleaned up)
```

### `POST /query`  (query as a request parameter)
```text
query: str

200 → { "answer": str, "sources": [ Source, … ] }
404 → no relevant documents found
```

`Source` = `{ source_name, start_page, chunk_content, chunk_headers, chunk_score }`.

## 6. Build journey (mentor mode)

The project was built design-first, in four phases, with an AI mentor coaching
decisions rather than writing code:

1. **Phase 1 — System design.** Data flow, metadata schema, chunking/embedding/
   vector-store choices, query-enhancement strategy, failure modes, and the
   OpenAPI contract — all locked before any code. Captured in `mentor_questions.md`.
2. **Phase 2 — Implementation.** Built in dependency order: config → logging →
   loaders → chunking → vectorstore → multiquery → answerer → routes → main,
   then Pydantic schemas and package `__init__` exports.
3. **Phase 3 — Evaluation.** RAGAS synthetic test set + RAG-Triad comparison of
   retrieval strategies; HyDE implemented here. See [evaluation.md](evaluation.md).
4. **Phase 4 — Deployment.** Containerization and observability. See
   [Deployment](#7-deployment) below.

## 7. Deployment

The service ships as a container, defined by three files at the repo root:

- **`Dockerfile`** — based on the official `uv` image
  (`ghcr.io/astral-sh/uv:python3.12-bookworm-slim`). Dependencies install from
  `uv.lock` with `uv sync --frozen` in a layer *before* the source is copied, so
  editing `app/` doesn't re-resolve the (large) dependency tree. The app package
  installs in a second layer. The container runs `uvicorn app.main:app --host
  0.0.0.0 --port 8000` from the CLI — binding `0.0.0.0` rather than relying on the
  `__main__` block, so the host/port live in the image, not the code.
- **`docker-compose.yml`** — publishes port 8000, injects secrets via
  `env_file: .env` (so keys are never baked into an image layer), mounts a named
  volume `chroma_data` at `/app/data/chroma` (matching `VECTOR_DATABASE_PATH`) for
  vector persistence across restarts, and health-checks `GET /api/v1/health` with
  a Python probe (the slim image has no `curl`).
- **`.dockerignore`** — excludes `.venv/`, `models/` (the 1.3 GB local eval
  weights), `data/`, `__pycache__/`, `.git/`, and `.env` from the build context.

**Observability (LangSmith).** Tracing is driven entirely by `LANGSMITH_*`
environment variables, which LangChain reads from `os.environ`. Because
`pydantic-settings` loads `.env` into the `Settings` *object* (not into the
process environment), `config.py` bridges the four `LANGSMITH_*` values into
`os.environ` right after `settings = Settings()` — so local runs trace too. In
the container the same vars arrive via compose `env_file`, so tracing works there
with no extra wiring.

> **Status:** the deployment artifacts are written; the image build and the
> volume persistence round-trip have not yet been verified end-to-end on the
> build host. That verification is the only remaining Phase 4 step.

## Roadmap

- **Verify the container** — build the image, confirm `/health` reports
  `connected`, and prove the `chroma_data` volume survives an ingest →
  `down` → `up` → query round-trip.
- **Content-hash `document_id`** — switch from filename to a hash of file bytes
  for true cross-filename deduplication.
- **Multi-vector / hypothetical-question indexing** — to tighten query-to-chunk
  similarity (designed in Phase 1, not yet built).
