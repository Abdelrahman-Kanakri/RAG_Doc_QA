# RAG Document QA

A **Retrieval-Augmented Generation** service that ingests PDF/Markdown documents,
indexes them in a vector store, and answers questions over them with **grounded,
cited answers** — no hallucinations, no out-of-scope replies.

Built with FastAPI, ChromaDB, and Mistral AI, with query enhancement
(MultiQuery + HyDE) and a full **RAGAS** evaluation harness comparing retrieval
strategies on the RAG Triad.

> This was my first end-to-end RAG project, written by hand in **mentor mode** —
> every line designed and typed myself, with an AI mentor coaching the design
> decisions rather than writing the code. The full design rationale and the
> four-phase build journey are documented in [`docs/`](docs/).

---

## What it demonstrates

- **Design-first RAG** — a deliberate metadata schema, chunking strategy, and
  embedding choice, each justified by trade-offs (see [docs/architecture.md](docs/architecture.md)).
- **Grounded generation** — a system prompt that refuses to answer outside the
  retrieved context, with per-source citations (file, page, section, score).
- **Query enhancement** — manual MultiQuery expansion and HyDE, implemented from
  scratch (not the off-the-shelf LangChain retrievers) for learning value.
- **Evaluation** — RAGAS synthetic test-set generation and a controlled
  RAG-Triad comparison of *baseline vs. MultiQuery vs. HyDE* (see [docs/evaluation.md](docs/evaluation.md)).
- **Production hygiene** — Pydantic-validated I/O, structured JSON logging,
  fail-loud error handling, a health check that verifies vector-store
  connectivity, LangSmith tracing of every retrieval/generation call, and a
  mocked unit-test suite.
- **Containerized** — a `uv`-based Dockerfile and a `docker-compose.yml` with a
  named volume for Chroma persistence and secrets injected at runtime (see
  [Deployment](#deployment-docker)).

---

## Architecture at a glance

```text
upload ─► load ─► chunk (+metadata) ─► embed ─► store          (ingestion)
                                                   │
query ─► expand (MultiQuery/HyDE) ─► retrieve ─► filter ─► generate + cite   (query)
```

| Layer        | Module                        | Responsibility                                  |
| ------------ | ----------------------------- | ----------------------------------------------- |
| API          | `app/api/routes.py`           | `/health`, `/ingest`, `/query` endpoints        |
| Ingestion    | `app/ingestion/`              | PDF/Markdown loading + recursive chunking       |
| Retrieval    | `app/retrieval/`              | Chroma store, MultiQuery, HyDE                   |
| Generation   | `app/generation/answerer.py`  | grounded answer + citation assembly             |
| Schemas      | `app/schemas/models.py`       | Pydantic response contracts                     |
| Core         | `app/core/`                   | settings (pydantic-settings) + structlog logging |
| Evaluation   | `eval/`                       | RAGAS test-set generation + RAG-Triad scoring   |

Full module-by-module walkthrough: **[docs/architecture.md](docs/architecture.md)**.

---

## Quick start

> Requires Python 3.12 and [`uv`](https://docs.astral.sh/uv/). Full instructions,
> including the `.env` variables, are in **[docs/setup.md](docs/setup.md)**.

```bash
# 1. Install dependencies into a managed venv
uv sync

# 2. Create your .env with the API keys + model names (see docs/setup.md)
cp .env.example .env   # then fill in the values

# 3. Run the API
uv run uvicorn app.main:app --reload
```

The API is then served at `http://127.0.0.1:8000`, with interactive docs at
`http://127.0.0.1:8000/docs`.

### Using the API

```bash
# Health (also reports vector-store connectivity)
curl http://127.0.0.1:8000/api/v1/health

# Ingest a document (PDF or Markdown only)
curl -F "file=@paper.pdf" http://127.0.0.1:8000/api/v1/ingest

# Ask a question
curl -X POST "http://127.0.0.1:8000/api/v1/query?query=what%20is%20backpropagation%3F"
```

---

## Deployment (Docker)

The service is containerized with a `uv`-based [`Dockerfile`](Dockerfile) and a
[`docker-compose.yml`](docker-compose.yml).

```bash
# Build and start (reads secrets from .env via env_file)
docker compose up -d --build

# Check health
curl http://localhost:8000/api/v1/health
```

Design notes:

- **Secrets** are injected at runtime via compose `env_file: .env` — never baked
  into an image layer. The same `.env` that drives local runs drives the container,
  including the `LANGSMITH_*` tracing vars.
- **Persistence** — a named volume (`chroma_data`) is mounted at `/app/data/chroma`
  (matching `VECTOR_DATABASE_PATH`), so ingested vectors survive restarts.
- **Health** — compose health-checks `GET /api/v1/health` with a Python probe
  (the slim image has no `curl`).
- **Build context** — a [`.dockerignore`](.dockerignore) keeps `.venv/`, `models/`
  (the 1.3 GB local eval weights), `data/`, and `.env` out of the image.

> **Status:** the deployment scripts are written but the image build has not yet
> been verified end-to-end on this machine. See
> [docs/setup.md](docs/setup.md#deployment-docker) for the full walkthrough and
> the persistence round-trip test.

---

## Evaluation

The retrieval strategies are compared as a controlled experiment — only the
retriever changes; the answer model and judge stay fixed.

```bash
# 1. Generate a synthetic test set from your documents
python -m eval.generate_testset

# 2. Score baseline vs. MultiQuery vs. HyDE on the RAG Triad
OPENSSL_CONF=/dev/null python -m eval.run_ragas   # → eval/results/results.csv
```

Metrics, provider roles, and the rationale are in **[docs/evaluation.md](docs/evaluation.md)**.

---

## Testing

```bash
uv run pytest
```

The suite is fully mocked — no network, no Chroma, no model calls. See
`test/conftest.py` for the fakes.

---

## Documentation

| Doc | Contents |
| --- | --- |
| [docs/setup.md](docs/setup.md)                     | Install, `.env` reference, run commands |
| [docs/architecture.md](docs/architecture.md)       | Data flow, modules, design decisions, API contract |
| [docs/evaluation.md](docs/evaluation.md)           | RAGAS test-set generation + RAG-Triad comparison |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Environment gotchas hit while building this |

---

## Project status

All four phases are complete. Phases 1–3 (design → implementation → evaluation)
are finished and verified; **Phase 4** (Dockerization + LangSmith tracing) is
implemented — the `Dockerfile`, `docker-compose.yml`, and `.dockerignore` are
written and LangSmith tracing is wired into `app/core/config.py`. The one
outstanding item is verifying the container build/persistence end-to-end on the
build host. See [docs/architecture.md](docs/architecture.md#roadmap).
