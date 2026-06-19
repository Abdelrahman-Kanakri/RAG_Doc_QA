# Setup

## Prerequisites

- **Python 3.12** (pinned via `.python-version`).
- **[uv](https://docs.astral.sh/uv/)** for dependency and venv management.
- API keys for the providers you intend to use (see [Environment variables](#environment-variables)).
- *(Optional)* an NVIDIA GPU for the local Hugging Face embeddings used during
  evaluation. The CPU path works too — change `device` in `eval/` if you have no GPU.

## Install

```bash
uv sync
```

This creates a managed virtual environment from `pyproject.toml` / `uv.lock`.
A `requirements.txt` is also provided for pip-based installs:

```bash
pip install -r requirements.txt
```

> **Why LangChain is pinned to 0.3.x:** `ragas` 0.4.x is incompatible with
> LangChain v1 (it hard-imports a module path removed in `langchain-community`
> 0.4). The whole LangChain stack is therefore pinned to the 0.3 line. See
> [troubleshooting.md](troubleshooting.md).

## Environment variables

The app loads configuration from a `.env` file at the project root via
`pydantic-settings`. Every key below **without a default is required** — the app
raises a `ValidationError` on startup if it is missing (fail-loud by design).

Create `.env`:

```dotenv
# ── Provider API keys ───────────────────────────────────────────────
MISTRAL_API_KEY=...        # required — embeddings + answer/retrieval LLM
GROQ_API_KEY=...           # required — testset generation + eval judge
HF_API_KEY=...             # required by config (dead config for local bge-m3)
GOOGLE_API_KEY=...         # required by config (Gemini was trialed, not used)

# ── Model names ─────────────────────────────────────────────────────
LARGE_MODEL_NAME=mistral-large-latest      # answer generation
MEDIUM_MODEL_NAME=mistral-medium-latest    # MultiQuery / HyDE generation
SMALL_MODEL_NAME=mistral-small-latest
EMBEDDING_MODEL_NAME=mistral-embed         # MUST match the ingestion embedder
GROQ_MODEL_NAME=llama-3.3-70b-versatile    # testset gen + judge
HF_EMBEDDING_MODEL_NAME=BAAI/bge-m3        # local eval embeddings
GOOGLE_MODEL_NAME=gemini-2.0-flash

# ── LangSmith (tracing) ─────────────────────────────────────────────
LANGSMITH_API_KEY=...
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=rag-doc-qa

# ── Vector store / chunking ─────────────────────────────────────────
VECTOR_DATABASE_PATH=data/chroma           # Chroma persistence dir
SIMILARITY_THRESHOLD=0.65                   # min similarity to keep a chunk
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
COLLECTION_NAME=documents

# ── Environment guard (see troubleshooting.md) ──────────────────────
OPENSSL_CONF=/dev/null
```

> **Do not wrap values in quotes.** `pydantic-settings` reads `"abc"` literally,
> including the quote characters — which corrupts API keys into
> `Bearer "abc"` and produces *Illegal header value* errors.

## Running the API

```bash
uv run uvicorn app.main:app --reload
```

- Served at `http://127.0.0.1:8000`.
- Interactive Swagger docs at `http://127.0.0.1:8000/docs`.
- All endpoints are mounted under the `/api/v1` prefix.

> `app/main.py` also has a `__main__` block, but prefer the `uvicorn` command
> above for local development (auto-reload, standard host/port).

## Running the evaluation

Run from the **project root** so module paths and relative output paths resolve:

```bash
# 1. Synthetic test set → eval/dataset_test/testset.csv
python -m eval.generate_testset

# 2. RAG-Triad comparison → eval/results/results.csv
OPENSSL_CONF=/dev/null python -m eval.run_ragas
```

The first eval run downloads the local `BAAI/bge-m3` weights into `models/`
(~1.3 GB, gitignored). See [evaluation.md](evaluation.md) for details.

## Running the tests

```bash
uv run pytest
```

Fully mocked — no network, Chroma, or model calls.
