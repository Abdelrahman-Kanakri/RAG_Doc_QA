# Troubleshooting & known issues

Environment-specific gotchas hit while building this on Fedora 43 / Nobara
(bleeding-edge glibc) with a `uv`-managed standalone CPython. Kept here because
they cost real debugging time and may bite anyone on a similar setup.

## 1. `import pandas` / `import ragas` segfaults (SIGSEGV)

**Symptom:** importing pandas, pyarrow, `ragas`, or `datasets` crashes with
exit 139. The Python traceback is unreliable — it points at numpy/pyarrow and
the crash site moves between runs (a sign of heap corruption).

**Cause:** `uv`'s python-build-standalone statically links OpenSSL. pandas (and
pyarrow's Arrow `curl` init) makes OpenSSL read the system `/etc/ssl/openssl.cnf`,
which loads the host **pkcs11 engine** `.so`; that engine crashes inside the
statically-linked OpenSSL.

**Fix:** set `OPENSSL_CONF=/dev/null` *before* the first OpenSSL use.

- In scripts, as the first lines (the eval scripts already do this):
  ```python
  import os
  os.environ.setdefault("OPENSSL_CONF", "/dev/null")
  ```
- On the shell, prefix ad-hoc commands:
  ```bash
  OPENSSL_CONF=/dev/null python -m eval.run_ragas
  ```
- Project-wide, it's in `.env`. The FastAPI app does **not** need it (it never
  enumerates OpenSSL digests; only pandas/pyarrow do).

**Diagnosis tip:** when a Python traceback is nonsense, get the real C culprit
with `gdb -batch -ex run -ex bt --args python -c "import pandas"`.

## 2. VS Code autocomplete crashes (same root cause)

The same segfault crashes the editor's Jedi/Pylance subprocess just from having
a pandas/pyarrow-importing file **open** — no run required.

**Fix:** `OPENSSL_CONF=/dev/null` is set in `.env` and in `.vscode/settings.json`
(`terminal.integrated.env.linux` + `python.envFile`).

> **Gotcha:** the Jedi subprocess inherits the extension-host environment
> captured at window startup, so editing `.env`/settings does **not** take effect
> until **Developer: Reload Window**.

## 3. RAGAS needs LangChain 0.3.x

`ragas` 0.4.x hard-imports a module path that was removed in
`langchain-community` 0.4 (LangChain v1). No ragas version supports v1 yet.

**Fix:** the whole LangChain stack is pinned to the 0.3 line in
`pyproject.toml` / `requirements.txt`. The app's stable APIs (PyPDFLoader,
ChatMistralAI, Chroma, splitters) survive the pin fine.

## 4. Provider rate limits during evaluation

Free tiers fail in different ways — knowing which matters:

- **TPM (tokens-per-minute) 429** — *recoverable*; tenacity waits out the ~60 s
  window and continues. This is the benign 429.
- **TPD (tokens-per-day) 429** — a *hard wall*, not fixable by retry. Groq limits
  are **per-model**, so switching `GROQ_MODEL_NAME` (e.g. to
  `llama-3.1-8b-instant`) gives a fresh daily budget.
- **422 Unprocessable Entity ≠ 429** — a malformed request, not retryable. Seen
  when leftover params from one provider (e.g. Gemini's `max_output_tokens` /
  `transport`) are passed to another (Mistral wants `max_tokens`, no `transport`).

**Mitigation in code:** the eval scripts use
`RunConfig(max_workers=1, max_retries=15, max_wait=60)` and local embeddings to
keep the API burst small.

## 5. Gemini is incompatible with the RAGAS async engine

`langchain-google-genai` uses `grpc.aio`, whose channel is bound to the event
loop it was created in. RAGAS runs each transform stage in its own
`asyncio.run()` (which closes the loop), breaking the gRPC channel — you get
`Event loop is closed` / `_interceptors_task` errors and a hard crash.
`transport="rest"` does **not** fix it (it only affects the sync client).

**Rule:** inside any RAGAS pipeline (test-set gen *and* the judge), use only
httpx-based providers — Groq, Mistral, OpenAI. Gemini is avoided throughout.

## 6. Local embeddings (`BAAI/bge-m3`) cache

`HuggingFaceEmbeddings(..., cache_folder="models")` downloads weights into
`<project>/models/` (~1.3 GB, gitignored) instead of `~/.cache/huggingface`.
Reuse is automatic on later runs; `HF_HUB_OFFLINE=1` (set at the top of the eval
scripts) skips the network revision-check for a fully offline load.

## 7. Leading-space filename trap

A recurring local glitch created source files with a **leading space** in the
name (e.g. `app/retrieval/ hyde.py`). Python can't import these →
`ModuleNotFoundError` for an obviously-present module. When an import fails,
check `ls -la` and `mv` to strip the space.

## 8. Always run modules from the project root

Use `python -m app.ingestion.loaders` / `python -m eval.run_ragas` from the
project root — never `python app/ingestion/loaders.py` directly. Running a file
directly puts its directory on `sys.path` (which can shadow stdlib `logging`) and
breaks the relative output paths the eval scripts write to.
