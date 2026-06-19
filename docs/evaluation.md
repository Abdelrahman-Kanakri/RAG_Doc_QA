# Evaluation

Phase 3 measures retrieval quality objectively, as a **controlled experiment**:
the only variable is the retrieval strategy (baseline vs. MultiQuery vs. HyDE);
the answer model, the judge, and the embeddings are held constant.

## Pipeline

```text
documents ─► eval/generate_testset.py ─► eval/dataset_test/testset.csv
                                               │
testset.csv ─► eval/run_ragas.py ─► (per strategy: retrieve → answer → score)
                                               ▼
                                     eval/results/results.csv
```

## Step 1 — Synthetic test-set generation (`eval/generate_testset.py`)

Uses RAGAS' `TestsetGenerator` to build a knowledge graph over the documents and
synthesise question / gold-answer / gold-context triples.

- **LLM:** Groq (`GROQ_MODEL_NAME`) — httpx-based, avoids the gRPC event-loop bug
  that makes Gemini unusable inside RAGAS (see [troubleshooting.md](troubleshooting.md)).
- **Embeddings:** local `BAAI/bge-m3` on CUDA — zero API calls, no rate limits.
  Generation embeddings only cluster the internal KG, so *any* embedder works;
  the output CSV is plain text and the vectors are discarded.
- **Throttling:** `RunConfig(max_workers=1, max_retries=15, max_wait=60)` to ride
  out free-tier rate limits.

Output CSV columns:

| Column               | Meaning                                              |
| -------------------- | -------------------------------------------------- |
| `user_input`         | the generated question                              |
| `reference`          | gold / ground-truth answer                          |
| `reference_contexts` | gold passages                                       |
| `persona_name`, `query_style`, `query_length`, `synthesizer_name` | synthesis metadata |

Run (from project root):

```bash
python -m eval.generate_testset
```

`generate_test_set()` is parameterised (`data_path`, `n_docs`, `test_set_size`,
`out_dir`) so a smoke test (`n_docs=5, test_set_size=5`) needs no code edits
before scaling up to the full corpus.

## Step 2 — RAG-Triad scoring (`eval/run_ragas.py`)

For each strategy, `build_dataset` runs it over every test-set row, generates an
answer with the app's own `generate_answer`, and assembles a RAGAS sample
(`user_input`, `retrieved_contexts`, `response`, `reference`). `evaluate` then
scores all rows and the numeric scores are averaged into one row per strategy.

### Strategies compared
- **baseline** — `vector_store.similarity_search(query, k=5)`.
- **multiquery** — `multi_query_retrieve(query, vs, n_variants=3, n_results=5)`.
- **hyde** — `hyde_retrieve(query, vs, n_results=5)`.

### Metrics (RAG Triad + recall)
- **Faithfulness** — is the answer supported by the retrieved context?
- **ResponseRelevancy** — does the answer address the question? (answer relevance)
- **LLMContextPrecisionWithReference** — are the retrieved contexts relevant? (context relevance)
- **LLMContextRecall** — did retrieval find the contexts the gold answer needs?

### Roles (kept strictly separate to avoid confounding)
- **System under test** — Mistral, used for *all* retrieval-side LLM work
  (MultiQuery expansion, HyDE passage) **and** final answer generation. This is
  the constant across all three strategies.
- **Judge** — Groq (`GROQ_MODEL_NAME`), a *different* model so it never grades
  its own output; httpx-based to avoid the Gemini gRPC bug.
- **Judge embeddings** — local `BAAI/bge-m3`, needed by the relevancy/context metrics.

Run (from project root):

```bash
OPENSSL_CONF=/dev/null python -m eval.run_ragas
```

Output: `eval/results/results.csv` — one row per strategy, one column per metric.

## Reading the results

The RAG Triad is **multi-objective** — there are four numbers per strategy, and
the "best" strategy may be a trade-off. Rank by the metric that matters most for
the use case (faithfulness is usually the priority for a grounded QA system).

## Pre-flight checklist

1. **Chroma must be populated** before scoring — run ingestion first. An empty
   collection returns no contexts → metrics come back NaN/0.
2. **Watch daily token budgets.** The judge fires roughly `4 metrics × rows ×
   3 strategies` calls plus answer/expansion generation. If Groq returns a
   per-day 429, switch `GROQ_MODEL_NAME` to a model with a fresh budget (limits
   are per-model).
3. **Run from the project root** so `eval/dataset_test/` and `eval/results/`
   resolve correctly.

See [troubleshooting.md](troubleshooting.md) for the provider rate-limit details
and the OpenSSL segfault guard.
