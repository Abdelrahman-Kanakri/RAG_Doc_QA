# Mentor Questions & Answers — RAG Doc QA

## Phase 1 — System Design

### Step 1: Metadata Schema

**Q: What metadata fields would you store alongside each chunk, and why is each one needed to produce a citation?**
A: Every field serves a specific job in either retrieval or citation generation. No field is stored "just in case."

**Q: What happens if two users upload files with the same name? What field makes each document globally unique?**
A: `document_id` — a hash of the file's byte content. Same content = same hash, different content = different hash, regardless of filename. A UUID would always be unique and couldn't detect duplicates.

**Q: If a chunk spans pages 4–6, is storing only `start_page` good enough for the user?**
A: No. The citation would say "page 4" but the answer might be on page 6. Store both `start_page` and `end_page` so the citation can say "pages 4–6."

**Q: What field records when the document was ingested, and why does it matter?**
A: `timestamp` — if the same document is uploaded twice (January vs March), the timestamp tells you which version is newer so you can detect and handle stale chunks.

**Q: What field holds the actual text the LLM reads when generating an answer?**
A: `chunk_content` (stored as ChromaDB's `document` field). The LLM reads the raw chunk text — not keywords or headers.

**Q: What free document structure (no LLM needed) tells you where in a paper a chunk comes from?**
A: Section headings — `chunk_headers`. Research papers have Introduction, Methods, Results, Discussion etc. These are extractable for free from document structure (Markdown `#` headers, PDF outline).

**Q: If you store a hypothetical question per chunk at index time, why does that improve retrieval later?**
A: At query time you have a question; in the vector store you have answer-shaped text. These don't look alike semantically. If you store a hypothetical question the chunk answers, then query-to-stored-question similarity is much tighter — this is multi-vector retrieval.

---

### Step 2: Chunking Strategy

**Q: Which chunking strategy is safest when you don't know what documents are coming in?**
A: Recursive character text splitting. It needs zero model calls, is purely algorithmic, and behaves consistently regardless of document type or size.

**Q: What does semantic chunking cost you at index time compared to recursive?**
A: Semantic chunking embeds every sentence to detect topic shifts — many embedding API calls just to decide where to cut. Recursive is pure string splitting — no model calls, no cost.

**Q: Why would you avoid proposition chunking in production despite it producing the cleanest chunks?**
A: It sends every chunk through an LLM to decompose it into atomic facts. For a 200-page document, that's hundreds of LLM calls at index time — catastrophic latency and cost.

**Q: What happens if `chunk_size` is too small — say, 100 characters?**
A: Chunks lose self-contained meaning. A chunk reading "...therefore the treatment was effective" has no context — effective for what? The LLM can't answer anything from it.

**Q: What happens if `chunk_overlap` is too large — say, 80% of chunk size?**
A: Redundancy. Chunks 1 and 2 share 80% of their text. When you retrieve top-5 chunks, you get 5 nearly identical results — all from the same region of the document. You've wasted 4 retrieval slots.

**Q: What word describes a chunk that contains enough context to stand alone?**
A: Self-contained. Each chunk should be independently meaningful without needing surrounding chunks for context.

---

### Step 3: Embedding Model

**Q: What happens to query latency if your embedding model runs locally on CPU?**
A: CPU inference is slow — embedding a single query can take seconds. Every user query pays this cost, making the 3s latency budget impossible to meet.

**Q: 1000 characters ≈ how many tokens?**
A: ~250 tokens. Rule of thumb: 1 token ≈ 4 characters. 1000 ÷ 4 = 250. Well within Mistral embed's 8192 token limit.

**Q: Why does a smaller embedding dimension (1024 vs 1536) help at retrieval time?**
A: Smaller vectors = less storage + faster cosine similarity computation across potentially millions of chunks.

**Q: What breaks in your system if the embedding API goes down mid-ingestion?**
A: Partial index — some chunks embedded and stored, others not. The document appears ingested but is incomplete. Solution: fail loudly with 503, don't silently partially index.

**Q: Is it safe to switch embedding models without re-indexing?**
A: No. Two different models produce vectors in different geometric spaces. Even at the same dimension size (1024), a Mistral vector and a BGE vector are not comparable. Switching models requires re-embedding everything.

---

### Step 4: Vector Store

**Q: ChromaDB metadata must be flat — which field in your schema violates this rule?**
A: `keywords: ["python", "rag", "embeddings"]` — a list. ChromaDB only accepts strings, ints, and floats in metadata. No nested objects, no lists.

**Q: How would you fix a list field to be ChromaDB-compatible?**
A: Serialize to a comma-separated string: `"python, rag, embeddings"`. Split on retrieval when you need to search by keyword.

**Q: One collection for all documents, or one per user? What's the trade-off?**
A: One collection + `user_id` in metadata + `where` filter on every query is simpler to operate. Per-user collections give hard isolation but are harder to manage. For single-user service: one collection. For multi-tenant SaaS: per-user collections (missing a `where` filter leaks data).

---

### Step 5: Query Enhancement

**Q: Which document type benefits most from MultiQuery — technical papers or blog posts?**
A: Technical papers. The vocabulary gap between how a user asks ("what causes memory loss?") and how a paper is written ("mechanisms of cognitive impairment") is much larger than for blog posts, which already use everyday language.

**Q: HyDE generates a fake answer from LLM training data. What's the risk for private uploaded documents?**
A: The LLM has never seen the user's private documents. Its hypothetical answer is generated from general training knowledge, which may be wrong or unrelated to the document content — steering retrieval toward the wrong chunks. The risk scales up for private/proprietary documents.

**Q: If you run MultiQuery (3 variants) + HyDE together, how many API calls happen before retrieval?**
A: 1 MultiQuery LLM call + 3 embedding calls + 1 HyDE LLM call + 1 HyDE embedding call = 6 API calls minimum. At 500ms–1s per LLM call, you've already exceeded the 3s latency budget before retrieval or generation.

**Q: Does HyDE's hallucination risk go up or down for documents the LLM has never seen?**
A: Up — significantly. The hypothetical is generated from training data with no knowledge of the document. It can confidently point retrieval in the completely wrong direction.

---

### Step 6: Failure Modes

**Q: What does your text extractor return for a scanned PDF with no text layer?**
A: Empty string or whitespace. The file is valid PDF but contains images, not text. Return HTTP 422 (Unprocessable Entity) — the format is supported but the content can't be processed.

**Q: What should you do with an empty chunk before embedding it?**
A: Delete/skip it. Filter empty strings before any embedding call — never store empty chunks in the vector store.

**Q: What should your system do if all top-5 retrieved chunks score below the similarity threshold?**
A: Return "I don't have information about this in your uploaded documents" — do not pass low-quality context to the LLM. Threshold: 0.7 similarity score.

**Q: How would you detect that the LLM is ignoring retrieved context?**
A: Enforce it in the system prompt: *"Answer only based on the provided context. If the context does not contain the answer, say no answer can be given."* You prevent it rather than detect it after the fact.

**Q: A user uploads policy.pdf twice. How do you detect the duplicate?**
A: Hash the file content bytes. Same file = same hash = same `document_id`. Check if `document_id` already exists in the collection before ingesting.

**Q: What single system prompt instruction keeps the LLM grounded to retrieved context?**
A: *"Answer only based on the provided context. If the context does not contain the answer, say no answer can be given."*

---

### Step 7: API Contract

**Q: What HTTP status code means "I understand this format but can't process the content"?**
A: 422 Unprocessable Entity — used for scanned PDFs (valid PDF format, but no extractable text).

**Q: What HTTP status code means "I don't support this file format at all"?**
A: 415 Unsupported Media Type — used for .xlsx, .docx, or any format outside your supported list.

**Q: After ingestion, what fields does the caller need in the response?**
A: `document_id` (to reference the document later), `chunks` (count of chunks created — confirms ingestion worked), `message` (human-readable confirmation).

**Q: What does a `Source` object need to contain so the user can locate the original text?**
A: `source_name` (which file), `start_page` (which page), `chunk_headers` (which section), `chunk_content` (the exact text used — so user can verify).

**Q: What two fields does `GET /health` need to be genuinely useful to a monitoring tool?**
A: `status: "healthy"` (is the API alive?) and `vector_store: "connected" | "disconnected"` (is ChromaDB reachable?). FastAPI running but ChromaDB down is a partial failure — health must distinguish these.

---

## Phase 2 — Implementation (questions added as we go)

### Step 1: config.py
**Q: What settings does your app need, based on every decision made in Phase 1?**
A: *(answer this yourself before implementing)*
