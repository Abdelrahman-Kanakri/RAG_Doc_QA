"""Synthetic test-set generation for RAG evaluation (Phase 3, step 1).

Uses RAGAS' ``TestsetGenerator`` to build a knowledge graph over the project
documents and synthesise question / ground-truth-answer / gold-context triples.
The resulting CSV (``eval/dataset_test/testset.csv``) is the input consumed by
``eval/run_ragas.py`` to score the retrieval strategies.

Provider split (chosen to survive free-tier rate limits):
    * LLM        — Groq (``settings.GROQ_MODEL_NAME``), httpx-based, no gRPC issues.
    * Embeddings — local Hugging Face ``BAAI/bge-m3`` on CUDA, zero API calls.
      The generation embeddings only cluster the internal KG, so any model works;
      the produced CSV is plain text and the vectors are discarded.

Environment guards (must run BEFORE importing pandas / ragas):
    * ``OPENSSL_CONF=/dev/null`` — avoids a SIGSEGV when pandas/pyarrow load the
      host's OpenSSL pkcs11 engine inside this uv-managed CPython.
    * ``HF_HUB_OFFLINE=1`` — load the cached bge-m3 weights without a network call.

Run from the project root: ``python -m eval.generate_testset``.
"""

import os
import logging
import time
from pathlib import Path

os.environ.setdefault("OPENSSL_CONF", "/dev/null")

# Offline mode for Hugging Face Hub to avoid network calls during tests
os.environ.setdefault("HF_HUB_OFFLINE", "1") 


from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas.run_config import RunConfig
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd
from app.ingestion import load_directory
from app.core import settings

# Set the Providers API key in the environment variables
os.environ["GROQ_API_KEY"] = settings.GROQ_API_KEY
os.environ["MISTRAL_API_KEY"] = settings.MISTRAL_API_KEY

logging.basicConfig(
    level = logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
    )

log = logging.getLogger("generate_testset")

# ─── RAGAS Function  ────────────────────────────────────────────────────────────────────
def generate_test_set(
    data_path: str = "data/docs/Nueral networks",
    n_docs: int | None = 5,
    test_set_size: int = 5,
    out_dir:str = "eval/dataset_test") -> pd.DataFrame:
    """Generate a RAGAS synthetic test set and save it as a CSV.

    Args:
        data_path (str): Directory of source documents to build the test set from
            (resolved by ``load_directory``).
        n_docs (int | None): Cap on how many loaded page-documents to feed the
            generator. ``None`` uses all of them; a small number gives a fast
            smoke test.
        test_set_size (int): Target number of question/answer rows to synthesise.
            The actual count may differ slightly — RAGAS rounds the query
            distribution and can drop rows for tiny corpora.
        out_dir (str): Directory where ``testset.csv`` is written (created if
            missing).

    Returns:
        pd.DataFrame: The generated test set with columns ``user_input`` (question),
        ``reference`` (gold answer), ``reference_contexts`` (gold passages) and
        RAGAS synthesis metadata.
    """
    
    # ── Load ──────────────────────────────────────────────────────────────────
    # Load the documents from the specified directory
    documents = load_directory(data_path)
    if n_docs: 
        documents = documents[:n_docs]
    if not documents:
        raise ValueError(f"No documents found in {data_path}. Please check the path and try again.")
    
    # ── Configuration ────────────────────────────────────────────────────────────────────
    # Initialize the RunConfig with the specified metrics and test set size
    run_config = RunConfig(max_workers = 1, max_retries = 15, max_wait = 60)
    
    # ── Generator Models ─────────────────────────────────────────────────────────────────
    # Initialize the LLM and Embeddings wrappers with the specified models and API key  
    generator_llm = LangchainLLMWrapper(ChatGroq(
            model=settings.GROQ_MODEL_NAME, 
            api_key=settings.GROQ_API_KEY, 
            temperature=0, 
            max_tokens=1024))
    
    generator_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=settings.HF_EMBEDDING_MODEL_NAME, 
            cache_folder="models",  
            model_kwargs={"device": "cuda"}))
    
    
    # ── Generate Test Set ───────────────────────────────────────────────────────────────
    # Generate the test set using the loaded documents and the initialized LLM and Embeddings wrappers
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings)
    
    
    log.info("Generating testset (size=%d)…", test_set_size)
    t0 = time.perf_counter()
    
    dataset = generator.generate_with_langchain_docs(
        documents,
        testset_size=test_set_size,
        run_config=run_config,
        raise_exceptions=True   
    )
    
    elapsed = time.perf_counter() - t0
    log.info("Generation finished in %.1fs", elapsed)
    
    
    # ── inspect ───────────────────────────────────────────────────────────────
    df: pd.DataFrame = dataset.to_pandas()
    print(df.head())
    
    # ── save ──────────────────────────────────────────────────────────────────
    # Check if the output directory exists, and create it if it doesn't
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{out_dir}/testset.csv", index=False)
    log.info("Saved → %s", out_dir)
    return df

if __name__ == "__main__":
    generate_test_set(n_docs=5, test_set_size=5)
