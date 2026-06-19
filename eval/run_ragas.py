"""RAGAS evaluation harness — RAG Triad scoring across retrieval strategies (Phase 3).

Runs a controlled experiment: the only thing that varies is the retrieval
strategy (baseline similarity search vs. multi-query vs. HyDE); everything else
is held constant. For each strategy it builds a RAGAS ``EvaluationDataset`` from
``eval/dataset_test/testset.csv``, scores it on four metrics, and writes the
averaged comparison table to ``eval/results/results.csv``.

Roles (kept strictly separate to avoid confounding the result):
    * System under test — Mistral, used for both retrieval-side LLM work
      (multi-query expansion, HyDE passage generation) and final answer generation.
    * Judge            — Groq (``settings.GROQ_MODEL_NAME``); a *different*,
      httpx-based model so the judge never grades its own output. (Gemini is
      avoided here — its gRPC async client breaks inside RAGAS' event loop.)
    * Judge embeddings — local ``BAAI/bge-m3``, needed by the relevancy/context
      metrics.

Metrics (the RAG Triad, plus context recall):
    Faithfulness, ResponseRelevancy, LLMContextPrecisionWithReference, LLMContextRecall.

Environment guards (``OPENSSL_CONF`` / ``HF_HUB_OFFLINE``) must be set before
pandas/ragas import — see ``eval/generate_testset.py`` for the rationale.

Run from the project root: ``python -m eval.run_ragas``.
"""

import os
os.environ.setdefault("OPENSSL_CONF", "/dev/null")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pandas as pd
from ragas import evaluate, EvaluationDataset
from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithReference, LLMContextRecall
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from pathlib import Path

from app.retrieval import hyde_retrieve
from app.retrieval import init_vectorStore, multi_query_retrieve 
from app.generation.answerer import generate_answer
from app.core import settings


os.environ["GROQ_API_KEY"] = settings.GROQ_API_KEY

# ── Retrieval strategies (the variable under test) ──────────────────────────
def baseline(query, vectorStore):
    """Baseline retrieval strategy using a single query.
    args: 
        query: str, the user query
        vectorStore: the vector store object
        k=5: int, the number of relevant documents to retrieve
    """
    
    return vectorStore.similarity_search(query, k=5)

def multiquery(query, vectorStore):
    """Multi-query retrieval strategy using multiple queries.
    args: 
        query: str, the user query
        vectorStore: the vector store object
        n_variants: int, the number of semantically similar queries to generate
        n_results: int, the number of relevant documents to retrieve for each generated query
    """
    
    return multi_query_retrieve(query, vectorStore, n_variants=3, n_results=5)

def hyde(query, vectorStore):
    """HyDE retrieval strategy using a hypothetical-document generator.
    args:
        query: str, the user query
        vectorStore: the vector store object
        n_results=5: int, the number of relevant documents to retrieve
    """

    return hyde_retrieve(query, vectorStore, n_results=5)

strategies = { 
    "baseline": baseline,
    "multiquery": multiquery,
    "hyde": hyde
}

# ── Dataset builder ─────────────────────────────────────────────────────────
# Building the Dataset generation function
def build_dataset(df, retrieve, vectorStore):
    """Build a RAGAS EvaluationDataset by running one retrieval strategy over the test set.

    For each test-set row it retrieves contexts with ``retrieve``, generates an
    answer with the app's ``generate_answer``, and assembles the four fields RAGAS
    needs per sample.

    args:
        df: pd.DataFrame, the test set — uses the 'user_input' (question) and
            'reference' (gold answer) columns produced by generate_testset.py
        retrieve: function (query, vectorStore) -> List[Document], the strategy under test
        vectorStore: the vector store object
    returns:
        EvaluationDataset with per-row fields: user_input, retrieved_contexts,
        response, reference.
    """
    samples = []
    for _, row in df.iterrows():
        docs = retrieve(row["user_input"], vectorStore)
        result = generate_answer(row["user_input"], docs)
        samples.append({
            "user_input": row["user_input"],
            "retrieved_contexts": [doc.page_content for doc in docs],
            "response": result["answer"],
            "reference": row["reference"]
        })
    return EvaluationDataset.from_list(samples)

# ── Main ────────────────────────────────────────────────────────────────────
def main():
    """Score every retrieval strategy on the test set and save the comparison table.

    Loads the test set, builds the judge (Groq LLM + local bge-m3 embeddings),
    then for each strategy builds its evaluation dataset, runs ``evaluate`` over
    the four RAG-Triad metrics, averages the numeric scores, and writes one row
    per strategy to ``eval/results/results.csv``.
    """
    df = pd.read_csv("eval/dataset_test/testset.csv")
    vectorStore = init_vectorStore()
    judge_llm = LangchainLLMWrapper(ChatGroq(
            model=settings.GROQ_MODEL_NAME, 
            api_key=settings.GROQ_API_KEY, 
            temperature=0, 
            max_tokens=1024))
    
    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=settings.HF_EMBEDDING_MODEL_NAME, 
            cache_folder="models",  
            model_kwargs={"device": "cuda"}))
    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall()
    ]
    cfg = RunConfig(max_workers=1, max_retries=15, max_wait=60)
    row = []
    
    for name, strategy in strategies.items():
        dataset = build_dataset(df, strategy, vectorStore)
        results = evaluate(dataset = dataset, metrics = metrics, llm = judge_llm, embeddings = judge_embeddings,  run_config =  cfg)
        scores = results.to_pandas().select_dtypes(include="number").mean().to_dict()
        row.append({
            "strategy": name,
            **scores
        })
    
    Path("eval/results/").mkdir(exist_ok=True)
    pd.DataFrame(row).to_csv("eval/results/results.csv", index=False)

if __name__ == "__main__":
    main()