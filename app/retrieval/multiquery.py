"""Multi-query retrieval — expands a user query into N variants via LLM, retrieves from ChromaDB, and deduplicates results above the similarity threshold."""

from typing import List
from langchain_core.documents import Document
from langchain_mistralai import ChatMistralAI
from langchain_chroma import Chroma
from app.retrieval.vectorstore import init_vectorStore
from app.core import settings
import os




mqr_instructions = """\
You are a query expansion module for a RAG retrieval pipeline.

Given the user query, generate exactly {n_variants} alternative phrasings of it.

Rules:
- Preserve the original intent precisely — meaning must not drift.
- Each rephrasing must be lexically or syntactically distinct from the others \
(vary word choice, sentence structure, or perspective — not just word order).
- Do not add, remove, or assume information beyond what the original query states.

Output format:
- Numbered list, one query per line (1. ... 2. ... etc.)
- No preamble, no explanation, no trailing text.

User query: {query}
"""


# Set the Mistral AI API key in the environment variables
os.environ["MISTRAL_API_KEY"] = settings.MISTRAL_API_KEY
llm = ChatMistralAI(model=settings.MEDIUM_MODEL_NAME,
                    temperature=0)


def multi_query_retrieve(query: str, vector_store: Chroma, 
                        n_variants: int = 3, n_results: int = 5) -> List[Document]:
    """  
    Generate multiple semantically similar queries based on the user query and retrieve relevant documents from the knowledge base.

    Args:
        query (str): The user query.
        vector_store (Chroma): The vector store to retrieve documents from.
        n_variants (int, optional): The number of semantically similar queries to generate. Defaults to 3.
        n_results (int, optional): The number of relevant documents to retrieve for each generated query. Defaults to 5.

    Returns:
        List[Document]: A list of retrieved documents.
    """
    formatted_instructions = mqr_instructions.format(n_variants = n_variants, query = query)
    
    # Generate multiple variants of the user query
    response = llm.invoke(formatted_instructions)
    queries = response.content.strip().split("\n")
    last_queries = [query.split(". ", 1)[1].strip() for query in queries]
    
    list_of_retrieved_docs = []
    seen_ids = set()
    for idx, variant in enumerate(last_queries):
        print(f"The {idx + 1} Prompt: {variant}")
        retrieved_docs = vector_store.similarity_search_with_score(variant, k=n_results)
        for doc, score in retrieved_docs: 
            chunk_id = doc.metadata.get("chunk_id")
            similarity_score = 1 - score
            if chunk_id not in seen_ids and similarity_score >= settings.SIMILARITY_THRESHOLD:
                # Filter out documents with a score greater than the threshold in the config.py
                seen_ids.add(chunk_id)
                doc.metadata["score"] = similarity_score
                list_of_retrieved_docs.append(doc)
                
    return list_of_retrieved_docs

# # Test Block Only
# from app.retrieval.vectorstore import init_vectorStore, add_chunks
# from app.ingestion.loaders import load_directory
# from app.ingestion.chunking import split_documents

# vector_store = init_vectorStore()
# #documents = load_directory()
# docs = load_directory()
# print(f"Loaded {len(docs)} documents")
# chunks = split_documents(docs)
# add_chunks(vector_store, chunks)

# results = multi_query_retrieve("what is backpropagation?", vector_store)
# for doc in results:
#     print(f"Chunk ID: {doc.metadata.get('chunk_id')}, Content: {doc.page_content[:100]}...")