"""Generates answers from retrieved document chunks using a Mistral LLM, returning the answer text and source metadata."""

import os
from typing import List
from langchain_core.documents import Document
from langchain_mistralai import ChatMistralAI

from app.core import settings

# ── Configuration & prompt ──────────────────────────────────────────────────
# Set the Mistral AI API key in the environment variables
os.environ["MISTRAL_API_KEY"] = settings.MISTRAL_API_KEY

LLM_instructions = """\
You are a retrieval-augmented generation assistant. Answer the user's question \
using ONLY the information present in the retrieved documents below.

Rules:
- If the documents contain the answer, respond concisely and directly.
- If the documents are insufficient or irrelevant, reply: \
"I don't have enough information in the retrieved documents to answer this."
- Do not add facts, assumptions, or prior knowledge beyond what the documents state.
- If multiple documents conflict, surface the conflict briefly rather than picking one silently.

Retrieved documents:
{documents}

User question: {query}
"""

# Initialize the large language model for answering queries
llm = ChatMistralAI(
        model=settings.LARGE_MODEL_NAME,
        temperature=0
    )

# ── Answer generation ───────────────────────────────────────────────────────
def generate_answer(query: str, documents: List[Document]) -> dict:
    """Answer the user query, grounded strictly in the retrieved documents.

    The documents are injected into a system prompt that forbids using outside
    knowledge, so the model either answers from the provided context or says it
    cannot. Retrieval is the caller's responsibility — this function only
    generates and assembles citations.

    Args:
        query (str): The user's question.
        documents (List[Document]): The retrieved chunks to ground the answer in.

    Returns:
        dict: ``{"answer": str, "sources": List[dict]}`` where each source carries
        ``source_name``, ``start_page``, ``chunk_content``, ``chunk_headers`` and
        ``chunk_score`` for citation.
    """
    formatted_instructions = LLM_instructions.format(documents = documents, query = query)
    
    # Call the LLM with the formatted instructions 
    response = llm.invoke(formatted_instructions)
    
    return {
        "answer": response.content,
        "sources": [
            {
                "source_name":doc.metadata.get("document_id", "Unknown") , 
                "start_page": doc.metadata.get("page", "Unknown"),
                "chunk_content": doc.page_content,
                "chunk_headers": doc.metadata.get("chunk_headers", "Unknown"),
                "chunk_score": doc.metadata.get("score", "Unknown")
            } for doc in documents]
        }
