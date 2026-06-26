"""HyDE retrieval — Hypothetical Document Embeddings.

Instead of embedding the raw user question (which is phrased like a *question*
and rarely matches the wording of the *answer* text stored in the index), HyDE
first asks an LLM to write a short hypothetical passage that *reads like a source
document answering the query*. That passage is then embedded and used for the
similarity search, so query and stored chunks live in the same "answer-shaped"
semantic space.

This module performs RETRIEVAL ONLY — generating the final grounded answer is the
job of :func:`app.generation.answerer.generate_answer`. Keeping answering out of
here lets the retrieval strategies (baseline / multi-query / HyDE) stay
interchangeable in the evaluation harness.
"""

import os
from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_mistralai import ChatMistralAI
from langchain_chroma import Chroma
from app.core import settings

# ── LLM setup ───────────────────────────────────────────────────────────────
os.environ["MISTRAL_API_KEY"] = settings.MISTRAL_API_KEY

llm = ChatMistralAI(model=settings.MEDIUM_MODEL_NAME,
                    temperature=0)


# ── Prompt template & chain ─────────────────────────────────────────────────
# Hyde Instructions
hyde_instructions = """\
You are a hypothetical document generator for a RAG retrieval pipeline.

Given the user query, write a single hypothetical passage that a real source document
would contain if it directly answered the query.

Rules:
- Write as if you are the source document, not as someone answering a question.
- Match the tone and register of a factual, informative text (no "As an AI…", no preamble).
- The passage must be semantically rich — use domain-appropriate vocabulary the real
documents are likely to share.
- Factual accuracy is not required; plausibility and lexical proximity to real documents is.
- Length: 3-5 sentences. No bullet points, no headers.

Output format:
- The passage only. No preamble, no label, no trailing text.

User query: {query}
"""
prompt_hyde = ChatPromptTemplate.from_template(hyde_instructions, output_parser=StrOutputParser())

# ── Retrieval ───────────────────────────────────────────────────────────────
def hyde_retrieve(query: str, vector_store: Chroma,
                n_results: int = 5) -> List[Document]:
    """  
    Generate a hypothetical answer passage, then retrieve real chunks similar to it (HyDE).

    Args:
        - query (str): The user query.
        - vector_store (Chroma): The vector store to retrieve documents from.
        - n_results (int, optional): The number of relevant documents to retrieve for each generated query. Defaults to 5.

    Returns:
        List[Document]: A list of retrieved documents.
    """ 
    # Formatted instructions for the HyDE prompt
    formatted_instructions = prompt_hyde.format(query=query)
    
    # Generate a hypothetical document passage using the LLM
    hypothetical_document = llm.invoke(formatted_instructions)
    
    # Strip and clean the content of the hypothetical document
    response = hypothetical_document.content.strip()
    
    retrieved_docs = vector_store.similarity_search_with_score(response, k=n_results)
    docs = []
    
    for doc, score in retrieved_docs:
        similarity_score = 1 - score
        doc.metadata["score"] = similarity_score      # match multiquery's score attachment
        docs.append(doc)
    return docs