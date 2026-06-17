"""Generates answers from retrieved document chunks using a Mistral LLM, returning the answer text and source metadata."""

import os
from typing import List
from langchain_core.documents import Document
from langchain_mistralai import ChatMistralAI

from app.core import settings

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

def generate_answer(query: str, documents: List[Document]) -> dict:
    """ Answer the user query based on the retrieved documents using a large language model."""
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


#  Test Block Only
# from app.retrieval.vectorstore import init_vectorStore, add_chunks
# from app.ingestion.loaders import load_directory
# from app.ingestion.chunking import split_documents
# from app.retrieval.multiquery import multiquery_retrieve

# vector_store = init_vectorStore()
# query = "what is backpropagation?"
# retrieved = multiquery_retrieve(query, vector_store)

# for doc in retrieved:
#     print(f"Chunk ID: {doc.metadata.get('chunk_id')}, Content: {doc.page_content[:100]}, Score: {doc.metadata.get('score')}")


# result = generate_answer(query, retrieved)
# print(f"Result: {result}\n\n\n\n\n")


# print(f"\n\n\n ## Answer: {result['answer']}\n\n\ndetails of sources used:")
# for doc in result['sources']:
#     print(
#         f"source name: {doc['source_name']}\n"
#         f"start page: {doc['start_page']}\n"
#         f"chunk headers: {doc['chunk_headers']}\n"
#         f"chunk score: {doc['chunk_score']}\n"
#         f"chunk content: {doc['chunk_content'][:100]}\n"
#     )
