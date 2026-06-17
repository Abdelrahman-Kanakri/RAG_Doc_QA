"""Retrieval pipeline — vector store management and multi-query retrieval."""

# Simple rule to avoid circular imports in Python packages:

# Modules inside a package must never import from their own package's __init__.

# So inside app/retrieval/, always use the direct path:


# from app.retrieval.vectorstore import init_vectorStore  # ✓
# from app.retrieval import init_vectorStore 

from .multiquery import multi_query_retrieve
from .vectorstore import init_vectorStore, add_chunks, view_collection


__all__ = ["init_vectorStore", "add_chunks", "view_collection", "multi_query_retrieve"]
