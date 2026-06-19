"""Pydantic response models for the health, ingest, and query endpoints."""

from pydantic import BaseModel, Field
from typing import List

# ── Health ──────────────────────────────────────────────────────────────────
class GetHealth(BaseModel):
    """ Model for health check response """
    status: str = Field(..., description="Health status of the application")
    vector_store: str = Field(..., description="Connection status of the vector store")

# ── Ingest ──────────────────────────────────────────────────────────────────
class ResponseIngest(BaseModel):
    """ Model for response after document ingestion """
    document_id: str = Field(..., description="Unique identifier for the ingested document")
    chunks: int = Field(..., description="Number of chunks created from the document")
    message: str = Field(..., description="Status message regarding the ingestion process")

# ── Query ───────────────────────────────────────────────────────────────────
class Source(BaseModel):
    """ Model for source information used in query response """
    source_name: str = Field(..., description="Name of the source document")
    start_page: str = Field(..., description="Starting page of the chunk")
    chunk_content: str = Field(..., description="Content of the chunk")
    chunk_headers: str = Field(..., description="Headers of the chunk")
    chunk_score: float = Field(..., description="Relevance score of the chunk")

class ResponseQuery(BaseModel):
    answer: str = Field(..., description="Answer to the query")
    sources: List[Source] = Field(..., description="List of sources used to generate the answer")
