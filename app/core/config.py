"""Application settings loaded from .env via pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import os 

class Settings(BaseSettings):
    """Typed application settings, loaded from ``.env`` at import time.

    Fields without a default (the API keys and model names) are *required* —
    pydantic raises a ``ValidationError`` on startup if any is missing from the
    environment, so the app fails loudly rather than running half-configured.
    Fields with a default (thresholds, chunk sizes, collection name) may be
    overridden via ``.env`` but are safe to omit. The module-level ``settings``
    singleton below is what the rest of the app imports.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra='ignore')
    
    # ── Provider API Keys ─────────────────────────────────────────────────────────────
    # Mistral AI API KEY
    MISTRAL_API_KEY: str = Field(
        description = "Mistral AI API key used in development period, and environment.", )
    
    # Groq API KEY
    GROQ_API_KEY: str = Field(
        description = "Groq API key used in development period, and environment.", )
    
    # HF API KEY
    HF_API_KEY: str = Field(
        description = "Hugging Face API key used in development period, and environment.", )
    
    # Google API KEY
    GOOGLE_API_KEY: str = Field(
        description = "Google API key used in development period, and environment.", )

    # ── Models Names ─────────────────────────────────────────────────────────────
    # Model names
    
    # Mistral AI Model Names
    LARGE_MODEL_NAME: str = Field(
        description = "Name of the Large Model used in the application.", )

    MEDIUM_MODEL_NAME: str = Field(
        description = "Name of the Medium Model used in the application.",)

    SMALL_MODEL_NAME: str = Field(
        description = "Name of the Small Model used in the application.", )
    
    # Embedding Model Name
    EMBEDDING_MODEL_NAME: str = Field(
        description = "Name of the embedding model used in the application.",)
    
    # Groq Model Name
    GROQ_MODEL_NAME: str = Field(
        description = "Name of the Groq model used in the application.",)
    
    # HF Embedding Model Name
    HF_EMBEDDING_MODEL_NAME : str = Field(
        description = "Name of the Hugging Face embedding model used in the application.",)
    
    # Google Model Name
    GOOGLE_MODEL_NAME: str = Field(
        description = "Name of the Google model used in the application.",)
    
    # ── Langsmith Observations ─────────────────────────────────────────────────────────────
    # LANGSMITH CONFIGURATIONS
    LANGSMITH_API_KEY: str = Field(
        description = "LangSmith API key used for tracing and monitoring.",
    )
    LANGSMITH_ENDPOINT: str = Field(
        description = "Endpoint for the LangSmith API.",
    )
    LANGSMITH_TRACING: str = Field(
        description = "Flag to enable/disable LangSmith tracing.",
    )
    LANGSMITH_PROJECT: str = Field(
        description = "Project name for LangSmith.",
    )
    
    # ── DB configurations ─────────────────────────────────────────────────────────────
    # Chroma DB configurations
    VECTOR_DATABASE_PATH: str = Field(
        description = "Path to the Chroma DB.",
    )
    SIMILARITY_THRESHOLD: float = Field(
        description = "Threshold for similarity search.",
        default = 0.65
    )
    CHUNK_SIZE: int = Field(
        description = "Size of each chunk for processing.",
        default = 1000
    )
    CHUNK_OVERLAP: int = Field(
        description = "Overlap size between chunks.",
        default = 200
    )
    COLLECTION_NAME: str = Field(
        description = "Name of the collection in the vector database.",
        default = "documents"
    )

settings = Settings()
os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
os.environ["LANGSMITH_ENDPOINT"] = settings.LANGSMITH_ENDPOINT
os.environ["LANGSMITH_TRACING"] = settings.LANGSMITH_TRACING
os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT
