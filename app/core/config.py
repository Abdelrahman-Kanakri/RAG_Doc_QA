from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    """Application settings."""
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra='ignore')
    
    # Mistral AI API KEY
    MISTRAL_API_KEY: str = Field(
        description = "Mistral AI API key used in development period, and environment.", )
    
    # Model names
    LARGE_MODEL_NAME: str = Field(
        description = "Name of the Large Model used in the application.", )

    MEDIUM_MODEL_NAME: str = Field(
        description = "Name of the Medium Model used in the application.",)

    SMALL_MODEL_NAME: str = Field(
        description = "Name of the Small Model used in the application.", )
    
    EMBEDDING_MODEL_NAME: str = Field(
        description = "Name of the embedding model used in the application.",)
    
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