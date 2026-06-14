from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    """Application settings."""
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra='ignore')
    
    # Mistral AI API KEY
    mistral_api_key: str = Field(
        description = "Mistral AI API key used in development period, and environment.", )
    
    # Model names
    large_model_name: str = Field(
        description = "Name of the Large Model used in the application.", )

    medium_model_name: str = Field(
        description = "Name of the Medium Model used in the application.",)

    small_model_name: str = Field(
        description = "Name of the Small Model used in the application.", )
    
    embedding_model_name: str = Field(
        description = "Name of the embedding model used in the application.",)
    
    # LANGSMITH CONFIGURATIONS
    langsmith_api_key: str = Field(
        description = "LangSmith API key used for tracing and monitoring.",
    )
    langsmith_endpoint: str = Field(
        description = "Endpoint for the LangSmith API.",
    )
    langsmith_tracing: str = Field(
        description = "Flag to enable/disable LangSmith tracing.",
    )
    langsmith_project: str = Field(
        description = "Project name for LangSmith.",
    )
    
    # Chroma DB configurations
    vector_database: str = Field(
        description = "Path to the Chroma DB.",
    )
    similarity_threshold: float = Field(
        description = "Threshold for similarity search.",
        default = 0.7
    )
    chunk_size: int = Field(
        description = "Size of each chunk for processing.",
        default = 1000
    )
    chunk_overlap_size: int = Field(
        description = "Overlap size between chunks.",
        default = 200
    )
    collection_name: str = Field(
        description = "Name of the collection in the vector database.",
        default = "documents"
    )

settings = Settings()