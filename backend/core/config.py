"""
Application configuration.

Values are read from environment variables or a .env file.
Defaults are set for local development. Override in production
by setting environment variables before launching the server.

Usage:
    from backend.core.config import settings
    print(settings.ollama_base_url)
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application
    app_version: str = "0.1.0"
    debug: bool = False

    # CORS — comma-separated list of allowed origins
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    ollama_timeout_seconds: int = 120

    # ClimateBERT
    climatebert_model_name: str = "climatebert/distilroberta-base-climate-detector"

    # ChromaDB
    chroma_persist_directory: str = "./data/chroma"

    # Document parsing
    max_document_size_mb: int = 50
    supported_file_types: list[str] = ["pdf", "docx", "txt"]

    # Pipeline
    claim_extraction_chunk_size: int = 1500   # characters per chunk sent to LLM
    claim_extraction_chunk_overlap: int = 200  # overlap between chunks

    # Max Ollama requests in flight at once. Bounds the client only — Ollama
    # serialises per loaded model unless OLLAMA_NUM_PARALLEL is also raised.
    max_concurrent_requests: int = 4


# Single shared instance imported throughout the app
settings = Settings()
