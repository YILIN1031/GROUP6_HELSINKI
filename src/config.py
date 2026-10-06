import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()


@dataclass
class Config:
    """Simple configuration object reading environment variables."""

    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model_name: str = os.getenv("MODEL_NAME", "llama3.2")

    # RAG settings
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    documents_path: str = os.getenv("DOCUMENTS_PATH", "data/documents")
    vector_store_path: str = os.getenv("VECTOR_STORE_PATH", "data/vector_store")
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "4"))
    rag_chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
    rag_chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
    # Provisional initial value chosen before evaluation; must be tuned on the
    # evaluation dataset. Not an empirically validated threshold.
    rag_min_score: float = float(os.getenv("RAG_MIN_SCORE", "0.5"))


# Instantiate global configuration object
config = Config()
