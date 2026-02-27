"""
Centralised configuration from environment variables.

MLOps principle: pipeline parameters are config, not code.
Every parameter that affects output should be here and logged to MLflow.
"""
from dotenv import load_dotenv
import os

load_dotenv()

# Data
OBSIDIAN_VAULT_PATH = os.getenv("OBSIDIAN_VAULT_PATH")
FILE_EXTENSIONS = os.getenv("FILE_EXTENSIONS", ".md,.pdf").split(",")

# Models
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Retrieval
TOP_K = int(os.getenv("TOP_K", "5"))
RESPONSE_MODE = os.getenv("RESPONSE_MODE", "compact")

# Prompt
SYSTEM_PROMPT_VERSION = os.getenv("SYSTEM_PROMPT_VERSION", "v1")

# Storage
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")

# Observability
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


def get_pipeline_params() -> dict:
    """Return all pipeline parameters for MLflow logging."""
    return {
        "llm_model": LLM_MODEL,
        "embed_model": EMBED_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "top_k": TOP_K,
        "response_mode": RESPONSE_MODE,
        "system_prompt_version": SYSTEM_PROMPT_VERSION,
    }
