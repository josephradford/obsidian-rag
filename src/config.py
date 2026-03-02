"""
Centralised configuration from environment variables.

MLOps principle: pipeline parameters are config, not code.
Every parameter that affects output should be here and logged to MLflow.
"""
import os
from typing import List, Dict, Any

from dotenv import load_dotenv

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
OLLAMA_REQUEST_TIMEOUT = float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "120.0"))


def validate_config() -> None:
    """Validate required configuration is present and valid."""
    errors = []
    
    # Re-read environment variables for validation (enables testing)
    vault_path = os.getenv("OBSIDIAN_VAULT_PATH")
    chunk_size = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))
    top_k = int(os.getenv("TOP_K", "5"))
    timeout = float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "120.0"))
    file_exts = os.getenv("FILE_EXTENSIONS", ".md,.pdf").split(",")
    
    if not vault_path:
        errors.append("OBSIDIAN_VAULT_PATH is required")
    elif not os.path.exists(vault_path):
        errors.append(f"Vault path does not exist: {vault_path}")
    
    if chunk_size <= 0:
        errors.append("CHUNK_SIZE must be positive")
    
    if chunk_overlap < 0:
        errors.append("CHUNK_OVERLAP must be non-negative")
    
    if chunk_overlap >= chunk_size:
        errors.append("CHUNK_OVERLAP must be less than CHUNK_SIZE")
    
    if top_k <= 0:
        errors.append("TOP_K must be positive")
    
    if timeout <= 0:
        errors.append("OLLAMA_REQUEST_TIMEOUT must be positive")
    
    if not file_exts or all(not ext.strip() for ext in file_exts):
        errors.append("FILE_EXTENSIONS must contain at least one valid extension")
    
    # Validate file extensions format
    for ext in file_exts:
        ext = ext.strip()
        if ext and not ext.startswith('.'):
            errors.append(f"File extension must start with '.': {ext}")
    
    if errors:
        raise ValueError(f"Configuration errors: {'; '.join(errors)}")


def get_pipeline_params() -> Dict[str, Any]:
    """Return all pipeline parameters for MLflow logging."""
    return {
        "llm_model": LLM_MODEL,
        "embed_model": EMBED_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "top_k": TOP_K,
        "response_mode": RESPONSE_MODE,
        "system_prompt_version": SYSTEM_PROMPT_VERSION,
        "file_extensions": ",".join(FILE_EXTENSIONS),
        "ollama_request_timeout": OLLAMA_REQUEST_TIMEOUT,
    }
