"""
Query engine with prompt versioning and per-query metrics.
"""
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import (
    CHROMA_PERSIST_DIR,
    EMBED_MODEL,
    LLM_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_REQUEST_TIMEOUT,
    RESPONSE_MODE,
    SYSTEM_PROMPT_VERSION,
    TOP_K,
)
from logging_config import get_logger

logger = get_logger(__name__)


def load_system_prompt(version: Optional[str] = None) -> str:
    """Load a versioned system prompt from file."""
    version = version or SYSTEM_PROMPT_VERSION
    prompts_dir = Path(__file__).parent / "prompts"
    # Reject any version string that contains path separators or traversal sequences
    if os.sep in version or "/" in version or "\\" in version or ".." in version:
        raise ValueError(f"Invalid prompt version: {version!r}")
    prompt_path = (prompts_dir / f"system_{version}.txt").resolve()
    if not prompt_path.is_relative_to(prompts_dir.resolve()):
        raise ValueError(f"Invalid prompt version: {version!r}")
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"System prompt version '{version}' not found at {prompt_path}. "
            f"Create src/prompts/system_{version}.txt to use this version."
        )
    with open(prompt_path, encoding="utf-8") as f:
        return f.read().strip()


def configure_settings() -> None:
    """Configure LlamaIndex global settings."""
    Settings.embed_model = OllamaEmbedding(
        model_name=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    Settings.llm = Ollama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        request_timeout=OLLAMA_REQUEST_TIMEOUT,
    )


def load_index() -> VectorStoreIndex:
    """Load existing ChromaDB index."""
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    chroma_collection = chroma_client.get_collection("obsidian_vault")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return VectorStoreIndex.from_vector_store(vector_store)


def query(
    question: str,
    top_k: Optional[int] = None,
    prompt_version: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Query the index. Returns response, sources, and metrics.
    """
    configure_settings()
    top_k = top_k if top_k is not None else TOP_K
    system_prompt = load_system_prompt(prompt_version)

    index = load_index()
    query_engine = index.as_query_engine(
        similarity_top_k=top_k,
        response_mode=RESPONSE_MODE,
        system_prompt=system_prompt,
    )

    start = time.time()
    response = query_engine.query(question)
    latency = time.time() - start

    sources: List[Dict[str, Any]] = [
        {
            "file": node.metadata.get("file_name", "unknown"),
            "score": node.score,
            "text_preview": node.text[:200],
        }
        for node in response.source_nodes
    ]

    logger.info(
        "Query completed",
        extra={"extra_data": {
            "question": question[:100],
            "latency_seconds": round(latency, 2),
            "num_sources": len(sources),
            "top_score": sources[0]["score"] if sources else None,
            "prompt_version": prompt_version or SYSTEM_PROMPT_VERSION,
        }},
    )

    return {
        "answer": str(response),
        "sources": sources,
        "metrics": {
            "latency_seconds": round(latency, 2),
            "num_sources": len(sources),
            "top_score": sources[0]["score"] if sources else None,
            "prompt_version": prompt_version or SYSTEM_PROMPT_VERSION,
            "model": LLM_MODEL,
        },
    }


if __name__ == "__main__":
    import sys

    user_question = " ".join(sys.argv[1:]) or "What topics are in my notes?"  # pylint: disable=invalid-name
    result = query(user_question)
    print(f"\nAnswer: {result['answer']}\n")
    print(f"Metrics: {result['metrics']}")
    print("\nSources:")
    for i, source in enumerate(result["sources"], 1):
        print(f"  {i}. {source['file']} (score: {source['score']:.3f})")
