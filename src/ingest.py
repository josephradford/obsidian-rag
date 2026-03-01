"""
Ingest Obsidian vault into ChromaDB vector store.

Every ingestion run is tracked in MLflow with parameters and metrics.
"""
import time
from typing import List, Tuple
import mlflow
import chromadb
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    Settings,
    Document,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from config import (
    OBSIDIAN_VAULT_PATH,
    EMBED_MODEL,
    LLM_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHROMA_PERSIST_DIR,
    OLLAMA_BASE_URL,
    OLLAMA_REQUEST_TIMEOUT,
    MLFLOW_TRACKING_URI,
    FILE_EXTENSIONS,
    validate_config,
    get_pipeline_params,
)
from logging_config import get_logger

logger = get_logger(__name__)


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


def load_documents(vault_path: str) -> List[Document]:
    """Load all files from the Obsidian vault based on configured extensions."""
    reader = SimpleDirectoryReader(
        input_dir=vault_path,
        recursive=True,
        required_exts=FILE_EXTENSIONS,
        exclude=[".obsidian", ".trash", "templates"],
    )
    documents = reader.load_data()
    logger.info(
        f"Loaded {len(documents)} documents",
        extra={"extra_data": {
            "num_documents": len(documents), 
            "vault_path": vault_path,
            "file_extensions": FILE_EXTENSIONS,
        }},
    )
    return documents


def create_index(documents: List[Document]) -> Tuple[VectorStoreIndex, int]:
    """Chunk documents and store embeddings in ChromaDB."""
    node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # Delete existing collection for clean re-ingestion
    try:
        chroma_client.delete_collection("obsidian_vault")
        logger.info("Deleted existing ChromaDB collection")
    except ValueError as e:
        # Collection doesn't exist - this is fine on first run
        logger.info("No existing collection to delete", extra={"extra_data": {"reason": str(e)}})
    except Exception as e:
        logger.warning("Failed to delete existing collection", extra={"extra_data": {"error": str(e)}})
        # Continue anyway but log the issue

    chroma_collection = chroma_client.create_collection("obsidian_vault")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[node_parser],
        show_progress=True,
    )

    num_chunks = len(chroma_collection.get()["ids"])
    logger.info(
        f"Index created with {num_chunks} chunks",
        extra={"extra_data": {"num_chunks": num_chunks, "chunk_size": CHUNK_SIZE}},
    )
    return index, num_chunks


def ingest() -> VectorStoreIndex:
    """Main ingestion pipeline — tracked in MLflow."""
    # Validate configuration before starting
    validate_config()
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("ingestion")

    configure_settings()

    with mlflow.start_run(run_name=f"ingest-chunk{CHUNK_SIZE}-overlap{CHUNK_OVERLAP}"):
        mlflow.log_params(get_pipeline_params())

        start = time.time()
        documents = load_documents(OBSIDIAN_VAULT_PATH)
        load_time = time.time() - start

        start = time.time()
        index, num_chunks = create_index(documents)
        index_time = time.time() - start

        mlflow.log_metrics({
            "num_documents": len(documents),
            "num_chunks": num_chunks,
            "chunks_per_document": num_chunks / max(len(documents), 1),
            "load_time_seconds": load_time,
            "index_time_seconds": index_time,
            "total_time_seconds": load_time + index_time,
        })

        logger.info(
            "Ingestion complete",
            extra={"extra_data": {
                "num_documents": len(documents),
                "num_chunks": num_chunks,
                "total_time": round(load_time + index_time, 1),
            }},
        )

    return index


if __name__ == "__main__":
    ingest()