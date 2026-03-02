"""Unit tests for ingest module."""
import os
from unittest.mock import MagicMock, patch

import pytest


class TestConfigureSettings:
    """Test LlamaIndex settings configuration."""

    @patch("ingest.OllamaEmbedding")
    @patch("ingest.Ollama")
    @patch("ingest.Settings")
    def test_sets_embed_model(self, mock_settings, mock_ollama, mock_embedding):
        """configure_settings assigns an OllamaEmbedding to Settings.embed_model."""
        from ingest import configure_settings
        configure_settings()
        assert mock_settings.embed_model == mock_embedding.return_value

    @patch("ingest.OllamaEmbedding")
    @patch("ingest.Ollama")
    @patch("ingest.Settings")
    def test_sets_llm(self, mock_settings, mock_ollama, mock_embedding):
        """configure_settings assigns an Ollama LLM to Settings.llm."""
        from ingest import configure_settings
        configure_settings()
        assert mock_settings.llm == mock_ollama.return_value

    @patch("ingest.OllamaEmbedding")
    @patch("ingest.Ollama")
    @patch("ingest.Settings")
    def test_uses_configured_timeout(self, mock_settings, mock_ollama, mock_embedding):
        """configure_settings passes OLLAMA_REQUEST_TIMEOUT to the LLM."""
        from ingest import configure_settings, OLLAMA_REQUEST_TIMEOUT
        configure_settings()
        _, kwargs = mock_ollama.call_args
        assert kwargs["request_timeout"] == OLLAMA_REQUEST_TIMEOUT


class TestLoadDocuments:
    """Test document loading from the vault."""

    @patch("ingest.SimpleDirectoryReader")
    def test_returns_documents(self, mock_reader_class):
        """load_documents returns the list of documents from the reader."""
        from ingest import load_documents
        fake_docs = [MagicMock(), MagicMock()]
        mock_reader_class.return_value.load_data.return_value = fake_docs

        result = load_documents("/some/path")

        assert result == fake_docs

    @patch("ingest.SimpleDirectoryReader")
    def test_uses_configured_extensions(self, mock_reader_class):
        """load_documents passes FILE_EXTENSIONS to SimpleDirectoryReader."""
        from ingest import load_documents, FILE_EXTENSIONS
        mock_reader_class.return_value.load_data.return_value = []

        load_documents("/some/path")

        _, kwargs = mock_reader_class.call_args
        assert kwargs["required_exts"] == FILE_EXTENSIONS

    @patch("ingest.SimpleDirectoryReader")
    def test_excludes_obsidian_dirs(self, mock_reader_class):
        """load_documents excludes hidden/system Obsidian directories."""
        from ingest import load_documents
        mock_reader_class.return_value.load_data.return_value = []

        load_documents("/some/path")

        _, kwargs = mock_reader_class.call_args
        assert ".obsidian" in kwargs["exclude"]
        assert ".trash" in kwargs["exclude"]

    @patch("ingest.SimpleDirectoryReader")
    def test_recursive_loading(self, mock_reader_class):
        """load_documents loads the vault recursively."""
        from ingest import load_documents
        mock_reader_class.return_value.load_data.return_value = []

        load_documents("/some/path")

        _, kwargs = mock_reader_class.call_args
        assert kwargs["recursive"] is True


class TestCreateIndex:
    """Test ChromaDB index creation."""

    def _mock_chroma_client(self):
        """Build a minimal mock ChromaDB client."""
        client = MagicMock()
        collection = MagicMock()
        collection.get.return_value = {"ids": ["id1", "id2", "id3"]}
        client.create_collection.return_value = collection
        return client, collection

    @patch("ingest.VectorStoreIndex")
    @patch("ingest.StorageContext")
    @patch("ingest.ChromaVectorStore")
    @patch("ingest.chromadb.PersistentClient")
    def test_returns_index_and_chunk_count(
        self, mock_client_cls, mock_store, mock_ctx, mock_index_cls
    ):
        """create_index returns the index object and the correct chunk count."""
        from ingest import create_index
        client, collection = self._mock_chroma_client()
        mock_client_cls.return_value = client

        index, num_chunks = create_index([MagicMock()])

        assert index == mock_index_cls.from_documents.return_value
        assert num_chunks == 3  # len of collection.get()["ids"]

    @patch("ingest.VectorStoreIndex")
    @patch("ingest.StorageContext")
    @patch("ingest.ChromaVectorStore")
    @patch("ingest.chromadb.PersistentClient")
    def test_deletes_existing_collection(
        self, mock_client_cls, mock_store, mock_ctx, mock_index_cls
    ):
        """create_index deletes any pre-existing collection before creating a new one."""
        from ingest import create_index
        client, _ = self._mock_chroma_client()
        mock_client_cls.return_value = client

        create_index([MagicMock()])

        client.delete_collection.assert_called_once_with("obsidian_vault")

    @patch("ingest.VectorStoreIndex")
    @patch("ingest.StorageContext")
    @patch("ingest.ChromaVectorStore")
    @patch("ingest.chromadb.PersistentClient")
    def test_handles_missing_collection_on_first_run(
        self, mock_client_cls, mock_store, mock_ctx, mock_index_cls
    ):
        """create_index does not raise if the collection doesn't exist yet."""
        from ingest import create_index
        client, _ = self._mock_chroma_client()
        client.delete_collection.side_effect = ValueError("Collection not found")
        mock_client_cls.return_value = client

        # Should complete without raising
        create_index([MagicMock()])
        client.create_collection.assert_called_once_with("obsidian_vault")

    @patch("ingest.VectorStoreIndex")
    @patch("ingest.StorageContext")
    @patch("ingest.ChromaVectorStore")
    @patch("ingest.chromadb.PersistentClient")
    def test_continues_after_unexpected_delete_error(
        self, mock_client_cls, mock_store, mock_ctx, mock_index_cls
    ):
        """create_index logs a warning but continues if delete raises unexpectedly."""
        from ingest import create_index
        client, _ = self._mock_chroma_client()
        client.delete_collection.side_effect = RuntimeError("Unexpected error")
        mock_client_cls.return_value = client

        # Should complete without raising
        create_index([MagicMock()])
        client.create_collection.assert_called_once_with("obsidian_vault")


class TestIngest:
    """Test the top-level ingest() orchestration."""

    @patch("ingest.validate_config")
    @patch("ingest.configure_settings")
    @patch("ingest.load_documents")
    @patch("ingest.create_index")
    @patch("ingest.mlflow")
    def test_calls_validate_config_first(
        self, mock_mlflow, mock_create, mock_load, mock_configure, mock_validate
    ):
        """ingest() validates configuration before doing any work."""
        from ingest import ingest

        mock_load.return_value = [MagicMock()]
        mock_create.return_value = (MagicMock(), 5)
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=None)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        ingest()

        mock_validate.assert_called_once()

    @patch("ingest.validate_config")
    @patch("ingest.configure_settings")
    @patch("ingest.load_documents")
    @patch("ingest.create_index")
    @patch("ingest.mlflow")
    def test_returns_index(
        self, mock_mlflow, mock_create, mock_load, mock_configure, mock_validate
    ):
        """ingest() returns the VectorStoreIndex produced by create_index."""
        from ingest import ingest

        fake_index = MagicMock()
        mock_load.return_value = [MagicMock()]
        mock_create.return_value = (fake_index, 5)
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=None)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        result = ingest()

        assert result == fake_index

    @patch("ingest.validate_config")
    @patch("ingest.configure_settings")
    @patch("ingest.load_documents")
    @patch("ingest.create_index")
    @patch("ingest.mlflow")
    def test_logs_metrics_to_mlflow(
        self, mock_mlflow, mock_create, mock_load, mock_configure, mock_validate
    ):
        """ingest() logs num_documents, num_chunks and timing metrics to MLflow."""
        from ingest import ingest

        mock_load.return_value = [MagicMock(), MagicMock()]
        mock_create.return_value = (MagicMock(), 10)
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=None)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        ingest()

        logged = mock_mlflow.log_metrics.call_args[0][0]
        assert logged["num_documents"] == 2
        assert logged["num_chunks"] == 10
        assert "load_time_seconds" in logged
        assert "index_time_seconds" in logged
        assert "total_time_seconds" in logged

    @patch("ingest.validate_config", side_effect=ValueError("bad config"))
    def test_propagates_config_validation_error(self, mock_validate):
        """ingest() propagates ValueError raised by validate_config."""
        from ingest import ingest
        with pytest.raises(ValueError, match="bad config"):
            ingest()
