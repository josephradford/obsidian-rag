"""Unit tests for query module."""
from unittest.mock import MagicMock, mock_open, patch

import pytest


class TestLoadSystemPrompt:
    """Tests for load_system_prompt()."""

    def test_returns_stripped_prompt_text(self, tmp_path, monkeypatch):
        """load_system_prompt returns stripped content of the prompt file."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "system_v1.txt").write_text("  You are helpful.  ", encoding="utf-8")

        # Redirect Path(__file__).parent to tmp_path so the function finds our prompts dir
        import query as query_mod
        monkeypatch.setattr(query_mod, "__file__", str(tmp_path / "query.py"))

        from query import load_system_prompt
        result = load_system_prompt("v1")
        assert result == "You are helpful."

    def test_raises_file_not_found_for_missing_version(self, tmp_path, monkeypatch):
        """load_system_prompt raises FileNotFoundError for unknown version."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()

        import query as query_mod
        monkeypatch.setattr(query_mod, "__file__", str(tmp_path / "query.py"))

        from query import load_system_prompt
        with pytest.raises(FileNotFoundError, match="v99"):
            load_system_prompt("v99")

    def test_uses_config_version_when_none_passed(self, tmp_path, monkeypatch):
        """load_system_prompt uses SYSTEM_PROMPT_VERSION from config when version is None."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()

        import query as query_mod
        config_version = query_mod.SYSTEM_PROMPT_VERSION
        (prompts_dir / f"system_{config_version}.txt").write_text("default", encoding="utf-8")
        monkeypatch.setattr(query_mod, "__file__", str(tmp_path / "query.py"))

        from query import load_system_prompt
        result = load_system_prompt(None)
        assert result == "default"

    def test_rejects_path_traversal_version(self):
        """load_system_prompt raises ValueError for path-traversal version strings."""
        from query import load_system_prompt
        with pytest.raises(ValueError, match="Invalid prompt version"):
            load_system_prompt("../../etc/passwd")


class TestConfigureSettings:
    """Tests for configure_settings()."""

    @patch("query.OllamaEmbedding")
    @patch("query.Ollama")
    @patch("query.Settings")
    def test_sets_embed_model(self, mock_settings, mock_ollama, mock_embedding):
        """configure_settings assigns an OllamaEmbedding to Settings.embed_model."""
        from query import configure_settings
        configure_settings()
        assert mock_settings.embed_model == mock_embedding.return_value

    @patch("query.OllamaEmbedding")
    @patch("query.Ollama")
    @patch("query.Settings")
    def test_sets_llm(self, mock_settings, mock_ollama, mock_embedding):
        """configure_settings assigns an Ollama LLM to Settings.llm."""
        from query import configure_settings
        configure_settings()
        assert mock_settings.llm == mock_ollama.return_value

    @patch("query.OllamaEmbedding")
    @patch("query.Ollama")
    @patch("query.Settings")
    def test_uses_configured_timeout(self, mock_settings, mock_ollama, mock_embedding):
        """configure_settings passes OLLAMA_REQUEST_TIMEOUT to the LLM."""
        from query import configure_settings, OLLAMA_REQUEST_TIMEOUT
        configure_settings()
        _, kwargs = mock_ollama.call_args
        assert kwargs["request_timeout"] == OLLAMA_REQUEST_TIMEOUT


class TestLoadIndex:
    """Tests for load_index()."""

    @patch("query.VectorStoreIndex")
    @patch("query.ChromaVectorStore")
    @patch("query.chromadb.PersistentClient")
    def test_gets_obsidian_vault_collection(
        self, mock_client_cls, mock_store, mock_index_cls
    ):
        """load_index fetches the 'obsidian_vault' collection from ChromaDB."""
        from query import load_index
        client = MagicMock()
        mock_client_cls.return_value = client

        load_index()

        client.get_collection.assert_called_once_with("obsidian_vault")

    @patch("query.VectorStoreIndex")
    @patch("query.ChromaVectorStore")
    @patch("query.chromadb.PersistentClient")
    def test_returns_vector_store_index(
        self, mock_client_cls, mock_store, mock_index_cls
    ):
        """load_index returns the VectorStoreIndex built from the ChromaDB store."""
        from query import load_index
        mock_client_cls.return_value = MagicMock()

        result = load_index()

        assert result == mock_index_cls.from_vector_store.return_value


class TestQuery:
    """Tests for query()."""

    def _make_source_node(self, file_name="test.md", score=0.9, text="x" * 300):
        node = MagicMock()
        node.metadata = {"file_name": file_name}
        node.score = score
        node.text = text
        return node

    @patch("query.configure_settings")  # outermost -> last param
    @patch("query.load_index")
    @patch("query.load_system_prompt")
    def test_sources_mapped_correctly(self, mock_prompt, mock_load_index, mock_configure):
        """query() maps source nodes to file/score/text_preview dicts."""
        from query import query
        mock_prompt.return_value = "prompt text"
        node = self._make_source_node(file_name="notes.md", score=0.85)
        mock_response = MagicMock()
        mock_response.source_nodes = [node]
        mock_load_index.return_value.as_query_engine.return_value.query.return_value = (
            mock_response
        )

        result = query("test question")

        assert result["sources"][0]["file"] == "notes.md"
        assert result["sources"][0]["score"] == 0.85
        assert len(result["sources"][0]["text_preview"]) <= 200

    @patch("query.configure_settings")  # outermost -> last param
    @patch("query.load_index")
    @patch("query.load_system_prompt")
    def test_metrics_contain_latency(self, mock_prompt, mock_load_index, mock_configure):
        """query() metrics dict includes a non-negative latency_seconds."""
        from query import query
        mock_prompt.return_value = "prompt"
        mock_response = MagicMock()
        mock_response.source_nodes = []
        mock_load_index.return_value.as_query_engine.return_value.query.return_value = (
            mock_response
        )

        result = query("test")

        assert "latency_seconds" in result["metrics"]
        assert result["metrics"]["latency_seconds"] >= 0

    @patch("query.configure_settings")  # outermost -> last param
    @patch("query.load_index")
    @patch("query.load_system_prompt")
    def test_top_score_is_none_when_no_sources(self, mock_prompt, mock_load_index, mock_configure):
        """query() sets top_score to None when there are no source nodes."""
        from query import query
        mock_prompt.return_value = "prompt"
        mock_response = MagicMock()
        mock_response.source_nodes = []
        mock_load_index.return_value.as_query_engine.return_value.query.return_value = (
            mock_response
        )

        result = query("test")

        assert result["metrics"]["top_score"] is None

    @patch("query.configure_settings")  # outermost -> last param
    @patch("query.load_index")
    @patch("query.load_system_prompt")
    def test_metrics_contain_prompt_version_and_model(
        self, mock_prompt, mock_load_index, mock_configure
    ):
        """query() metrics include prompt_version and model from config."""
        from query import query, SYSTEM_PROMPT_VERSION, LLM_MODEL
        mock_prompt.return_value = "prompt"
        mock_response = MagicMock()
        mock_response.source_nodes = []
        mock_load_index.return_value.as_query_engine.return_value.query.return_value = (
            mock_response
        )

        result = query("test")

        assert result["metrics"]["prompt_version"] == SYSTEM_PROMPT_VERSION
        assert result["metrics"]["model"] == LLM_MODEL

    @patch("query.configure_settings")  # outermost -> last param
    @patch("query.load_index")
    @patch("query.load_system_prompt")
    def test_answer_is_string(self, mock_prompt, mock_load_index, mock_configure):
        """query() answer field is a string."""
        from query import query
        mock_prompt.return_value = "prompt"
        mock_response = MagicMock()
        mock_response.source_nodes = []
        mock_load_index.return_value.as_query_engine.return_value.query.return_value = (
            mock_response
        )

        result = query("test")

        assert isinstance(result["answer"], str)

    @patch("query.configure_settings")  # outermost -> last param
    @patch("query.load_index")
    @patch("query.load_system_prompt")
    def test_explicit_prompt_version_appears_in_metrics(
        self, mock_prompt, mock_load_index, mock_configure
    ):
        """query() uses the explicit prompt_version when provided."""
        from query import query
        mock_prompt.return_value = "prompt"
        mock_response = MagicMock()
        mock_response.source_nodes = []
        mock_load_index.return_value.as_query_engine.return_value.query.return_value = (
            mock_response
        )

        result = query("test", prompt_version="v2")

        assert result["metrics"]["prompt_version"] == "v2"
