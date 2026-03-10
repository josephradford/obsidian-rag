"""Unit tests for the FastAPI API module."""
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


def make_app_client():
    """
    Return a TestClient with the lifespan mocked out.

    Patches configure_settings and load_index so no real Ollama/ChromaDB is
    needed.  The client is used as a context manager so the lifespan runs.
    """
    with patch("api.configure_settings"), patch("api.load_index") as mock_load_index:
        mock_load_index.return_value = MagicMock()
        from api import app
        return TestClient(app)


def make_source_node(file_name="notes.md", score=0.85, text="x" * 300):
    """Build a mock LlamaIndex source node."""
    node = MagicMock()
    node.metadata = {"file_name": file_name}
    node.score = score
    node.text = text
    return node


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    """Tests for GET /health."""

    def test_returns_healthy_status(self):
        """GET /health returns 200 with status 'healthy'."""
        with patch("api.configure_settings"), patch("api.load_index") as mock_load_index:
            mock_load_index.return_value = MagicMock()
            from api import app
            with TestClient(app) as client:
                response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_index_loaded_true_after_startup(self):
        """GET /health reports index_loaded: true after lifespan startup."""
        with patch("api.configure_settings"), patch("api.load_index") as mock_load_index:
            mock_load_index.return_value = MagicMock()
            from api import app
            with TestClient(app) as client:
                response = client.get("/health")
        assert response.json()["index_loaded"] is True


# ---------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    """Tests for GET /metrics."""

    def test_returns_zero_totals_on_fresh_start(self):
        """GET /metrics returns zero counts when no queries have been made."""
        import api as api_module
        # Reset metrics store to a clean state
        api_module.metrics_store.update(
            {"total_queries": 0, "total_latency_seconds": 0.0, "errors": 0}
        )
        with patch("api.configure_settings"), patch("api.load_index") as mock_load_index:
            mock_load_index.return_value = MagicMock()
            from api import app
            with TestClient(app) as client:
                response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert data["total_queries"] == 0
        assert data["errors"] == 0

    def test_average_latency_zero_when_no_queries(self):
        """GET /metrics returns 0 average latency when no queries made."""
        import api as api_module
        api_module.metrics_store.update(
            {"total_queries": 0, "total_latency_seconds": 0.0, "errors": 0}
        )
        with patch("api.configure_settings"), patch("api.load_index") as mock_load_index:
            mock_load_index.return_value = MagicMock()
            from api import app
            with TestClient(app) as client:
                response = client.get("/metrics")
        assert response.json()["average_latency_seconds"] == 0.0


# ---------------------------------------------------------------------------
# /query
# ---------------------------------------------------------------------------

class TestQueryEndpoint:
    """Tests for POST /query."""

    def _make_mock_response(self, source_nodes=None, answer="Test answer"):
        """Build a mock LlamaIndex response."""
        mock_response = MagicMock()
        mock_response.source_nodes = source_nodes or []
        mock_response.__str__ = lambda self: answer
        return mock_response

    def test_returns_200_with_valid_question(self):
        """POST /query returns 200 for a valid question."""
        with patch("api.configure_settings"), \
             patch("api.load_index") as mock_load_index, \
             patch("api.load_system_prompt", return_value="prompt"):
            mock_engine = MagicMock()
            mock_engine.query.return_value = self._make_mock_response()
            mock_load_index.return_value.as_query_engine.return_value = mock_engine
            from api import app
            with TestClient(app) as client:
                response = client.post("/query", json={"question": "What are my notes about?"})
        assert response.status_code == 200

    def test_response_contains_answer_sources_metrics(self):
        """POST /query response body contains answer, sources, and metrics keys."""
        with patch("api.configure_settings"), \
             patch("api.load_index") as mock_load_index, \
             patch("api.load_system_prompt", return_value="prompt"):
            mock_engine = MagicMock()
            mock_engine.query.return_value = self._make_mock_response()
            mock_load_index.return_value.as_query_engine.return_value = mock_engine
            from api import app
            with TestClient(app) as client:
                response = client.post("/query", json={"question": "test"})
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "metrics" in data

    def test_sources_mapped_correctly(self):
        """POST /query maps source nodes to file/score/text_preview."""
        node = make_source_node(file_name="notes.md", score=0.9)
        with patch("api.configure_settings"), \
             patch("api.load_index") as mock_load_index, \
             patch("api.load_system_prompt", return_value="prompt"):
            mock_engine = MagicMock()
            mock_engine.query.return_value = self._make_mock_response(source_nodes=[node])
            mock_load_index.return_value.as_query_engine.return_value = mock_engine
            from api import app
            with TestClient(app) as client:
                response = client.post("/query", json={"question": "test"})
        source = response.json()["sources"][0]
        assert source["file"] == "notes.md"
        assert source["score"] == pytest.approx(0.9)
        assert len(source["text_preview"]) <= 200

    def test_metrics_contain_expected_keys(self):
        """POST /query metrics include latency, num_sources, top_score, prompt_version, model."""
        with patch("api.configure_settings"), \
             patch("api.load_index") as mock_load_index, \
             patch("api.load_system_prompt", return_value="prompt"):
            mock_engine = MagicMock()
            mock_engine.query.return_value = self._make_mock_response()
            mock_load_index.return_value.as_query_engine.return_value = mock_engine
            from api import app
            with TestClient(app) as client:
                response = client.post("/query", json={"question": "test"})
        metrics = response.json()["metrics"]
        assert "latency_seconds" in metrics
        assert "num_sources" in metrics
        assert "top_score" in metrics
        assert "prompt_version" in metrics
        assert "model" in metrics

    def test_top_score_none_when_no_sources(self):
        """POST /query sets top_score to null when there are no sources."""
        with patch("api.configure_settings"), \
             patch("api.load_index") as mock_load_index, \
             patch("api.load_system_prompt", return_value="prompt"):
            mock_engine = MagicMock()
            mock_engine.query.return_value = self._make_mock_response(source_nodes=[])
            mock_load_index.return_value.as_query_engine.return_value = mock_engine
            from api import app
            with TestClient(app) as client:
                response = client.post("/query", json={"question": "test"})
        assert response.json()["metrics"]["top_score"] is None

    def test_empty_question_rejected(self):
        """POST /query returns 422 for an empty question string."""
        with patch("api.configure_settings"), patch("api.load_index") as mock_load_index:
            mock_load_index.return_value = MagicMock()
            from api import app
            with TestClient(app) as client:
                response = client.post("/query", json={"question": ""})
        assert response.status_code == 422

    def test_query_engine_error_returns_500(self):
        """POST /query returns 500 and increments error counter when query engine raises."""
        import api as api_module
        api_module.metrics_store.update(
            {"total_queries": 0, "total_latency_seconds": 0.0, "errors": 0}
        )
        with patch("api.configure_settings"), \
             patch("api.load_index") as mock_load_index, \
             patch("api.load_system_prompt", return_value="prompt"):
            mock_engine = MagicMock()
            mock_engine.query.side_effect = RuntimeError("Ollama unavailable")
            mock_load_index.return_value.as_query_engine.return_value = mock_engine
            from api import app
            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.post("/query", json={"question": "test"})
        assert response.status_code == 500
        assert api_module.metrics_store["errors"] == 1

    def test_metrics_incremented_after_successful_query(self):
        """POST /query increments total_queries in the metrics store."""
        import api as api_module
        api_module.metrics_store.update(
            {"total_queries": 0, "total_latency_seconds": 0.0, "errors": 0}
        )
        with patch("api.configure_settings"), \
             patch("api.load_index") as mock_load_index, \
             patch("api.load_system_prompt", return_value="prompt"):
            mock_engine = MagicMock()
            mock_engine.query.return_value = self._make_mock_response()
            mock_load_index.return_value.as_query_engine.return_value = mock_engine
            from api import app
            with TestClient(app) as client:
                client.post("/query", json={"question": "test"})
        assert api_module.metrics_store["total_queries"] == 1


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

class TestLifespan:
    """Tests for app startup behaviour."""

    def test_configure_settings_called_on_startup(self):
        """App lifespan calls configure_settings() during startup."""
        with patch("api.configure_settings") as mock_configure, \
             patch("api.load_index") as mock_load_index:
            mock_load_index.return_value = MagicMock()
            from api import app
            with TestClient(app):
                pass
        mock_configure.assert_called_once()

    def test_load_index_called_on_startup(self):
        """App lifespan calls load_index() during startup."""
        with patch("api.configure_settings"), \
             patch("api.load_index") as mock_load_index:
            mock_load_index.return_value = MagicMock()
            from api import app
            with TestClient(app):
                pass
        mock_load_index.assert_called_once()
