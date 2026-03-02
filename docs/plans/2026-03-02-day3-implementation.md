# Day 3: Query Engine with Prompt Versioning — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement `src/query.py` and `src/prompts/system_v1.txt` — the query side of the RAG pipeline — with unit tests using TDD.

**Architecture:** `query.py` mirrors the structure of `ingest.py`: separate functions for settings, index loading, and the main operation. `load_system_prompt()` reads versioned prompt files from `src/prompts/`, enabling A/B testing via config. `query()` loads the existing ChromaDB index and returns a structured dict with answer, sources, and metrics.

**Tech Stack:** LlamaIndex, ChromaDB, Ollama, Python unittest.mock, pytest

---

## Context

- `src/config.py`: provides `SYSTEM_PROMPT_VERSION`, `TOP_K`, `RESPONSE_MODE`, `OLLAMA_REQUEST_TIMEOUT`, `EMBED_MODEL`, `LLM_MODEL`, `OLLAMA_BASE_URL`, `CHROMA_PERSIST_DIR`
- `src/ingest.py`: reference for `configure_settings()` pattern
- `tests/test_ingest.py`: reference for test style (class-based, `@patch` decorators, descriptive docstrings)
- `pyproject.toml`: `pythonpath = ["src"]` — tests import `from query import ...` directly

---

## Task 1: Create system_v1.txt

**Files:**
- Create: `src/prompts/system_v1.txt`

**Step 1: Create the prompts directory and file**

```bash
mkdir -p src/prompts
```

Create `src/prompts/system_v1.txt` with this exact content:

```
You are a helpful assistant that answers questions based on the provided context from a personal knowledge base.

Rules:
- Only answer based on the provided context. If the context doesn't contain enough information, say so.
- Quote or reference specific notes where possible.
- Be concise and direct.
- If the question is ambiguous, ask for clarification rather than guessing.
```

**Step 2: Verify the file**

```bash
cat src/prompts/system_v1.txt
```

Expected: prompt text printed without errors.

---

## Task 2: Create query.py skeleton

**Files:**
- Create: `src/query.py`

**Step 1: Create the skeleton with all imports and function stubs**

Create `src/query.py`:

```python
"""
Query engine with prompt versioning and per-query metrics.
"""
import os
import time
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
    raise NotImplementedError


def configure_settings() -> None:
    """Configure LlamaIndex global settings."""
    raise NotImplementedError


def load_index() -> VectorStoreIndex:
    """Load existing ChromaDB index."""
    raise NotImplementedError


def query(
    question: str,
    top_k: Optional[int] = None,
    prompt_version: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Query the index. Returns response, sources, and metrics.
    """
    raise NotImplementedError


if __name__ == "__main__":
    import sys

    configure_settings()
    question = " ".join(sys.argv[1:]) or "What topics are in my notes?"
    result = query(question)
    print(f"\nAnswer: {result['answer']}\n")
    print(f"Metrics: {result['metrics']}")
    print("\nSources:")
    for i, source in enumerate(result["sources"], 1):
        print(f"  {i}. {source['file']} (score: {source['score']:.3f})")
```

**Step 2: Verify it imports cleanly**

```bash
cd src && python -c "import query" && echo "OK"
```

Expected: `OK` (no import errors; `NotImplementedError` only raises on call).

---

## Task 3: Write all tests (all should fail)

**Files:**
- Create: `tests/test_query.py`

**Step 1: Create the test file**

Create `tests/test_query.py`:

```python
"""Unit tests for query module."""
from unittest.mock import MagicMock, mock_open, patch

import pytest


class TestLoadSystemPrompt:
    """Tests for load_system_prompt()."""

    def test_returns_stripped_prompt_text(self):
        """load_system_prompt returns stripped content of the prompt file."""
        with patch("query.os.path.exists", return_value=True), \
             patch("builtins.open", mock_open(read_data="  You are helpful.  ")):
            from query import load_system_prompt
            result = load_system_prompt("v1")
        assert result == "You are helpful."

    def test_raises_file_not_found_for_missing_version(self):
        """load_system_prompt raises FileNotFoundError for unknown version."""
        with patch("query.os.path.exists", return_value=False):
            from query import load_system_prompt
            with pytest.raises(FileNotFoundError, match="v99"):
                load_system_prompt("v99")

    def test_uses_config_version_when_none_passed(self):
        """load_system_prompt uses SYSTEM_PROMPT_VERSION from config when version is None."""
        from query import load_system_prompt
        with patch("query.os.path.exists", return_value=True), \
             patch("builtins.open", mock_open(read_data="default")):
            result = load_system_prompt(None)
        assert result == "default"


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

    @patch("query.load_index")
    @patch("query.load_system_prompt")
    def test_sources_mapped_correctly(self, mock_prompt, mock_load_index):
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

    @patch("query.load_index")
    @patch("query.load_system_prompt")
    def test_metrics_contain_latency(self, mock_prompt, mock_load_index):
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

    @patch("query.load_index")
    @patch("query.load_system_prompt")
    def test_top_score_is_none_when_no_sources(self, mock_prompt, mock_load_index):
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

    @patch("query.load_index")
    @patch("query.load_system_prompt")
    def test_metrics_contain_prompt_version_and_model(
        self, mock_prompt, mock_load_index
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

    @patch("query.load_index")
    @patch("query.load_system_prompt")
    def test_answer_is_string(self, mock_prompt, mock_load_index):
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
```

**Step 2: Run the tests — all must fail**

```bash
pytest tests/test_query.py -v
```

Expected: all 14 tests FAIL with `NotImplementedError`.

---

## Task 4: Implement load_system_prompt()

**Files:**
- Modify: `src/query.py` — replace the `load_system_prompt` stub

**Step 1: Replace the stub with the implementation**

Replace the `load_system_prompt` body in `src/query.py`:

```python
def load_system_prompt(version: Optional[str] = None) -> str:
    """Load a versioned system prompt from file."""
    version = version or SYSTEM_PROMPT_VERSION
    prompt_path = os.path.join(
        os.path.dirname(__file__), "prompts", f"system_{version}.txt"
    )
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(
            f"System prompt version '{version}' not found at {prompt_path}. "
            f"Create src/prompts/system_{version}.txt to use this version."
        )
    with open(prompt_path) as f:
        return f.read().strip()
```

**Step 2: Run load_system_prompt tests**

```bash
pytest tests/test_query.py::TestLoadSystemPrompt -v
```

Expected: 3 PASSED.

---

## Task 5: Implement configure_settings()

**Files:**
- Modify: `src/query.py` — replace the `configure_settings` stub

**Step 1: Replace the stub**

```python
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
```

**Step 2: Run configure_settings tests**

```bash
pytest tests/test_query.py::TestConfigureSettings -v
```

Expected: 3 PASSED.

---

## Task 6: Implement load_index()

**Files:**
- Modify: `src/query.py` — replace the `load_index` stub

**Step 1: Replace the stub**

```python
def load_index() -> VectorStoreIndex:
    """Load existing ChromaDB index."""
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    chroma_collection = chroma_client.get_collection("obsidian_vault")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return VectorStoreIndex.from_vector_store(vector_store)
```

**Step 2: Run load_index tests**

```bash
pytest tests/test_query.py::TestLoadIndex -v
```

Expected: 2 PASSED.

---

## Task 7: Implement query()

**Files:**
- Modify: `src/query.py` — replace the `query` stub

**Step 1: Replace the stub**

```python
def query(
    question: str,
    top_k: Optional[int] = None,
    prompt_version: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Query the index. Returns response, sources, and metrics.
    """
    top_k = top_k or TOP_K
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
```

**Step 2: Run all query() tests**

```bash
pytest tests/test_query.py::TestQuery -v
```

Expected: 6 PASSED.

---

## Task 8: Run full test suite and commit

**Step 1: Run all tests**

```bash
pytest --tb=short
```

Expected: all tests PASS (existing tests for config, logging, ingest plus the 14 new query tests).

**Step 2: Run linting**

```bash
pylint src/query.py
```

Expected: score >= 8.0. Fix any issues before committing.

**Step 3: Commit**

```bash
git add src/prompts/system_v1.txt src/query.py tests/test_query.py
git commit -m "Day 3: query engine with prompt versioning and unit tests

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 9: Code review

**Step 1: Invoke the code review skill**

Use `superpowers:requesting-code-review` on the new files.

Focus areas:
- Does `query.py` follow the same quality standards as `ingest.py`?
- Are edge cases covered (empty sources, missing file_name metadata)?
- Is the test coverage complete for the public interface?
- Any security issues with file path construction?

---

## Task 10: Update CLAUDE.md if needed

**Step 1: Review CLAUDE.md for gaps**

Check whether these sections need updating:
- **Project Structure**: add `src/prompts/` directory
- **Key Commands**: add `python src/query.py "question"` command
- **Important Implementation Details**: add `### Query Engine (src/query.py)` section (it has a stub, verify it's accurate)
- **Troubleshooting**: check the existing query-related entries match the implementation

**Step 2: Apply any necessary updates**

Only update if content is missing or inaccurate. The `query.py` section already exists in CLAUDE.md under "Important Implementation Details" — verify it matches the actual implementation.

**Step 3: Commit if changed**

```bash
git add CLAUDE.md
git commit -m "Update CLAUDE.md to reflect Day 3 implementation

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```
