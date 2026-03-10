# Day 3 Design: Query Engine with Prompt Versioning

**Date:** 2026-03-02
**Status:** Approved

## Context

Days 1-2 are complete: config management, structured JSON logging, and the
ingestion pipeline with MLflow tracking are all implemented with code quality
improvements (type hints, config validation, FILE_EXTENSIONS support).

Day 3 delivers the query side of the pipeline: a versioned system prompt and a
query engine that retrieves from the ChromaDB index built by ingest.py.

## Files

| File | Action |
|---|---|
| `src/prompts/system_v1.txt` | Create |
| `src/query.py` | Create |
| `tests/test_query.py` | Create |

## Design Decisions

### configure_settings() duplication
Keep a `configure_settings()` in `query.py` separate from the one in
`ingest.py`. The API (Day 5-6) will be the right place to centralise this when
the index is loaded once at startup. No premature abstraction.

### Quality standard
Apply the same standards as Day 2: full type hints, clear error messages on
missing files/config. No Ollama health-check pre-flight (medium-priority item
deferred per PLAN.md roadmap).

## src/prompts/system_v1.txt

Verbatim from PLAN.md:

```
You are a helpful assistant that answers questions based on the provided context
from a personal knowledge base.

Rules:
- Only answer based on the provided context. If the context doesn't contain
  enough information, say so.
- Quote or reference specific notes where possible.
- Be concise and direct.
- If the question is ambiguous, ask for clarification rather than guessing.
```

## src/query.py

Four public functions:

### load_system_prompt(version: str | None) -> str
- Resolves version to `SYSTEM_PROMPT_VERSION` from config if not provided
- Constructs path: `<module_dir>/prompts/system_{version}.txt`
- Raises `FileNotFoundError` with a clear message if the file doesn't exist
- Returns the stripped prompt text

### configure_settings() -> None
- Sets `Settings.embed_model` to `OllamaEmbedding` with configured model/URL
- Sets `Settings.llm` to `Ollama` with configured model/URL/timeout
- Identical pattern to `ingest.configure_settings()`

### load_index() -> VectorStoreIndex
- Opens `chromadb.PersistentClient` at `CHROMA_PERSIST_DIR`
- Calls `get_collection("obsidian_vault")`
- Returns `VectorStoreIndex.from_vector_store(ChromaVectorStore(...))`

### query(question, top_k, prompt_version) -> dict[str, Any]
- `top_k` defaults to config `TOP_K`, `prompt_version` defaults to config
- Calls `load_index()` then `load_system_prompt()`
- Builds query engine: `index.as_query_engine(similarity_top_k, response_mode, system_prompt)`
- Times the query, extracts source nodes
- Returns:
  ```python
  {
      "answer": str,
      "sources": [{"file": str, "score": float, "text_preview": str}],
      "metrics": {
          "latency_seconds": float,
          "num_sources": int,
          "top_score": float | None,
          "prompt_version": str,
          "model": str,
      }
  }
  ```
- Logs: question (first 100 chars), latency, num_sources, top_score, prompt_version

## tests/test_query.py

Four test classes:

### TestLoadSystemPrompt
- Happy path: creates a temp file, asserts correct content returned
- Missing file: raises `FileNotFoundError` with informative message
- No version passed: falls back to `SYSTEM_PROMPT_VERSION` from config

### TestConfigureSettings
- Embed model assigned to `Settings.embed_model`
- LLM assigned to `Settings.llm`
- `OLLAMA_REQUEST_TIMEOUT` passed to Ollama constructor

### TestLoadIndex
- Calls `get_collection("obsidian_vault")` on ChromaDB client
- Returns a `VectorStoreIndex`

### TestQuery
- Sources mapped correctly (file, score, text_preview)
- Latency key present in metrics
- prompt_version and model present in metrics
- top_score is None when no source nodes
