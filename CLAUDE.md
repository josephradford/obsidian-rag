# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an Obsidian RAG (Retrieval-Augmented Generation) pipeline that queries a personal Obsidian vault using local LLMs. The project is built with MLOps practices from day one, demonstrating practical AI/ML engineering skills including experiment tracking, prompt versioning, structured logging, and evaluation.

**Current Phase**: Phase 1 - Local RAG Pipeline with MLOps Practices
**Target Completion**: April 2026

## Development Environment

**Platform**: macOS (Apple Silicon, 8GB RAM)
**Python Version**: Python 3 with virtual environment (`.venv`)

### Prerequisites Setup
```bash
# Activate virtual environment
source .venv/bin/activate

# Install Ollama models
ollama pull llama3.2:3b          # Main LLM (~2GB)
ollama pull nomic-embed-text     # Embedding model (~274MB)
```

## Key Commands

### Setup
```bash
# Initial setup (creates venv, installs deps, pulls models)
./scripts/setup.sh

# Activate environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Core Pipeline Operations
```bash
# Ingest Obsidian vault into ChromaDB
python src/ingest.py

# Query via CLI
python src/query.py "Your question here"

# Run FastAPI server
uvicorn src.api:app --reload --port 8000

# View MLflow tracking UI
mlflow ui --port 5001
```

### Evaluation and Experiments
```bash
# Run evaluation script
python eval/evaluate.py

# Run chunking experiments
./scripts/run_experiment.sh
```

### Testing
```bash
# Test API endpoints
curl http://localhost:8000/health
curl http://localhost:8000/metrics
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Your question", "top_k": 5}'

# Interactive API docs
# Open http://localhost:8000/docs
```

## Architecture

### Tech Stack
- **Document Source**: Obsidian vault (Markdown files)
- **Ingestion/Retrieval**: LlamaIndex
- **Embedding Model**: nomic-embed-text (via Ollama)
- **Vector Database**: ChromaDB (embedded, no server)
- **LLM**: Ollama - Llama 3.2 3B
- **API Layer**: FastAPI
- **Experiment Tracking**: MLflow (local)
- **Logging**: Python logging with JSON formatter

### Project Structure
```
obsidian-rag/
├── src/
│   ├── config.py           # Centralized pipeline parameters from .env
│   ├── logging_config.py   # Structured JSON logging setup
│   ├── ingest.py          # Vault ingestion pipeline with MLflow tracking
│   ├── query.py           # Query engine with prompt versioning
│   ├── api.py             # FastAPI server with /query, /health, /metrics
│   └── prompts/           # Versioned system prompts (system_v1.txt, system_v2.txt, etc.)
├── tests/                 # Unit tests
├── eval/
│   ├── test_questions.json # Evaluation dataset
│   └── evaluate.py        # Automated evaluation script
├── docs/
│   ├── architecture.md    # Architecture Decision Records (ADRs)
│   ├── data_provenance.md # Data source documentation
│   ├── model_card.md      # Model card template
│   └── experiment_log.md  # Human-readable experiment notes
└── scripts/
    └── setup.sh          # One-command setup script
```

### Data Flow
1. Obsidian vault (.md files) → Ingestion Pipeline → ChromaDB (vectors)
2. User Query (HTTP) → FastAPI Server → Query Engine
3. Query Engine retrieves from ChromaDB → Sends to Ollama LLM with system prompt
4. All operations tracked in MLflow

## MLOps Principles

### Configuration Management
- ALL pipeline parameters are in `.env` (not `.env.example`)
- All parameters logged to MLflow on every run
- Changing model, chunk size, or prompt version is a config change, not code change
- Use `src/config.py` to access all configuration values

### Experiment Tracking
- Every ingestion run is tracked in MLflow with params and metrics
- Every evaluation run is tracked in MLflow
- Use `mlflow.set_experiment()` to organize runs
- Log parameters with `mlflow.log_params(get_pipeline_params())`
- Log metrics with `mlflow.log_metrics()`

### Prompt Versioning
- System prompts are versioned as text files in `src/prompts/`
- Format: `system_v1.txt`, `system_v2.txt`, etc.
- Version controlled in Git for full diff history
- Switchable via `SYSTEM_PROMPT_VERSION` in `.env`
- MLflow logs which prompt version was used for each run

### Structured Logging
- Use `logging_config.get_logger(__name__)` for all logging
- JSON-formatted logs with timestamp, level, module, message
- Include extra data: `logger.info("message", extra={"extra_data": {...}})`
- Logs include metrics like latency, chunk count, source scores

### Evaluation
- Test dataset in `eval/test_questions.json` with expected sources and answers
- Metrics: source recall, answer coverage, latency, pass rate
- All evaluation runs tracked in MLflow for comparison

## Important Implementation Details

### Ingestion Pipeline (src/ingest.py)
- Deletes and recreates ChromaDB collection on each run (clean re-ingestion)
- Uses `SentenceSplitter` for chunking with configurable size and overlap
- Tracks: num_documents, num_chunks, chunks_per_document, load_time, index_time
- All runs logged to MLflow experiment "ingestion"

### Query Engine (src/query.py)
- Loads system prompt from versioned file based on config
- Returns: answer, sources (with file, score, text_preview), metrics
- Logs: question, latency, num_sources, top_score, prompt_version

### FastAPI Server (src/api.py)
- Loads index once at startup (lifespan context manager)
- Endpoints: `/query` (POST), `/health` (GET), `/metrics` (GET)
- Query request includes: question, top_k, prompt_version
- In-memory metrics tracking (replaced by CloudWatch in Phase 2)

### ChromaDB
- Persistent client at path specified by `CHROMA_PERSIST_DIR`
- Collection name: "obsidian_vault"
- No separate server needed (embedded mode)

## Common Development Patterns

### Running Experiments
1. Modify parameters in `.env` (e.g., CHUNK_SIZE, SYSTEM_PROMPT_VERSION)
2. Run ingestion: `python src/ingest.py`
3. Run evaluation: `python eval/evaluate.py`
4. View results in MLflow: `mlflow ui --port 5001`
5. Document findings in `docs/experiment_log.md`
6. Create ADR in `docs/architecture.md` if making a decision

### Adding a New System Prompt
1. Create `src/prompts/system_vN.txt` with new prompt text
2. Update `SYSTEM_PROMPT_VERSION=vN` in `.env`
3. Run queries or evaluation to test
4. Compare results with previous version in MLflow
5. Document decision in experiment log and ADR

### Modifying Pipeline Parameters
1. Update value in `.env` file
2. Run ingestion to rebuild index with new parameters
3. MLflow automatically logs the new parameters
4. Compare metrics across runs in MLflow UI

## Phase 1 Scope and Constraints

### What's In Scope
- Local RAG pipeline (ingest, retrieve, generate)
- FastAPI server with basic endpoints
- Prompt engineering and versioning
- MLflow experiment tracking
- Structured logging (foundation for CloudWatch)
- Basic evaluation dataset and script
- Started: Architecture Decision Records, data provenance, model card

### What's NOT In Scope (Future Phases)
- Docker containers (Phase 2)
- AWS deployment (Phase 2)
- CloudWatch monitoring (Phase 2)
- Fine-tuning (Phase 3)
- Full evaluation framework with LLM-as-judge (Phase 3)
- Weights & Biases integration (Phase 3)
- Complete governance documentation (Phase 4)
- AIAF mapping, risk register, SOUP (Phase 4)

### Key Constraints
- 8GB RAM limit - close Chrome/Slack/Teams while developing
- Use smaller Llama 3.2 3B model (larger models in Phase 2 on AWS)
- Local-only - no cloud services yet

## Architecture Decisions

Key ADRs are documented in `docs/architecture.md`:
- ADR-001: LlamaIndex over LangChain (better RAG defaults)
- ADR-002: ChromaDB for dev, Qdrant planned for production
- ADR-003: Chunking strategy (determined via experiments)
- ADR-004: System prompt versioning as files (Git tracking, MLflow logging)
- ADR-005: Config-driven pipeline (all params in .env, logged to MLflow)

## Troubleshooting

### Ollama times out during ingestion
Increase `request_timeout` in `configure_settings()` in the relevant file

### Poor retrieval quality
Run chunking experiments via `./scripts/run_experiment.sh` and compare in MLflow

### LLM hallucinating
Strengthen system prompt constraints (try system_v2.txt or create v3)

### Import errors
Ensure virtual environment is activated: `source .venv/bin/activate`

### Slow queries (>30s)
Close memory-intensive apps or reduce `top_k` to 3

### MLflow UI shows no data
Check `MLFLOW_TRACKING_URI` in `.env` matches `./mlruns`

## Configuration Files

### .env (not committed)
Contains all runtime configuration including:
- OBSIDIAN_VAULT_PATH
- LLM_MODEL, EMBED_MODEL
- CHUNK_SIZE, CHUNK_OVERLAP
- TOP_K, RESPONSE_MODE
- SYSTEM_PROMPT_VERSION
- CHROMA_PERSIST_DIR, MLFLOW_TRACKING_URI
- LOG_LEVEL

### .env.example (committed)
Template for `.env` - copy and customize for local setup

## Git Workflow

- Main branch: `main`
- Use branch protection and PR template (to be set up)
- Commit discipline: config management, logging, and tracking from first commit
- Never commit: `.venv/`, `.env`, `data/`, `mlruns/`, `mlartifacts/`, `logs/`
