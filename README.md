# Obsidian RAG Pipeline

A Retrieval-Augmented Generation system that queries a personal Obsidian knowledge base using local LLMs, built with MLOps practices from day one.

## Project Status

🚧 **Currently in development** - Setting up the foundational infrastructure for Phase 1.

## Why This Project

This project demonstrates practical AI/ML engineering skills:
- **RAG pipeline** — document ingestion, chunking, retrieval, and generation
- **MLOps discipline** — experiment tracking (MLflow), prompt versioning, config-driven pipeline, structured logging
- **Evaluation** — quantitative metrics and automated evaluation harness
- **Governance foundations** — data provenance, model cards, Architecture Decision Records (ADRs)

The focus is on building with production-quality practices, not just getting a RAG pipeline working.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Obsidian    │────▶│  Ingestion   │────▶│  ChromaDB   │
│  Vault (.md) │     │  Pipeline    │     │  (vectors)  │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
┌─────────────┐     ┌──────────────┐            │
│  User Query  │────▶│  FastAPI     │◀───────────┘
│  (HTTP)      │     │  Server      │
└─────────────┘     └──────┬───────┘
                           │
                    ┌──────▼───────┐     ┌──────────────┐
                    │  Ollama      │     │  MLflow      │
                    │  (LLM)       │     │  (tracking)  │
                    └──────────────┘     └──────────────┘
```

### Tech Stack

| Component | Tool | Why |
|-----------|------|-----|
| Document source | Obsidian vault (Markdown) | Real personal data for quality evaluation |
| Ingestion & retrieval | LlamaIndex | Better RAG defaults than LangChain |
| Embedding model | nomic-embed-text (Ollama) | Runs locally, no API costs |
| Vector database | ChromaDB | Embedded mode, fast iteration |
| LLM | Llama 3.2 3B (Ollama) | Runs on 8GB Apple Silicon |
| API layer | FastAPI | Industry standard for model serving |
| Experiment tracking | MLflow | Tracks params/metrics/artifacts |
| Logging | Python logging (JSON) | Structured, ready for CloudWatch |

## Setup

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai) installed
- macOS with Apple Silicon (8GB RAM minimum)

### Installation

```bash
# Clone the repository
git clone https://github.com/josephradford/obsidian-rag.git
cd obsidian-rag

# Run setup script
./scripts/setup.sh

# Configure your Obsidian vault path
cp .env.example .env
# Edit .env and set OBSIDIAN_VAULT_PATH
```

### Quick Start (Coming Soon)

Once implementation is complete:

```bash
# Activate virtual environment
source .venv/bin/activate

# Ingest your Obsidian vault
python src/ingest.py

# Query via CLI
python src/query.py "What topics are in my notes?"

# Start API server
uvicorn src.api:app --port 8000
# Open http://localhost:8000/docs
```

## MLOps Practices

| Practice | Implementation |
|----------|---------------|
| Experiment tracking | MLflow — every ingestion and eval run logged with params and metrics |
| Prompt versioning | System prompts as versioned files, switchable via config |
| Config management | All pipeline parameters in .env, logged to MLflow per run |
| Structured logging | JSON-formatted logs with metrics (latency, sources, scores) |
| Evaluation | Automated eval script: source recall, answer coverage, pass rate |
| Observability | /metrics endpoint tracking query count, avg latency, errors |
| Data provenance | Documented in docs/data_provenance.md |
| Architecture decisions | ADR format in docs/architecture.md |
| Reproducibility | Pinned deps, setup script, .env.example |

## Development Roadmap

### Phase 1: Local RAG Pipeline with MLOps Practices (Current)
- [x] Project setup and prerequisites
- [ ] Config management and structured logging
- [ ] Ingestion pipeline with experiment tracking
- [ ] Query engine with prompt versioning
- [ ] FastAPI server with metrics
- [ ] Evaluation framework foundation
- [ ] Architecture documentation (ADRs)

### Phase 2: AWS Deployment
- Docker containerization
- EC2 deployment with GPU
- CloudWatch integration (logging and metrics)
- Infrastructure as Code (Terraform/CDK)
- Production-grade inference serving (vLLM)

### Phase 3: Advanced Evaluation
- Fine-tuning experiments
- LLM-as-judge evaluation
- Weights & Biases integration
- Comprehensive evaluation framework

### Phase 4: AI Governance
- AIAF (AI Assurance Framework) mapping
- Model validation reports
- Risk register
- SOUP (Software of Unknown Provenance) documentation

## Project Structure

```
obsidian-rag/
├── src/
│   ├── config.py              # Centralized pipeline configuration
│   ├── logging_config.py      # Structured JSON logging
│   ├── ingest.py              # Vault ingestion pipeline
│   ├── query.py               # Query engine with prompt management
│   ├── api.py                 # FastAPI server
│   └── prompts/               # Versioned system prompts
├── tests/                     # Unit tests
├── eval/                      # Evaluation dataset and scripts
├── docs/                      # Architecture docs, ADRs, governance
├── scripts/                   # Setup and utility scripts
├── .env.example              # Configuration template
└── requirements.txt          # Pinned dependencies
```

## Contributing

This is a personal portfolio project, but suggestions and feedback are welcome through issues.

## License

MIT License - See LICENSE file for details.

## Acknowledgments

Built following MLOps best practices inspired by IEC 62304 (medical device software) and production ML systems.
