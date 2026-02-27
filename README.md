# Obsidian RAG Pipeline

A RAG system for querying your Obsidian vault using local LLMs with MLOps practices.

## Installation

**Prerequisites:**
- Python 3.9+
- [Ollama](https://ollama.ai)

**Setup:**

```bash
git clone https://github.com/josephradford/obsidian-rag.git
cd obsidian-rag
./scripts/setup.sh
```

Configure your vault:

```bash
cp .env.example .env
# Edit .env and set OBSIDIAN_VAULT_PATH
```

## Usage

```bash
source .venv/bin/activate

# Ingest your vault
python src/ingest.py

# Query
python src/query.py "What topics are in my notes?"

# Start API server
uvicorn src.api:app --port 8000
```

API docs: http://localhost:8000/docs

## Tech Stack

- LlamaIndex + ChromaDB for RAG
- Ollama (Llama 3.2 3B + nomic-embed-text)
- FastAPI for serving
- MLflow for experiment tracking

## Contributing

This is a personal portfolio project. Suggestions welcome via issues.

## License

MIT
