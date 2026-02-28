## Week 1: Ingestion, Retrieval, and MLOps Foundation

### Goal
Working ingestion and retrieval with logging, experiment tracking, and prompt versioning from the first query.

### Day 1: Config, logging, and first governance docs

Three things before any ML code: config management, structured logging, and governance doc templates.

**`.env.example`** (committed)
```
OBSIDIAN_VAULT_PATH=/path/to/your/vault
LLM_MODEL=llama3.2:3b
EMBED_MODEL=nomic-embed-text
CHUNK_SIZE=512
CHUNK_OVERLAP=50
TOP_K=5
RESPONSE_MODE=compact
SYSTEM_PROMPT_VERSION=v1
CHROMA_PERSIST_DIR=./data/chroma
OLLAMA_BASE_URL=http://localhost:11434
MLFLOW_TRACKING_URI=./mlruns
LOG_LEVEL=INFO
```

**`src/config.py`** — All pipeline parameters centralised. Changing model, chunk size, or prompt version is a config change, not a code change.
```python
"""
Centralised configuration from environment variables.

MLOps principle: pipeline parameters are config, not code.
Every parameter that affects output should be here and logged to MLflow.
"""
from dotenv import load_dotenv
import os

load_dotenv()

# Data
OBSIDIAN_VAULT_PATH = os.getenv("OBSIDIAN_VAULT_PATH")

# Models
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Retrieval
TOP_K = int(os.getenv("TOP_K", "5"))
RESPONSE_MODE = os.getenv("RESPONSE_MODE", "compact")

# Prompt
SYSTEM_PROMPT_VERSION = os.getenv("SYSTEM_PROMPT_VERSION", "v1")

# Storage
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")

# Observability
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


def get_pipeline_params() -> dict:
    """Return all pipeline parameters for MLflow logging."""
    return {
        "llm_model": LLM_MODEL,
        "embed_model": EMBED_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "top_k": TOP_K,
        "response_mode": RESPONSE_MODE,
        "system_prompt_version": SYSTEM_PROMPT_VERSION,
    }
```

**`src/logging_config.py`** — Structured JSON logging. Sets up clean transition to CloudWatch in Phase 2.
```python
"""
Structured JSON logging.

Why JSON: parseable by CloudWatch (Phase 2), searchable,
and each log entry can carry metrics (latency, chunk count, etc).
"""
import logging
import json
import sys
from datetime import datetime, timezone

from config import LOG_LEVEL


class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_data"):
            log_entry.update(record.extra_data)
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    return logger
```

**`docs/data_provenance.md`** — Start now, fill in after first ingest.
```markdown
# Data Provenance

## Source
- **Type:** Personal Obsidian vault (Markdown files)
- **Location:** Local filesystem, synced via iCloud
- **Access:** Private — not included in repository
- **Content:** Personal notes on [topics — fill in]
- **Volume:** [X files, ~Y MB — fill in after first ingest]

## Preprocessing
- File types: .md only
- Excluded: .obsidian/, .trash/, templates/
- Preprocessing applied: [fill in — e.g. wikilink stripping, frontmatter handling]

## Versioning approach
- Raw vault is not versioned (personal, changes constantly)
- Ingestion metadata tracked via MLflow (file count, chunk count, timestamp)
- Evaluation dataset (eval/test_questions.json) IS version controlled
```

**`docs/model_card.md`** — Template. Completed properly in Phase 4.
```markdown
# Model Card: Obsidian RAG Pipeline

## Model Details
- **Pipeline version:** 0.1.0
- **LLM:** [model name and version]
- **Embedding model:** [model name]
- **Framework:** LlamaIndex + Ollama

## Intended Use
- **Primary use:** Question-answering over personal knowledge base
- **Users:** Single user (developer/owner of the vault)
- **Out of scope:** [fill in]

## Data
- See data_provenance.md

## Evaluation
- [To be completed in Phase 3]

## Limitations
- [Fill in as you discover them]

## Ethical Considerations
- Private data only — no PII of others ingested
```

**Commit:**
```bash
git add .
git commit -m "MLOps foundation: config, logging, governance doc templates"
```

### Day 2: Ingestion pipeline with experiment tracking

**`src/ingest.py`**
```python
"""
Ingest Obsidian vault into ChromaDB vector store.

Every ingestion run is tracked in MLflow with parameters and metrics.
"""
import time
import mlflow
import chromadb
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    Settings,
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
    MLFLOW_TRACKING_URI,
    get_pipeline_params,
)
from logging_config import get_logger

logger = get_logger(__name__)


def configure_settings():
    """Configure LlamaIndex global settings."""
    Settings.embed_model = OllamaEmbedding(
        model_name=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    Settings.llm = Ollama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        request_timeout=120.0,
    )


def load_documents(vault_path: str):
    """Load all markdown files from the Obsidian vault."""
    reader = SimpleDirectoryReader(
        input_dir=vault_path,
        recursive=True,
        required_exts=[".md"],
        exclude=[".obsidian", ".trash", "templates"],
    )
    documents = reader.load_data()
    logger.info(
        f"Loaded {len(documents)} documents",
        extra={"extra_data": {"num_documents": len(documents), "vault_path": vault_path}},
    )
    return documents


def create_index(documents):
    """Chunk documents and store embeddings in ChromaDB."""
    node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # Delete existing collection for clean re-ingestion
    try:
        chroma_client.delete_collection("obsidian_vault")
    except ValueError:
        pass

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


def ingest():
    """Main ingestion pipeline — tracked in MLflow."""
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
```

**Run it:**
```bash
cd ~/projects/obsidian-rag
source .venv/bin/activate
# Make sure Ollama is running (ollama serve)
python src/ingest.py

# View your first experiment:
mlflow ui --port 5001
# Open http://localhost:5001
```

**What to observe and write in `docs/experiment_log.md`:**
- How many documents loaded? How many chunks created?
- How long did ingestion take?
- What's the chunks-per-document ratio?

**Commit:**
```bash
git add .
git commit -m "Ingestion pipeline with MLflow tracking"
```

### Day 3: System prompt versioning

A core MLOps practice for LLM systems that most tutorials skip. The system prompt shapes every response — it needs to be versioned and tracked like code.

**`src/prompts/system_v1.txt`**
```
You are a helpful assistant that answers questions based on the provided context from a personal knowledge base.

Rules:
- Only answer based on the provided context. If the context doesn't contain enough information, say so.
- Quote or reference specific notes where possible.
- Be concise and direct.
- If the question is ambiguous, ask for clarification rather than guessing.
```

**Why version prompts as files?**
- Git tracks changes — full diff history
- MLflow logs which version was used for each experiment
- A/B test prompts by changing one config value
- Phase 4 governance docs can reference specific prompt versions

### Day 3–4: Query engine with prompt management and metrics

**`src/query.py`**
```python
"""
Query engine with prompt versioning and per-query metrics.
"""
import os
import time
import chromadb
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from config import (
    EMBED_MODEL,
    LLM_MODEL,
    TOP_K,
    RESPONSE_MODE,
    SYSTEM_PROMPT_VERSION,
    CHROMA_PERSIST_DIR,
    OLLAMA_BASE_URL,
)
from logging_config import get_logger

logger = get_logger(__name__)


def load_system_prompt(version: str = None) -> str:
    """Load a versioned system prompt from file."""
    version = version or SYSTEM_PROMPT_VERSION
    prompt_path = os.path.join(
        os.path.dirname(__file__), "prompts", f"system_{version}.txt"
    )
    with open(prompt_path) as f:
        return f.read().strip()


def configure_settings():
    """Configure LlamaIndex global settings."""
    Settings.embed_model = OllamaEmbedding(
        model_name=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    Settings.llm = Ollama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        request_timeout=120.0,
    )


def load_index():
    """Load existing ChromaDB index."""
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    chroma_collection = chroma_client.get_collection("obsidian_vault")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return VectorStoreIndex.from_vector_store(vector_store)


def query(question: str, top_k: int = None, prompt_version: str = None):
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

    sources = [
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

    configure_settings()
    question = " ".join(sys.argv[1:]) or "What topics are in my notes?"
    result = query(question)
    print(f"\nAnswer: {result['answer']}\n")
    print(f"Metrics: {result['metrics']}")
    print("\nSources:")
    for i, source in enumerate(result["sources"], 1):
        print(f"  {i}. {source['file']} (score: {source['score']:.3f})")
```

**Test it:**
```bash
python src/query.py "What do I know about sprint planning?"
```

**Commit:**
```bash
git add .
git commit -m "Query engine with prompt versioning and structured logging"
```

---

## Week 2: API, Experiments, and Evaluation Start

### Goal
FastAPI server with metrics, chunking and prompt experiments tracked in MLflow, start the evaluation dataset.

### Day 5–6: FastAPI server with per-query metrics

**`src/api.py`**
```python
"""
FastAPI server for the Obsidian RAG pipeline.

Includes /metrics endpoint for observability (precursor to CloudWatch in Phase 2).
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel, Field

from query import configure_settings, load_index, load_system_prompt
from config import LLM_MODEL, TOP_K, RESPONSE_MODE, SYSTEM_PROMPT_VERSION
from logging_config import get_logger
import time

logger = get_logger(__name__)


# Simple in-memory metrics (replaced by CloudWatch in Phase 2)
metrics_store = {
    "total_queries": 0,
    "total_latency_seconds": 0.0,
    "errors": 0,
}


class AppState:
    index = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_settings()
    AppState.index = load_index()
    logger.info("Index loaded, ready to serve")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Obsidian RAG API",
    description="Query your Obsidian vault using RAG",
    version="0.1.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=TOP_K, ge=1, le=20)
    prompt_version: str = Field(default=SYSTEM_PROMPT_VERSION)


class SourceNode(BaseModel):
    file: str
    score: float
    text_preview: str


class QueryMetrics(BaseModel):
    latency_seconds: float
    num_sources: int
    top_score: float | None
    prompt_version: str
    model: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceNode]
    metrics: QueryMetrics


@app.post("/query", response_model=QueryResponse)
async def query_vault(request: QueryRequest):
    system_prompt = load_system_prompt(request.prompt_version)
    query_engine = AppState.index.as_query_engine(
        similarity_top_k=request.top_k,
        response_mode=RESPONSE_MODE,
        system_prompt=system_prompt,
    )

    start = time.time()
    try:
        response = query_engine.query(request.question)
        latency = time.time() - start

        metrics_store["total_queries"] += 1
        metrics_store["total_latency_seconds"] += latency

        sources = [
            SourceNode(
                file=node.metadata.get("file_name", "unknown"),
                score=node.score,
                text_preview=node.text[:200],
            )
            for node in response.source_nodes
        ]

        return QueryResponse(
            answer=str(response),
            sources=sources,
            metrics=QueryMetrics(
                latency_seconds=round(latency, 2),
                num_sources=len(sources),
                top_score=sources[0].score if sources else None,
                prompt_version=request.prompt_version,
                model=LLM_MODEL,
            ),
        )
    except Exception as e:
        metrics_store["errors"] += 1
        logger.error(f"Query failed: {e}")
        raise


@app.get("/health")
async def health():
    return {"status": "healthy", "index_loaded": AppState.index is not None}


@app.get("/metrics")
async def get_metrics():
    """Basic metrics. Replaced by CloudWatch in Phase 2."""
    avg_latency = (
        metrics_store["total_latency_seconds"] / metrics_store["total_queries"]
        if metrics_store["total_queries"] > 0
        else 0
    )
    return {
        "total_queries": metrics_store["total_queries"],
        "average_latency_seconds": round(avg_latency, 2),
        "errors": metrics_store["errors"],
    }
```

**Run and test:**
```bash
uvicorn src.api:app --reload --port 8000

curl http://localhost:8000/health
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What do I know about Agile?", "top_k": 5}'
curl http://localhost:8000/metrics

# Interactive docs: http://localhost:8000/docs
```

**Commit:**
```bash
git add .
git commit -m "FastAPI server with per-query metrics"
```

### Day 7–8: Chunking experiments tracked in MLflow

**`scripts/run_experiment.sh`**
```bash
#!/bin/bash
# Run chunking experiments. Results tracked in MLflow.
source .venv/bin/activate

echo "=== chunk_size=256, overlap=25 ==="
CHUNK_SIZE=256 CHUNK_OVERLAP=25 python src/ingest.py

echo "=== chunk_size=512, overlap=50 ==="
CHUNK_SIZE=512 CHUNK_OVERLAP=50 python src/ingest.py

echo "=== chunk_size=1024, overlap=100 ==="
CHUNK_SIZE=1024 CHUNK_OVERLAP=100 python src/ingest.py

echo "Done. View results: mlflow ui --port 5001"
```

After each ingestion, run the same test queries and compare. Document in `docs/experiment_log.md`:

```markdown
# Experiment Log

## 2026-04-XX: Chunking comparison

### Setup
- Vault: ~[X] notes
- Models: llama3.2:3b, nomic-embed-text
- Test queries: [list 5 queries]

### Results
| Config | Chunks | Ingest time | Q1 relevance | Q2 relevance | Notes |
|--------|--------|-------------|-------------|-------------|-------|
| 256/25 | | | | | |
| 512/50 | | | | | |
| 1024/100 | | | | | |

### Decision
- Chose [X] because [reason]. Documented in docs/architecture.md as ADR-003.
```

### Day 8: Prompt experiment

**`src/prompts/system_v2.txt`**
```
You are a knowledge assistant for a personal Obsidian vault. Your role is to help the user find and synthesise information from their notes.

Rules:
- Answer ONLY from the provided context. Never use external knowledge.
- If the context is insufficient, say "I don't have enough information in your notes about this."
- Reference the source file names so the user can find the original notes.
- Be specific — cite details from the notes rather than giving generic answers.
- Keep answers concise: 2-4 sentences unless the user asks for detail.
```

**Compare both prompts on the same questions:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What do I know about Docker?", "prompt_version": "v1"}'

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What do I know about Docker?", "prompt_version": "v2"}'
```

Log in `docs/experiment_log.md`: which prompt grounds better? Fewer hallucinations? More concise?

**Commit:**
```bash
git add .
git commit -m "Prompt v2 and chunking experiments"
```

---

## Week 3: Evaluation and Documentation

### Goal
Evaluation dataset, automated scoring tracked in MLflow, architecture docs, README.

### Day 9–10: Evaluation dataset and script

**`eval/test_questions.json`** — 20–30 questions you know the answers to:
```json
[
  {
    "question": "What ceremonies are part of Scrum?",
    "expected_source_files": ["scrum-notes.md", "agile-practices.md"],
    "expected_answer_contains": ["sprint planning", "retrospective", "daily standup"],
    "category": "factual_retrieval"
  },
  {
    "question": "What is the purpose of a DFMEA?",
    "expected_source_files": ["medical-device-notes.md"],
    "expected_answer_contains": ["failure mode", "risk"],
    "category": "factual_retrieval"
  },
  {
    "question": "Summarise my notes on Docker networking",
    "expected_source_files": ["docker-notes.md"],
    "expected_answer_contains": ["bridge", "network"],
    "category": "synthesis"
  }
]
```

**`eval/evaluate.py`** — Automated eval with MLflow tracking. This is the foundation for the full evaluation framework in Phase 3 (LLM-as-judge, W&B, LangSmith).
```python
"""
Evaluate RAG pipeline against test questions.
Tracks results in MLflow for comparison across configs.

Metrics:
- Source recall: Did we retrieve the right files?
- Answer coverage: Does the answer contain expected keywords?
- Latency: How long did each query take?
"""
import json
import mlflow

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from query import configure_settings, query
from config import MLFLOW_TRACKING_URI, get_pipeline_params


def evaluate(test_file: str = "eval/test_questions.json"):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("evaluation")

    configure_settings()

    with open(test_file) as f:
        test_cases = json.load(f)

    with mlflow.start_run(run_name=f"eval-{len(test_cases)}q"):
        mlflow.log_params(get_pipeline_params())
        mlflow.log_param("num_test_cases", len(test_cases))

        results = []
        for i, case in enumerate(test_cases, 1):
            result = query(case["question"])

            # Source recall
            retrieved_files = [s["file"] for s in result["sources"]]
            expected_files = case.get("expected_source_files", [])
            source_hits = sum(
                1 for f in expected_files if any(f in rf for rf in retrieved_files)
            )
            source_recall = source_hits / len(expected_files) if expected_files else 0

            # Answer coverage
            answer_lower = result["answer"].lower()
            expected_terms = case.get("expected_answer_contains", [])
            term_hits = sum(1 for t in expected_terms if t.lower() in answer_lower)
            answer_coverage = term_hits / len(expected_terms) if expected_terms else 0

            results.append({
                "question": case["question"],
                "category": case.get("category", "unknown"),
                "source_recall": source_recall,
                "answer_coverage": answer_coverage,
                "latency_seconds": result["metrics"]["latency_seconds"],
            })

            status = "✓" if source_recall > 0.5 and answer_coverage > 0.5 else "✗"
            print(f"  {status} Q{i}: {case['question'][:60]}...")
            print(f"    Sources: {source_recall:.0%} | Coverage: {answer_coverage:.0%} | {result['metrics']['latency_seconds']:.1f}s")

        # Aggregate
        avg_recall = sum(r["source_recall"] for r in results) / len(results)
        avg_coverage = sum(r["answer_coverage"] for r in results) / len(results)
        avg_latency = sum(r["latency_seconds"] for r in results) / len(results)
        pass_rate = sum(
            1 for r in results if r["source_recall"] > 0.5 and r["answer_coverage"] > 0.5
        ) / len(results)

        mlflow.log_metrics({
            "avg_source_recall": avg_recall,
            "avg_answer_coverage": avg_coverage,
            "avg_latency_seconds": avg_latency,
            "pass_rate": pass_rate,
        })
        mlflow.log_dict(results, "eval_results.json")

        print(f"\n{'='*60}")
        print(f"RESULTS ({len(results)} questions)")
        print(f"  Avg source recall:   {avg_recall:.0%}")
        print(f"  Avg answer coverage: {avg_coverage:.0%}")
        print(f"  Avg latency:         {avg_latency:.1f}s")
        print(f"  Pass rate:           {pass_rate:.0%}")
        print(f"\nLogged to MLflow — view at http://localhost:5001")

    return results


if __name__ == "__main__":
    evaluate()
```

**Run:**
```bash
python eval/evaluate.py
mlflow ui --port 5001  # Compare eval runs across configs
```

### Day 11: Architecture Decision Records

**`docs/architecture.md`**
```markdown
# Architecture Decisions

## ADR-001: LlamaIndex over LangChain
- **Date:** [date]
- **Decision:** LlamaIndex
- **Rationale:** Better defaults for document ingestion and retrieval
- **Tradeoff:** LangChain has wider ecosystem for complex agent chains

## ADR-002: ChromaDB for dev, Qdrant for production
- **Date:** [date]
- **Decision:** ChromaDB locally, Qdrant on AWS (Phase 2)
- **Rationale:** ChromaDB is embedded (no server). Qdrant has production features.

## ADR-003: Chunking strategy
- **Date:** [date]
- **Decision:** [your choice] because [your evidence from experiments]
- **Evidence:** See experiment_log.md and MLflow experiment "ingestion"

## ADR-004: System prompt versioning as files
- **Date:** [date]
- **Decision:** Prompts as text files in src/prompts/, version in config
- **Rationale:** Git tracks changes, MLflow logs which version used, A/B testing is config-only

## ADR-005: Config-driven pipeline
- **Date:** [date]
- **Decision:** All pipeline parameters in .env, logged to MLflow every run
- **Rationale:** Reproducibility — any run can be recreated from its MLflow params
```

### Day 12–13: README and setup script

**`scripts/setup.sh`**
```bash
#!/bin/bash
set -e
echo "Setting up Obsidian RAG pipeline..."

command -v python3 >/dev/null 2>&1 || { echo "Python 3 required"; exit 1; }
command -v ollama >/dev/null 2>&1 || { echo "Ollama required: brew install ollama"; exit 1; }

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

ollama pull llama3.2:3b
ollama pull nomic-embed-text

if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env — edit OBSIDIAN_VAULT_PATH before running"
fi

echo "Setup complete. Edit .env, then run: python src/ingest.py"
```

**`README.md`**
```markdown
# Obsidian RAG Pipeline

A Retrieval-Augmented Generation system that queries a personal Obsidian
knowledge base using local LLMs, built with MLOps practices.

## Why this project

Built to demonstrate practical AI/ML engineering skills:
- **RAG pipeline** — ingestion, chunking, retrieval, generation
- **MLOps discipline** — experiment tracking (MLflow), prompt versioning,
  config-driven pipeline, structured logging, reproducible setup
- **Evaluation** — quantitative metrics, automated eval harness
- **Governance foundations** — data provenance, model card, ADRs
  (expanded in later phases with AIAF mapping and validation reports)

## Quick start

    git clone [repo]
    cd obsidian-rag
    ./scripts/setup.sh
    # Edit .env with your vault path
    python src/ingest.py
    uvicorn src.api:app --port 8000
    # Open http://localhost:8000/docs

## MLOps practices

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

## Evaluation results

[Fill in after running eval — summary table]

## Roadmap

- [x] Phase 1: Local RAG pipeline with MLOps practices
- [ ] Phase 2: AWS deployment (Docker, EC2, CloudWatch, IaC)
- [ ] Phase 3: Fine-tuning and full evaluation framework (LLM-as-judge, W&B)
- [ ] Phase 4: AI governance documentation (NSW AIAF, model validation, SOUP)
```

**Commit and push:**
```bash
git add .
git commit -m "README, setup script, architecture docs"
git push -u origin main
```

### Day 14: First blog post

Draft:
> **"Building a RAG Pipeline with MLOps Practices: What I Learned from Medical Device Software"**
> - What RAG is and why I built it
> - How IEC 62304 thinking influenced my approach (versioning, traceability, evaluation)
> - Architecture decisions and what surprised me
> - What's next: AWS deployment and AI governance docs

---

## Phase 1 Definition of Done

### Core pipeline
- [ ] Obsidian vault ingested into ChromaDB
- [ ] Can query via CLI and get relevant, grounded answers
- [ ] FastAPI server with /query, /health, and /metrics endpoints

### MLOps practices
- [ ] All pipeline parameters centralised in config, not hardcoded
- [ ] Every ingestion run tracked in MLflow with params and metrics
- [ ] Every eval run tracked in MLflow with params and metrics
- [ ] Structured JSON logging on all operations with latency and metadata
- [ ] System prompts versioned as files, switchable via config
- [ ] Pinned Python dependencies in requirements.txt
- [ ] Reproducible setup via scripts/setup.sh and .env.example

### Experiments
- [ ] At least one chunking experiment tracked in MLflow with documented results
- [ ] At least one prompt experiment (v1 vs v2) with documented comparison

### Evaluation (foundation for Phase 3)
- [ ] Evaluation dataset of 20+ questions created
- [ ] Evaluation script runs, produces source recall / answer coverage / pass rate
- [ ] Eval results logged to MLflow as metrics and artifacts

### Documentation (foundation for Phase 4)
- [ ] Architecture Decision Records started (3+ ADRs)
- [ ] Data provenance doc started
- [ ] Model card template created
- [ ] Experiment log with human-readable observations

### Portfolio
- [ ] Git repo on GitHub with clean structure
- [ ] README documents the project, MLOps practices, and roadmap
- [ ] First blog post drafted

## What to avoid

- **Don't build Phase 2–4 yet.** No Docker, no AWS, no fine-tuning, no AIAF reports.
- **Don't optimise too early.** Get the pipeline working end-to-end first.
- **Don't worry about a UI.** Swagger docs at /docs is your UI.
- **Don't try multiple frameworks.** Stick with LlamaIndex.
- **Don't ingest your entire vault if it's huge.** Start with ~50–100 notes.

## Troubleshooting

| Problem | Likely cause | Fix |
|---------|-------------|-----|
| Ollama times out during ingestion | Embedding large batch | Increase `request_timeout` in configure_settings() |
| ChromaDB errors on re-ingestion | Stale collection | The ingest script deletes and recreates automatically |
| Poor retrieval quality | Chunk size wrong | Run experiment script, compare in MLflow |
| LLM hallucinating despite good sources | Prompt not constraining enough | Try system_v2.txt or write a v3 |
| Import errors | Wrong Python env | `source .venv/bin/activate` |
| Mac sluggish during queries | 8GB RAM under pressure | Close Chrome/Slack/Teams |
| Slow queries (>30s) | Too many apps competing for RAM | Close other apps; reduce top_k to 3 |
| MLflow UI shows no data | Wrong tracking URI | Check MLFLOW_TRACKING_URI in .env matches ./mlruns |

## What Phase 2 builds on

Phase 2 (AWS Deployment) takes everything from Phase 1 and:
- Containerises it with Docker
- Deploys to EC2 with a GPU (swap to Llama 3.1 8B+ model)
- Replaces in-memory /metrics with CloudWatch
- Replaces JSON stdout logging with CloudWatch Logs
- Adds Terraform/CDK for infrastructure-as-code
- Puts FastAPI behind a load balancer with HTTPS
- Introduces vLLM for production-grade inference serving

The config management, structured logging, MLflow tracking, prompt versioning, and evaluation framework all carry forward directly.