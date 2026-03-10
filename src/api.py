"""
FastAPI server for the Obsidian RAG pipeline.

Includes /metrics endpoint for observability (precursor to CloudWatch in Phase 2).
"""
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import LLM_MODEL, RESPONSE_MODE, SYSTEM_PROMPT_VERSION, TOP_K
from logging_config import get_logger
from query import configure_settings, load_index, load_system_prompt

logger = get_logger(__name__)

# Simple in-memory metrics (replaced by CloudWatch in Phase 2)
metrics_store = {
    "total_queries": 0,
    "total_latency_seconds": 0.0,
    "errors": 0,
}


class AppState:
    """Holds application-level state loaded at startup."""
    index = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load index once at startup; clean up on shutdown."""
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
    """Request body for /query."""
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=TOP_K, ge=1, le=20)
    prompt_version: Optional[str] = Field(default=SYSTEM_PROMPT_VERSION)


class SourceNode(BaseModel):
    """A single retrieved source chunk."""
    file: str
    score: float
    text_preview: str


class QueryMetrics(BaseModel):
    """Per-query performance metrics."""
    latency_seconds: float
    num_sources: int
    top_score: Optional[float]
    prompt_version: str
    model: str


class QueryResponse(BaseModel):
    """Response body for /query."""
    answer: str
    sources: list[SourceNode]
    metrics: QueryMetrics


@app.post("/query", response_model=QueryResponse)
async def query_vault(request: QueryRequest):
    """Query the Obsidian vault and return a grounded answer."""
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

        logger.info(
            "Query completed",
            extra={"extra_data": {
                "question": request.question[:100],
                "latency_seconds": round(latency, 2),
                "num_sources": len(sources),
                "top_score": sources[0].score if sources else None,
                "prompt_version": request.prompt_version,
            }},
        )

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
    except Exception as exc:
        metrics_store["errors"] += 1
        logger.error("Query failed", extra={"extra_data": {"error": str(exc)}})
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
async def health():
    """Health check — confirms the index is loaded."""
    return {"status": "healthy", "index_loaded": AppState.index is not None}


@app.get("/metrics")
async def get_metrics():
    """Basic query metrics. Replaced by CloudWatch in Phase 2."""
    total = metrics_store["total_queries"]
    avg_latency = (
        metrics_store["total_latency_seconds"] / total if total > 0 else 0.0
    )
    return {
        "total_queries": total,
        "average_latency_seconds": round(avg_latency, 2),
        "errors": metrics_store["errors"],
    }
