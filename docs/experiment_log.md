# Experiment Log

## 2026-02-28: First Ingestion Run - Day 2 Implementation

### Setup
- Vault: Test vault with 2 markdown files (notes.md, technical.md)
- Models: llama3.2:3b, nomic-embed-text
- Chunk size: 512 tokens, overlap: 50 tokens
- Platform: macOS with Python 3.12 (resolved Python 3.14 compatibility issues)

### Results
| Metric | Value | Notes |
|--------|--------|--------|
| Documents loaded | 2 | Test files: notes.md, technical.md |
| Chunks created | 2 | 1 chunk per document (documents were small) |
| Chunks per document | 1.0 | Documents fit within single 512-token chunks |
| Load time | ~0.1s | Very fast for small test vault |
| Index time | ~1.7s | Includes embedding generation via Ollama |
| Total time | 1.8s | Fast due to small dataset size |

### Observations
1. **Structured logging working**: JSON logs with metrics successfully generated
2. **MLflow tracking active**: Experiment "ingestion" created, run logged with all parameters and metrics
3. **ChromaDB integration successful**: Vector store created and populated
4. **Ollama embedding generation**: nomic-embed-text model working properly
5. **Small chunk count**: Each document produced only 1 chunk due to small size

### Technical Notes
- Fixed Python 3.14 compatibility issue by using Python 3.12
- Resolved relative import issue in logging_config.py
- ChromaDB collection deletion error handled gracefully on first run
- MLflow shows warning about filesystem backend deprecation (will address in Phase 2)

### Next Steps
- Test with larger documents to see chunking behavior
- Implement query engine to test retrieval quality
- Create system prompt versioning as specified in plan

### MLflow UI
- Available at http://localhost:5001
- Shows complete run with parameters and metrics
- Ready for experiment comparison when we run chunking experiments