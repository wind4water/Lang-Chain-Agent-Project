# RAG (Retrieval-Augmented Generation) Module

## Overview

This is a production-ready RAG system with the following features:

- **Multi-Model Support**: Different embedding models for different document types (BGE for documents, UniXcoder for code)
- **Hybrid Retrieval**: Vector + Keyword retrieval with RRF fusion
- **Intelligent Routing**: Auto-detect query intent and narrow search scope
- **Incremental Sync**: Smart sync based on document registry
- **ES Integration**: Elasticsearch for keyword retrieval

## Architecture

```
app/rag/
├── __init__.py                 # Module exports
├── config.py                   # Configuration management
├── system.py                   # Main RAG system controller
│
├── core/                       # Core components
│   ├── chain.py                # RAG Q&A chain with fallback
│   ├── embeddings.py           # Embedding model manager
│   ├── code_embeddings.py      # UniXcoder code embeddings
│   ├── vectorstore.py          # Chroma vector store manager
│   ├── document_registry.py    # Document metadata tracking
│   ├── multi_model_manager.py  # Multi-model lifecycle
│   ├── multi_model_config.py   # Multi-model configuration
│   └── retrievers/             # Retrieval components
│       ├── hybrid_retriever.py
│       ├── multi_model_hybrid_retriever.py
│       └── es_keyword_retriever.py
│
├── loaders/                    # Document loading
│   └── document_loader.py
│
├── processors/                 # Document processing
│   └── splitter.py
│
└── utils/                      # Utilities
    ├── logging_utils.py        # Structured logging
    └── logging_config.py       # Log configuration
```

## Quick Start

### Configuration

Set environment variables in `.env`:

```bash
# RAG Configuration
RAG_ENABLED=true
RAG_REBUILD_ON_STARTUP=false
RAG_TOP_K=6
RAG_RETRIEVAL_MODE=hybrid

# Embedding Models
EMBEDDING_TYPE=local
EMBEDDING_MODEL=bge-base-zh-v1.5
EMBEDDING_DEVICE=cpu

# Multi-Model Config (JSON)
MULTI_MODEL_CONFIG={"configs":[{"path":"./data/documents","model_type":"bge","model_name":"BAAI/bge-base-zh-v1.5","dimension":768},{"path":"./data/projects","model_type":"unixcoder","model_name":"microsoft/unixcoder-base","dimension":768}]}

# Elasticsearch (Optional)
ES_ENABLED=false
ES_HOSTS=http://127.0.0.1:9200
ES_INDEX_NAME=rag_keyword_chunks

# Logging
RAG_LOG_LEVEL=INFO
RAG_VERBOSE=false
```

### Usage

```python
from app.rag.system import rag_system

# Initialize
await rag_system.initialize()

# Query
result = await rag_system.query("What is LanguageEnum?", with_sources=True)
print(result["answer"])
for source in result["sources"]:
    print(f"- {source['filename']}: {source['content'][:100]}")

# Rebuild knowledge base
await rag_system.rebuild_knowledge_base()

# Sync (incremental)
await rag_system.sync_knowledge_base()
```

## Multi-Model Architecture

The system supports different embedding models for different document types:

| Path | Model Type | Model Name | Purpose |
|------|-----------|------------|---------|
| `./data/documents` | bge | BAAI/bge-base-zh-v1.5 | General documents |
| `./data/projects/*.md` | bge | BAAI/bge-base-zh-v1.5 | Project documentation |
| `./data/projects/*.{java,py,js}` | unixcoder | microsoft/unixcoder-base | Source code |

### Why UniXcoder for Code?

UniXcoder is specifically designed for code understanding and retrieval:

- **Better Retrieval**: Optimized for code search tasks
- **Multi-Language**: Supports Java, Python, JavaScript, etc.
- **Unified Space**: Same embedding space for queries and code

## Retrieval Pipeline

```
User Query
    ↓
[Intent Detection]
    ↓
[Smart Routing] → doc_group filter (documents/projects)
    ↓
[Parallel Retrieval]
    ├── Document Retriever (BGE)
    ├── Code Retriever (UniXcoder)
    └── ES Keyword Retriever
    ↓
[RRF Fusion] (Reciprocal Rank Fusion)
    ↓
[Context Assembly]
    ↓
[LLM Generation]
    ↓
Answer + Sources
```

## Logging

### Log Levels

| Level | Usage |
|-------|-------|
| DEBUG | Detailed debugging (enable with RAG_VERBOSE=true) |
| INFO | Key workflow steps |
| WARNING | Degradation, configuration issues |
| ERROR | Operation failures |

### Performance Logging

Use the `timed` context manager for performance monitoring:

```python
from app.rag.utils.logging_utils import timed

with timed("document_loading", logger):
    docs = loader.load()
```

Logs output:
```
[PERF] document_loading 开始
[PERF] document_loading 完成 | 耗时: 1.234s
```

### Structured Logging

```python
from app.rag.utils.logging_utils import log_structured, RAGLogEvent

log_structured(
    logger, logging.INFO,
    RAGLogEvent.RETRIEVE_COMPLETE,
    "Retrieval completed",
    query="user query",
    results_count=10
)
```

## API Reference

### RAGSystem

Main controller class for the RAG system.

#### Methods

- `initialize()`: Initialize the system
- `query(question, with_sources=True, doc_group=None)`: Query the system
- `rebuild_knowledge_base()`: Full rebuild of the knowledge base
- `sync_knowledge_base()`: Incremental sync
- `get_stats()`: Get system statistics

### Configuration Options

See `RAGConfig` in `config.py` for all configuration options.

| Option | Default | Description |
|--------|---------|-------------|
| `rag_log_level` | INFO | Log level |
| `rag_verbose` | false | Enable verbose logging |
| `retrieval_mode` | vector | Retrieval mode (vector/hybrid) |
| `top_k` | 4 | Number of results to return |
| `chunk_size` | 1000 | Text chunk size |
| `chunk_overlap` | 200 | Text overlap between chunks |

## Troubleshooting

### Model Download Issues

If models fail to download automatically:

```bash
# Download UniXcoder manually
python download_unixcoder.py
```

### Vector Store Issues

To reset the vector store:

```bash
rm -rf ./data/vectordb/chroma/
# Restart application to rebuild
```

### Memory Issues

For large document sets:

```bash
# Use CPU instead of GPU
EMBEDDING_DEVICE=cpu

# Reduce batch size
EMBEDDING_BATCH_SIZE=32
```

## Development

### Adding a New Embedding Model

1. Create embedding class in `core/code_embeddings.py`
2. Add model configuration to `multi_model_config.py`
3. Update `multi_model_manager.py` to support the new type

### Testing

```bash
# Test UniXcoder
python test_unixcoder.py

# Test RAG system
python -m pytest tests/unit/test_rag_simple.py -v
```

## License

MIT License
