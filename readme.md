## Project Overview

Bank BorjaM RAG API - A banking virtual assistant using Retrieval-Augmented Generation (RAG) built with FastAPI, LangChain, and ChromaDB. The system supports both **local** (Ollama with local embeddings) and **cloud** (OpenAI) operation modes.

## Development Commands

### Environment Setup
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Vectorstore Management
```bash
# Rebuild the vectorstore (required before first run or after changing documents)
python -m scripts.rebuild_vectorstore --force

# Alternative script (fixed version)
python -m scripts.rebuild_vectorstore_fixed --force
```

### Running the Application
```bash
# Development mode (with auto-reload)
uvicorn app.main:app --reload

# Production mode (uses settings from .env)
python -m app.main

# Custom host/port
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Local LLM Setup (MODE=local)
```bash
# Ensure Ollama is running
ollama serve

# Pull the required model
ollama run llama3.2:3b
```

### Testing Endpoints
```bash
# Health check
curl http://localhost:8000/api/v1/health

# Stats
curl http://localhost:8000/api/v1/stats

# Chat endpoint
curl -X POST http://localhost:8000/api/v1/chat/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Qué servicios bancarios ofrecen?"}'
```

### .env

```
# ==============================================================================
# BANK BORJAM RAG - CONFIGURATION FILE
# ==============================================================================
# Environment configuration for local and cloud deployment modes

# ==============================================================================
# OPERATION MODE
# ==============================================================================
MODE=cloud
# Options: local (private, no costs) | cloud (OpenAI, better performance)

# ==============================================================================
# PROJECT SETTINGS
# ==============================================================================
PROJECT_NAME=Borgian Bank RAG API
ENVIRONMENT=development
# Options: development | production | testing

# ==============================================================================
# SERVER CONFIGURATION
# ==============================================================================
HOST=127.0.0.1
PORT=8000
ENABLE_DOCS=true

# ==============================================================================
# PATHS
# ==============================================================================
DOCS_PATH=./docs
VECTORSTORE_PATH=./vectorstore
LOGS_PATH=./logs/conversations.jsonl

# ==============================================================================
# LOCAL MODELS (MODE=local)
# ==============================================================================
LOCAL_EMBEDDING_MODEL=all-MiniLM-L6-v2
LOCAL_LLM_MODEL=llama3.2:3b
OLLAMA_BASE_URL=http://localhost:11434

# ==============================================================================
# CLOUD MODELS (MODE=cloud)
# ==============================================================================
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
CLOUD_LLM_MODEL=gpt-4o-mini

# ==============================================================================
# RETRIEVAL CONFIGURATION
# ==============================================================================
RETRIEVER_K=5
RETRIEVER_FETCH_K=20
LAMBDA_MULT=0.7

# ==============================================================================
# CHUNKING CONFIGURATION
# ==============================================================================
CHUNK_SIZE=800
CHUNK_OVERLAP=120

# ==============================================================================
# API KEYS (Only required for cloud mode)
# ==============================================================================
OPENAI_API_KEY=sk-proj-...

# ==============================================================================
# LOGGING
# ==============================================================================
LOG_LEVEL=INFO
```

## Architecture

### Clean Architecture Structure
The codebase follows **Clean Architecture** principles with clear separation of concerns:

```
app/
├── core/           # Configuration and settings (Pydantic settings with .env)
├── api/v1/         # API layer (FastAPI routers and endpoints)
├── services/       # Business logic (RAG service orchestration)
├── rag/            # RAG components (LLM, embeddings, vectorstore, retriever)
├── schemas/        # Data models (Pydantic schemas for request/response)
└── utils/          # Utilities (logging, helpers)
```

### Key Design Patterns

**1. Factory Pattern** - `app/rag/llm.py`
- `LLMFactory` creates appropriate LLM provider based on MODE setting
- `LocalLLMProvider` for Ollama (local mode)
- `CloudLLMProvider` for OpenAI (cloud mode)

**2. Repository Pattern** - `app/rag/vectorstore.py`
- `VectorStoreManager` abstracts all vector database operations
- Centralized CRUD operations for the vectorstore
- Includes validation and statistics methods

**3. Strategy Pattern** - `app/rag/retriever.py`
- `OptimizedRetriever` uses Maximum Marginal Relevance (MMR) for diversity
- `HybridRetriever` extends for future keyword + vector hybrid search

**4. Dependency Injection** - `app/api/v1/endpoints/chat.py`
- FastAPI's `Depends()` for service injection
- Ensures testability and loose coupling

### Configuration System

**Environment-based configuration** using Pydantic Settings (`app/core/config.py`):
- All settings loaded from `.env` file
- Strong validation with Pydantic validators
- MODE switching: `local` vs `cloud`
- Path resolution and directory creation on startup

**Critical validators:**
- `ensure_api_key_if_cloud` - Requires OPENAI_API_KEY when MODE=cloud
- `validate_chunk_overlap` - Ensures overlap < chunk_size
- Path normalization for DOCS_PATH, VECTORSTORE_PATH, LOGS_PATH

### RAG Pipeline Flow

1. **Request** → `app/api/v1/endpoints/chat.py` (FastAPI endpoint)
2. **Service** → `app/services/rag_services.py` (business logic orchestration)
3. **Retrieval** → `app/rag/retriever.py` (document retrieval with MMR)
4. **Context** → Format documents + conversation history
5. **Generation** → `app/rag/llm.py` (LLM inference)
6. **Response** → Structured response with sources and metadata
7. **Logging** → `logs/conversations.jsonl` (append-only conversation log)

### Document Processing

**Vectorstore building** (`scripts/rebuild_vectorstore.py`):
- Loads markdown documents from `DOCS_PATH` (default: `./docs`)
- **Two-stage chunking**:
  1. `MarkdownHeaderTextSplitter` - Splits by headers (#, ##, ###)
  2. `RecursiveCharacterTextSplitter` - Further chunks by size/overlap
- **Banking metadata extraction**: Keywords, content type, priority
- **Metadata serialization**: Complex objects converted to JSON strings for ChromaDB compatibility
- Persists to `VECTORSTORE_PATH` (default: `./vectorstore`)

### Embeddings

**Mode-dependent embeddings** (`app/rag/embeddings.py`):
- **Local**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- **Cloud**: OpenAI `text-embedding-3-small` (1536 dimensions)
- Cached with `@lru_cache(maxsize=1)` for singleton pattern
- Dimension validation to prevent embedding mismatch errors

### LLM Configuration

**Local (Ollama)**:
- Model: `llama3.2:3b`
- Temperature: 0.3
- Max tokens: 800 (num_predict)
- Connection validation checks Ollama server and model availability

**Cloud (OpenAI)**:
- Model: `gpt-4o-mini`
- Temperature: 0.3
- Max tokens: 800
- Timeout: 30s, max retries: 2

### Retrieval Configuration

**MMR (Maximum Marginal Relevance)**:
- `k=5`: Documents returned to user
- `fetch_k=20`: Documents fetched for MMR re-ranking
- `lambda_mult=0.7`: Balance between relevance (1.0) and diversity (0.0)

### Logging

**Structured logging** with `structlog` (`app/utils/logger.py`):
- JSON format for production parsing
- Contextual fields (session_id, processing_time, etc.)
- Conversation history logged to `LOGS_PATH` (append-only JSONL)

## Important Implementation Details

### Vectorstore Validation
The `VectorStoreManager._validate_vectorstore()` method performs critical checks:
- Document count > 0
- Embedding dimension matches current configuration
- If dimension mismatch detected, vectorstore MUST be rebuilt

### LRU Caching
Several components use `@lru_cache(maxsize=1)` for singleton pattern:
- `get_llm()` - Reuses LLM instance
- `get_embeddings()` - Reuses embedding model
- `get_retriever()` - Reuses retriever configuration
- `get_vectorstore()` - Reuses vectorstore connection

**Cache clearing**: Call `function_name.cache_clear()` when forcing reload

### Application Lifespan
`app/main.py` implements `@asynccontextmanager` for startup/shutdown:
- Startup: Checks vectorstore exists, validates Ollama connection (local mode)
- Shutdown: Clean resource cleanup

### CORS Configuration
CORS middleware configured for local development:
- Allowed origins: `localhost:3000`, `localhost:5173`, `127.0.0.1:5173`
- All methods and headers allowed

### Metadata Serialization
ChromaDB only supports simple metadata types (str, int, float, bool). Complex types (lists, dicts) MUST be serialized to JSON strings:
```python
metadata["banking_keywords"] = json.dumps(found_keywords, ensure_ascii=False)
```

## Common Issues

### Vectorstore Not Found
**Error**: "Vectorstore not found or invalid"
**Solution**: Run `python -m scripts.rebuild_vectorstore --force`

### Embedding Dimension Mismatch
**Error**: "Embedding dimension mismatch"
**Cause**: Switched between local/cloud mode with existing vectorstore
**Solution**: Delete `vectorstore/` directory and rebuild

### Ollama Connection Failed
**Error**: "Cannot connect to Ollama"
**Solution**:
1. Start Ollama: `ollama serve`
2. Ensure model is pulled: `ollama pull llama3.2:3b`

### OpenAI API Key Missing
**Error**: "OPENAI_API_KEY is required when MODE=cloud"
**Solution**: Add `OPENAI_API_KEY=sk-...` to `.env` file

## Testing and Validation

No formal test suite currently exists. Manual testing endpoints:
- `/api/v1/health` - System health check
- `/api/v1/stats` - Vectorstore and system statistics
- `/api/v1/chat/ask` - Main RAG endpoint
- `/api/v1/chat/service-stats` - RAG service metrics

## File Locations

- **Source documents**: `./docs/` (markdown files)
- **Vectorstore**: `./vectorstore/` (ChromaDB persistence)
- **Conversation logs**: `./logs/conversations.jsonl` (append-only)
- **Configuration**: `.env` (git-ignored, see readme for template)

## PyCharm Warnings Suppression

The codebase uses `# noinspection` comments to suppress false positives:
- `PyProtectedMember` - Accessing `vectorstore._collection` (internal API)
- `PyMethodMayBeStatic` - Methods that could be static but maintain instance context
