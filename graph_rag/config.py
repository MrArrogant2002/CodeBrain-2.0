"""
Graph RAG configuration.
Change OLLAMA_MODEL here to switch the LLM for all graph_rag commands,
or pass --model on the CLI to override per-run.

Neo4j credentials are read from the environment (or the project .env file);
the defaults below match docker-compose.yml for local development.
"""

import os
from pathlib import Path


def _load_dotenv() -> None:
    """Load KEY=VALUE pairs from the project .env without overriding real env vars."""
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()

# ── Neo4j ──────────────────────────────────────────────────────────────────────
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "codebrain2024")

# ── Embedding ──────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # sentence-transformers model
EMBEDDING_DIM = 384  # must match the model above
VECTOR_INDEX_NAME = "code_entity_embeddings"

# ── LLM ───────────────────────────────────────────────────────────────────────
OLLAMA_MODEL = "codellama"  # any model pulled with: ollama pull <name>
OLLAMA_BASE_URL = "http://localhost:11434"

# ── Retrieval ──────────────────────────────────────────────────────────────────
VECTOR_TOP_K = 5  # initial vector-search entry points
GRAPH_HOP_DEPTH = 1  # CALLS hops to follow from each entry point
