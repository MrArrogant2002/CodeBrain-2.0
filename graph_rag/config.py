"""
Graph RAG configuration.
Change OLLAMA_MODEL here to switch the LLM for all graph_rag commands,
or pass --model on the CLI to override per-run.
"""

# ── Neo4j ──────────────────────────────────────────────────────────────────────
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "codebrain2024"  # must match NEO4J_AUTH in docker-compose.yml

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
