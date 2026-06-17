"""
Normal RAG configuration.
Change OLLAMA_MODEL here to switch the LLM for all normal_rag commands,
or pass --model on the CLI to override per-run.
"""

# ── Embedding ──────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # sentence-transformers model
EMBEDDING_DIM = 384  # must match the model above

# ── LLM ───────────────────────────────────────────────────────────────────────
OLLAMA_MODEL = "codellama"  # any model pulled with: ollama pull <name>
OLLAMA_BASE_URL = "http://localhost:11434"

# ── Retrieval ──────────────────────────────────────────────────────────────────
VECTOR_TOP_K = 5  # how many chunks to retrieve per query
