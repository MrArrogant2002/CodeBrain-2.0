# CodeBrain

**Normal RAG vs Graph RAG for code question-answering — a side-by-side comparison prototype.**

CodeBrain answers natural-language questions about a codebase using two competing retrieval strategies, so their behavior can be compared on identical queries:

| | Normal RAG | Graph RAG |
|---|---|---|
| Retrieval | FAISS flat vector search | Neo4j vector search + `CALLS`-edge expansion |
| Context | Top-k textually similar chunks | Entry points **plus** their callers and callees |
| Infrastructure | None (in-memory) | Neo4j via Docker |
| Strength | Simple, zero setup | Assembles real execution paths across files |

Both systems parse **Python (`.py`) and C (`.c`/`.h`)** with tree-sitter, embed with sentence-transformers (`all-MiniLM-L6-v2`), and generate answers with a local LLM through Ollama (default `codellama`).

## What you need to provide

Only the small demo corpus ships with the repo — **to analyze a real codebase you add it yourself**:

- `sample_codebase/` — included. A small Python auth/API demo with deliberate deep call chains; everything works out of the box against it.
- **Your own codebase — not included.** Copy any directory of `.py`, `.c`, or `.h` files into the project root (e.g. `MyProject_Source/`) and point the commands at it with `--path MyProject_Source`. Build-output directories (`Debug/`, `Release/`, `build/`, `dist/`) are skipped automatically.

> The examples below use `BMS_Source_Code/` (NXP S32K144 battery-management firmware, ~1,600 entities across 74 C files) as the private corpus. That folder is confidential and deliberately **not in this repository** (see `.gitignore`) — substitute your own source directory.

If you keep your codebase private too, add its folder name to `.gitignore` before your first commit.

## How it works

```
Normal RAG:  source ──tree-sitter──> chunks ──embed──> FAISS ──top-k──> LLM ──> answer

Graph RAG:   source ──tree-sitter──> entities + CALLS ──embed──> Neo4j
             query ──embed──> vector search ──> entry points
                   ──expand CALLS edges──> callers + callees ──> LLM ──> answer
```

Graph RAG nodes are keyed by qualified name (`file::name`), so same-named functions
in different C files remain distinct — bare names collide in real firmware.

## Quick start

Prerequisites: Python 3.10+, [Ollama](https://ollama.com), and Docker Desktop (Graph RAG only).

```bash
# 1. Environment
python -m venv .venv
source .venv/Scripts/activate        # Windows Git Bash; PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Pull an LLM
ollama pull codellama                # or: qwen2.5-coder:7b (better at code Q&A)
```

### Normal RAG (no database needed)

```bash
python -m normal_rag.run --query "How does the login flow verify a password?"
python -m normal_rag.run --path BMS_Source_Code --query "How are cell voltages read from the MC33771C?"
```

### Graph RAG

```bash
docker compose up -d neo4j                                   # wait ~30s for healthy
python -m graph_rag.ingest --path BMS_Source_Code --clear    # parse -> embed -> write graph
python -m graph_rag.run --query "How does the SPI communication with the BCC work?"
```

### Side-by-side comparison

```bash
python compare.py --path BMS_Source_Code --query "What happens in the main measurement loop?"
```

Common flags: `--model <ollama-model>` to switch LLMs, `--top-k <n>` for more context,
no `--query` for interactive mode.

## Project structure

```
normal_rag/          FAISS pipeline: config, parser, vector store, CLI
graph_rag/           Neo4j pipeline: config, parser, graph store, hybrid retriever, ingest, CLI
compare.py           Runs both systems on one query with timing summary
sample_codebase/     Python demo corpus (included)
<your codebase>/     Your own .py/.c/.h source directory (you add this; gitignored if private)
docker-compose.yml   Neo4j 5.15 container
DOCS.md              Complete documentation
```

## Documentation

See **[DOCS.md](DOCS.md)** for the full reference: configuration, changing models,
Neo4j browser queries, pipeline internals, and troubleshooting.
