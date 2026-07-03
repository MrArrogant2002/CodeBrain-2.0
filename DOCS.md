# CodeBrain — Complete Documentation

A prototype that compares two retrieval strategies for code Q&A:

- **Normal RAG** — FAISS flat vector search. Finds the most similar code chunks by embedding similarity alone.
- **Graph RAG** — Neo4j property graph + CALLS-edge expansion. Finds entry points by similarity, then follows call relationships to assemble the full execution context.

Both systems parse **Python (`.py`) and C (`.c`/`.h`)** codebases and answer questions using a local LLM via Ollama.

**What is included vs. what you add:** only the small Python `sample_codebase/` ships with the repo — all commands work against it out of the box. To analyze a real codebase, copy your own directory of `.py`/`.c`/`.h` files into the project root and pass its name via `--path`. The documentation's examples use `BMS_Source_Code/` (NXP S32K144 battery-management firmware) as the private corpus; that folder is confidential, listed in `.gitignore`, and **not part of the repository** — substitute your own source directory. If your codebase is private too, add its folder name to `.gitignore` before committing.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [File Reference](#2-file-reference)
3. [Prerequisites](#3-prerequisites)
4. [One-Time Setup](#4-one-time-setup)
5. [Running Normal RAG](#5-running-normal-rag)
6. [Running Graph RAG](#6-running-graph-rag)
7. [Running the Comparison](#7-running-the-comparison)
8. [Changing the LLM Model](#8-changing-the-llm-model)
9. [Configuration Reference](#9-configuration-reference)
10. [The Sample Codebase](#10-the-sample-codebase)
11. [Docker & Neo4j](#11-docker--neo4j)
12. [How It Works](#12-how-it-works)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Project Structure

```
CODEBRAIN/
|
|-- sample_codebase/          Shared demo Python files (used by both systems)
|   |-- auth.py
|   |-- database.py
|   |-- api_handler.py
|   `-- data_validator.py
|
|-- BMS_Source_Code/          Private C firmware corpus (S32K144 + MC33771C BCC)
|   |                         NOT in git (.gitignore) -- you supply your own codebase
|   |                         folder here; any directory of .py/.c/.h files works
|   |-- Sources/              Application code (main.c, mc33771c/, uja_sbc/, ...)
|   |-- Generated_Code/       Processor Expert peripheral config
|   |-- SDK/                  NXP platform drivers
|   `-- Debug/                Build output -- automatically skipped by the parsers
|
|-- normal_rag/               Normal RAG system (FAISS, no server required)
|   |-- config.py
|   |-- code_parser.py
|   |-- vector_store.py
|   |-- ollama_client.py
|   `-- run.py
|
|-- graph_rag/                Graph RAG system (Neo4j, requires Docker)
|   |-- config.py
|   |-- code_parser.py
|   |-- graph_store.py
|   |-- hybrid_retriever.py
|   |-- ollama_client.py
|   |-- ingest.py
|   `-- run.py
|
|-- compare.py                Runs both systems side-by-side on the same query
|-- docker-compose.yml        Neo4j 5.15 Enterprise container
|-- .env                      Neo4j credentials (not committed to git)
|-- requirements.txt          All Python dependencies
`-- DOCS.md                   This file
```

---

## 2. File Reference

### `sample_codebase/`

The shared demo codebase. Both systems parse and query these files.

| File | What it contains |
|---|---|
| `auth.py` | `hash_password`, `verify_password`, `generate_token`, `verify_token`, `is_admin` |
| `database.py` | SQLite helpers: `get_connection`, `initialize_schema`, `insert_user`, `get_user_by_username`, `get_user_by_id`, `list_all_users`, `log_action` |
| `api_handler.py` | REST-like handlers: `register_user`, `login`, `get_profile`, `admin_list_users` — these call auth + database + validator |
| `data_validator.py` | `validate_username`, `validate_password`, `validate_role`, `validate_user_input`, `sanitize_string` |

These four files contain real, interconnected call chains — `login` calls `verify_password` which calls `hash_password`, etc. — making them ideal for comparing the two retrieval strategies.

---

### `normal_rag/`

Self-contained Normal RAG pipeline. No server or database required.

| File | Purpose |
|---|---|
| `config.py` | All tunable settings: embedding model, Ollama model, vector top-k |
| `code_parser.py` | Parses `.py`, `.c`, and `.h` files with tree-sitter into `CodeChunk` objects (functions, classes, structs, enums, typedefs); skips build dirs like `Debug/` |
| `vector_store.py` | Wraps a FAISS `IndexFlatL2` — builds the index from chunk embeddings, runs top-k search |
| `ollama_client.py` | Single `ask(prompt, model)` function that calls Ollama and returns the response string |
| `run.py` | Main entry point — CLI with `--path`, `--query`, `--model`, `--top-k` flags; also has interactive mode |

---

### `graph_rag/`

Graph RAG pipeline. Requires Neo4j running via Docker.

| File | Purpose |
|---|---|
| `config.py` | All tunable settings; Neo4j credentials read from environment / `.env` with dev defaults |
| `code_parser.py` | Parses `.py`, `.c`, and `.h` files with tree-sitter into `ParsedEntity` objects (includes `calls` list); skips build dirs like `Debug/` |
| `graph_store.py` | Neo4j wrapper — creates schema, batch-ingests nodes and CALLS edges (identity = `qname` i.e. `file::name`, so same-named C functions in different files stay distinct), runs vector search and graph traversal queries |
| `hybrid_retriever.py` | Orchestrates: embed query → vector search → expand `GRAPH_HOP_DEPTH` hops via CALLS edges → format context string |
| `ollama_client.py` | Single `ask(prompt, model)` function that calls Ollama and returns the response string |
| `ingest.py` | CLI — parse → embed → write to Neo4j. Run once before querying |
| `run.py` | Main entry point — queries Neo4j + generates answer. CLI with `--query`, `--model`, `--top-k`; also has interactive mode |

---

### Root files

| File | Purpose |
|---|---|
| `compare.py` | Runs both Normal RAG and Graph RAG on the same query and prints results side by side with timing |
| `docker-compose.yml` | Defines the `codebrain-neo4j` container (Neo4j 5.15 Enterprise, ports 7687 + 7474) |
| `.env` | Neo4j connection config — not committed to git, do not share |
| `requirements.txt` | All Python dependencies for both systems |
| `DOCS.md` | This file |

---

## 3. Prerequisites

| Tool | Purpose | Install |
|---|---|---|
| Python 3.10+ | Run all code | python.org |
| Docker Desktop | Run Neo4j (Graph RAG only) | docker.com |
| Ollama | Run the local LLM | ollama.com |

Verify each is available:

```bash
python --version
docker --version
ollama --version
```

Make sure **Docker Desktop is open and running** before any Graph RAG step.

### Adding a codebase to analyze

`sample_codebase/` is included and needs no setup. For anything else, copy the source
directory into the project root:

```
CODEBRAIN/
|-- MyFirmware_Source/     <- your folder (any name; .py/.c/.h files, any nesting)
```

Then use `--path MyFirmware_Source` in the commands below. Build directories
(`Debug/`, `Release/`, `build/`, `dist/`) inside it are skipped automatically.
If the code is private, add the folder name to `.gitignore` first.

---

## 4. One-Time Setup

### Step 1 — Create and activate virtual environment

```bash
python -m venv .venv

# Windows (Git Bash / MINGW64):
source .venv/Scripts/activate

# Windows (PowerShell):
.venv\Scripts\Activate.ps1

# macOS / Linux:
source .venv/bin/activate
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Pull a model into Ollama

```bash
ollama pull codellama        # default model (~3.8 GB)
# or
ollama pull qwen2.5-coder:7b  # better at code Q&A (~4.7 GB)
```

List all models you have pulled:

```bash
ollama list
```

---

## 5. Running Normal RAG

Normal RAG needs no database or Docker. It parses `sample_codebase/`, builds a FAISS index in memory, retrieves the top-k most similar chunks, and sends them to Ollama.

### Ask a single question

```bash
python -m normal_rag.run --query "How does the login flow verify a password?"
```

### Interactive mode (ask multiple questions)

```bash
python -m normal_rag.run
```

Type a question and press Enter. Repeat. Press `Ctrl+C` to quit.

### Change the model for one run

```bash
python -m normal_rag.run --query "..." --model qwen2.5-coder:7b
```

### Change the number of retrieved chunks

```bash
python -m normal_rag.run --query "..." --top-k 8
```

### Combine flags

```bash
python -m normal_rag.run --query "What does register_user do?" --model qwen2.5-coder:7b --top-k 8
```

### Query a different codebase (e.g. the BMS firmware)

```bash
python -m normal_rag.run --path BMS_Source_Code --query "How are cell voltages read from the MC33771C?"
```

---

## 6. Running Graph RAG

Graph RAG stores the code as a property graph in Neo4j. You must start Neo4j and run ingestion before querying.

### Step 1 — Start Neo4j

```bash
docker compose up -d neo4j
```

Wait ~30 seconds for the container to become healthy:

```bash
docker ps
# codebrain-neo4j should show: Up ... (healthy)
```

Open the Neo4j browser (optional, to explore the graph visually):
```
http://localhost:7474
Username: neo4j
Password: codebrain2024
```

### Step 2 — Ingest a codebase (run once, or when the source changes)

```bash
# Python demo corpus
python -m graph_rag.ingest --path sample_codebase --clear

# or: real C firmware corpus (~1600 entities, embedding takes a few minutes)
python -m graph_rag.ingest --path BMS_Source_Code --clear
```

Expected output:
```
HH:MM:SS  INFO     Step 1/3  Parsing: sample_codebase
HH:MM:SS  INFO       -> 23 entities extracted
HH:MM:SS  INFO     Step 2/3  Embedding with sentence-transformers ...
HH:MM:SS  INFO       -> embeddings ready (dim=384)
HH:MM:SS  INFO     Step 3/3  Writing to Neo4j ...
HH:MM:SS  INFO     Done: 23 nodes, 88 potential CALLS edges
```

> `--clear` wipes the graph before writing. Omit it to add on top of existing data.

### Step 3 — Ask a question

```bash
python -m graph_rag.run --query "How does the login flow verify a password?"
```

### Interactive mode

```bash
python -m graph_rag.run
```

### Change model or top-k

```bash
python -m graph_rag.run --query "..." --model qwen2.5-coder:7b --top-k 8
```

---

## 7. Running the Comparison

`compare.py` runs both systems on the same question and prints the retrieved context and LLM answer for each, followed by a timing summary.

### Default query (login password verification)

```bash
python compare.py
```

### Custom query

```bash
python compare.py --query "What happens when a user registers?"
```

### Same model for both systems

```bash
python compare.py --query "..." --model qwen2.5-coder:7b
```

### Different model per system

```bash
python compare.py --query "..." --normal-model codellama --graph-model qwen2.5-coder:7b
```

### Change top-k

```bash
python compare.py --query "..." --top-k 8
```

> Graph RAG must have been ingested before running compare. Start Neo4j and run `python -m graph_rag.ingest --path sample_codebase --clear` first.

---

## 8. Changing the LLM Model

There are three ways to change which Ollama model is used, from most permanent to most temporary:

### Option A — Change the default in config (permanent)

Edit `normal_rag/config.py`:
```python
OLLAMA_MODEL = "qwen2.5-coder:7b"   # change this line
```

Edit `graph_rag/config.py`:
```python
OLLAMA_MODEL = "qwen2.5-coder:7b"   # change this line
```

All subsequent runs use the new model unless overridden.

### Option B — Override per run with `--model`

```bash
python -m normal_rag.run --query "..." --model qwen2.5-coder:7b
python -m graph_rag.run  --query "..." --model qwen2.5-coder:7b
python compare.py        --query "..." --model qwen2.5-coder:7b
```

### Option C — Use different models per system in compare

```bash
python compare.py --normal-model codellama --graph-model qwen2.5-coder:7b
```

### Pulling a new model

```bash
ollama pull <model-name>
```

Common models for code Q&A:

| Model | Size | Pull command |
|---|---|---|
| `codellama` | 3.8 GB | `ollama pull codellama` |
| `qwen2.5-coder:7b` | 4.7 GB | `ollama pull qwen2.5-coder:7b` |
| `deepseek-coder:6.7b` | 3.8 GB | `ollama pull deepseek-coder:6.7b` |
| `llama3:8b` | 4.7 GB | `ollama pull llama3:8b` |

---

## 9. Configuration Reference

### `normal_rag/config.py`

| Setting | Default | What it does |
|---|---|---|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model used to embed code chunks and queries |
| `EMBEDDING_DIM` | `384` | Must match the embedding model output dimension |
| `OLLAMA_MODEL` | `codellama` | Default LLM for answer generation |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `VECTOR_TOP_K` | `5` | How many chunks to retrieve per query |

### `graph_rag/config.py`

| Setting | Default | What it does |
|---|---|---|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j Bolt connection string (env var / `.env` override) |
| `NEO4J_USER` | `neo4j` | Neo4j username (env var / `.env` override) |
| `NEO4J_PASSWORD` | `codebrain2024` | Must match `NEO4J_AUTH` in `docker-compose.yml` (env var / `.env` override) |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Must be the same model used during ingestion |
| `EMBEDDING_DIM` | `384` | Must match the embedding model |
| `VECTOR_INDEX_NAME` | `code_entity_embeddings` | Name of the Neo4j vector index |
| `OLLAMA_MODEL` | `codellama` | Default LLM for answer generation |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `VECTOR_TOP_K` | `5` | Initial vector-search entry points |
| `GRAPH_HOP_DEPTH` | `1` | How many CALLS hops to follow from each entry point (2+ expands transitively — context grows fast) |

> If you change `EMBEDDING_MODEL` or `EMBEDDING_DIM`, you must re-run ingestion with `--clear` so the graph and vector index match the new model.

---

## 10. The Sample Codebase

`sample_codebase/` contains four Python files that form a small but realistic auth + API system with deep call chains — ideal for demonstrating the difference between flat vector search and graph-based retrieval.

### Call chain example

```
api_handler.login
  -> data_validator.validate_username
  -> database.get_user_by_username
  -> auth.verify_password
       -> auth.hash_password
  -> auth.generate_token
  -> database.log_action

api_handler.register_user
  -> data_validator.validate_user_input
       -> data_validator.validate_username
       -> data_validator.validate_password
       -> data_validator.validate_role
  -> auth.hash_password
  -> database.insert_user
  -> database.log_action
```

Normal RAG retrieves chunks that are textually similar to the query.
Graph RAG retrieves the entry point by similarity, then follows the arrows above to pull in the full execution path automatically.

---

## 11. Docker & Neo4j

### Start Neo4j

```bash
docker compose up -d neo4j
```

### Stop Neo4j (keeps data)

```bash
docker compose down
```

### Stop Neo4j and delete all data

```bash
docker compose down -v
```

After `down -v`, run ingestion again before querying.

### Check container status

```bash
docker ps
```

### View Neo4j logs

```bash
docker logs codebrain-neo4j
```

### Neo4j Browser

Open `http://localhost:7474` in a browser.
- Username: `neo4j`
- Password: `codebrain2024`

Useful Cypher queries to explore the graph:

```cypher
-- All nodes
MATCH (n:CodeEntity) RETURN n LIMIT 50

-- All CALLS edges
MATCH (a:CodeEntity)-[:CALLS]->(b:CodeEntity) RETURN a, b

-- What does login call?
MATCH (n:CodeEntity {name: "login"})-[:CALLS]->(c) RETURN n, c

-- Who calls verify_password?
MATCH (c)-[:CALLS]->(n:CodeEntity {name: "verify_password"}) RETURN c, n

-- Node identity is qname (file::name); name alone may match several nodes
MATCH (n:CodeEntity) WHERE n.name = "CAN0_ORed_IRQHandler" RETURN n.qname
```

### Changing the Neo4j password

1. Edit `docker-compose.yml`, change `NEO4J_AUTH: neo4j/<newpassword>`
2. Edit `graph_rag/config.py`, change `NEO4J_PASSWORD = "<newpassword>"`
3. Run `docker compose down -v` then `docker compose up -d neo4j` to recreate with new credentials

---

## 12. How It Works

### Normal RAG pipeline

```
sample_codebase/*.py
       |
       v
tree-sitter AST parse
  -> extract function + class nodes
  -> CodeChunk (name, type, source_code, file_path)
       |
       v
sentence-transformers (all-MiniLM-L6-v2)
  -> 384-dim embedding per chunk
       |
       v
FAISS IndexFlatL2 (in memory)
       |
  User query -> embed query -> L2 search -> top-k chunks
       |
       v
Ollama (codellama)
  prompt = context chunks + question
       |
       v
Answer
```

### Graph RAG pipeline

```
INGESTION (once):
sample_codebase/*.py
       |
       v
tree-sitter AST parse
  -> extract functions + classes + CALLS relationships
  -> ParsedEntity (name, type, source_code, file_path, calls=[...])
       |
       v
sentence-transformers -> 384-dim embedding per entity
       |
       v
Neo4j
  MERGE (:CodeEntity) nodes with embedding property
  MERGE (:CodeEntity)-[:CALLS]->(:CodeEntity) edges
  CREATE VECTOR INDEX on embedding (cosine, 384-dim)

QUERY (each time):
User query
       |
       v
embed query -> Neo4j vector search -> top-k entry points
       |
       v
For each entry point:
  MATCH (entry)-[:CALLS]->(callee)   [what it calls]
  MATCH (caller)-[:CALLS]->(entry)   [what calls it]
       |
       v
Deduplicated entity set -> formatted context string
  [VECTOR HIT] = most relevant entity
  [CALLEE]     = functions it calls
  [CALLER]     = functions that call it
       |
       v
Ollama (codellama)
  prompt = structured context + question
       |
       v
Answer
```

---

## 13. Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'normal_rag'` | Running from wrong directory | Run from the `CODEBRAIN/` root, not from inside a subfolder |
| `ModuleNotFoundError: No module named 'tree_sitter_python'` | Missing package | `pip install tree-sitter tree-sitter-python` |
| `ModuleNotFoundError: No module named 'tree_sitter_c'` | Missing package | `pip install tree-sitter-c` |
| `RuntimeError: Ollama error` | Ollama not running | Start Ollama: run `ollama serve` in a separate terminal |
| `Model not found` | Model not pulled | `ollama pull <model-name>` |
| `AuthError: The client is unauthorized` | Wrong Neo4j password | Check `NEO4J_PASSWORD` in `graph_rag/config.py` matches `docker-compose.yml` |
| `ServiceUnavailable: Failed to establish connection` | Neo4j not running | `docker compose up -d neo4j`, wait 30s |
| `No results returned` | Graph not ingested | `python -m graph_rag.ingest --path sample_codebase --clear` |
| `UnicodeEncodeError: charmap codec` | Windows terminal encoding | Run `set PYTHONIOENCODING=utf-8` in CMD, or `$env:PYTHONIOENCODING="utf-8"` in PowerShell, or use Git Bash |
| `No entities found` | Wrong working directory | Run all commands from `CODEBRAIN/` root |
