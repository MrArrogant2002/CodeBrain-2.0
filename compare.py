"""
Side-by-side comparison: Normal RAG vs Graph RAG.

Both systems run on the same query and the same sample_codebase/.
Results are printed side by side with retrieval and LLM timing.

Usage:
    python compare.py
    python compare.py --query "How does login verify a password?"
    python compare.py --query "..." --model qwen2.5-coder:7b
    python compare.py --query "..." --normal-model codellama --graph-model qwen2.5-coder:7b
    python compare.py --top-k 8
"""

import argparse
import logging
import sys
import time
from typing import List

from sentence_transformers import SentenceTransformer

from graph_rag import ollama_client as graph_ollama
from graph_rag.config import OLLAMA_MODEL as GRAPH_DEFAULT_MODEL
from graph_rag.config import VECTOR_TOP_K
from graph_rag.graph_store import GraphStore
from graph_rag.hybrid_retriever import HybridRetriever
from normal_rag import ollama_client as normal_ollama
from normal_rag.code_parser import CodeChunk, embed_chunks, parse_directory
from normal_rag.config import EMBEDDING_MODEL
from normal_rag.config import OLLAMA_MODEL as NORMAL_DEFAULT_MODEL
from normal_rag.vector_store import VectorStore

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

CODEBASE_PATH = "sample_codebase"
WIDE = "=" * 72
THIN = "-" * 72


# ── Normal RAG ─────────────────────────────────────────────────────────────────


def _normal_retrieve(
    query: str, chunks: List[CodeChunk], store: VectorStore, top_k: int
) -> str:
    model = SentenceTransformer(EMBEDDING_MODEL)
    q_vec = model.encode(query).tolist()
    hits = store.search(q_vec, top_k)
    sections = []
    for rank, (chunk, _) in enumerate(hits, 1):
        header = (
            f"[{rank}] {chunk.chunk_type.upper()}: {chunk.name}  ({chunk.file_path})"
        )
        sections.append(f"{header}\n{chunk.source_code}")
    return "\n\n".join(sections)


# ── Graph RAG ──────────────────────────────────────────────────────────────────


def _graph_retrieve(query: str, top_k: int) -> str:
    with GraphStore() as store:
        retriever = HybridRetriever(store)
        return retriever.retrieve(query, top_k=top_k)


# ── Comparison ─────────────────────────────────────────────────────────────────


def compare(query: str, normal_model: str, graph_model: str, top_k: int) -> None:
    print(f"\n{WIDE}")
    print("  NORMAL RAG  vs  GRAPH RAG -- Side-by-Side Comparison")
    print(WIDE)
    print(f"  Query        : {query}")
    print(f"  Normal model : {normal_model}")
    print(f"  Graph  model : {graph_model}")
    print(f"  Top-K        : {top_k}")
    print(WIDE)

    # ── shared parse + embed (Normal RAG needs it; Graph RAG uses Neo4j)
    print("\n[*] Parsing and embedding sample_codebase/ for Normal RAG ...")
    chunks = parse_directory(CODEBASE_PATH)
    if not chunks:
        print(
            f"ERROR: No chunks found in '{CODEBASE_PATH}'. Run from the project root."
        )
        sys.exit(1)
    chunks = embed_chunks(chunks)
    normal_store = VectorStore(chunks)

    # ── NORMAL RAG ──────────────────────────────────────────────────────────────
    print(f"\n{THIN}")
    print("  NORMAL RAG  (FAISS flat vector search -- no graph)")
    print(THIN)

    t0 = time.perf_counter()
    normal_context = _normal_retrieve(query, chunks, normal_store, top_k)
    normal_ret_t = time.perf_counter() - t0

    print(f"\n[Retrieved {top_k} chunks in {normal_ret_t:.3f}s]\n")
    print(normal_context)

    normal_prompt = (
        "You are a code assistant. Use only the provided code context to answer the question.\n\n"
        f"=== CODE CONTEXT ===\n{normal_context}\n\n"
        f"=== QUESTION ===\n{query}\n\n"
        "=== ANSWER ==="
    )
    print(f"\n[Generating answer with {normal_model} ...]\n")
    t0 = time.perf_counter()
    normal_answer = normal_ollama.ask(normal_prompt, model=normal_model)
    normal_llm_t = time.perf_counter() - t0

    print(f"[Normal RAG answer -- {normal_llm_t:.1f}s]\n")
    print(normal_answer)

    # ── GRAPH RAG ───────────────────────────────────────────────────────────────
    print(f"\n{THIN}")
    print("  GRAPH RAG  (Neo4j vector search + CALLS edge expansion)")
    print(THIN)

    t0 = time.perf_counter()
    try:
        graph_context = _graph_retrieve(query, top_k)
    except Exception as exc:
        print(f"ERROR: Could not reach Neo4j -- {exc}")
        print("Start Neo4j: docker compose up -d neo4j")
        print("Then ingest: python -m graph_rag.ingest --path sample_codebase --clear")
        sys.exit(1)
    graph_ret_t = time.perf_counter() - t0

    graph_entity_count = (
        graph_context.count("[VECTOR HIT]")
        + graph_context.count("[CALLEE]")
        + graph_context.count("[CALLER]")
    )
    print(f"\n[Retrieved {graph_entity_count} entities in {graph_ret_t:.3f}s]\n")
    print(graph_context)

    graph_prompt = (
        "You are a code assistant. Use only the provided code context to answer the question.\n"
        "[VECTOR HIT] = most relevant entity, [CALLEE] = what it calls, [CALLER] = what calls it.\n\n"
        f"=== CODE CONTEXT ===\n{graph_context}\n\n"
        f"=== QUESTION ===\n{query}\n\n"
        "=== ANSWER ==="
    )
    print(f"\n[Generating answer with {graph_model} ...]\n")
    t0 = time.perf_counter()
    graph_answer = graph_ollama.ask(graph_prompt, model=graph_model)
    graph_llm_t = time.perf_counter() - t0

    print(f"[Graph RAG answer -- {graph_llm_t:.1f}s]\n")
    print(graph_answer)

    # ── SUMMARY ─────────────────────────────────────────────────────────────────
    print(f"\n{WIDE}")
    print("  SUMMARY")
    print(WIDE)
    print(
        f"  Normal RAG : {top_k} chunks       | retrieval {normal_ret_t:.3f}s | LLM {normal_llm_t:.1f}s"
    )
    print(
        f"  Graph  RAG : {graph_entity_count} entities    | retrieval {graph_ret_t:.3f}s | LLM {graph_llm_t:.1f}s"
    )
    print(WIDE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Normal RAG vs Graph RAG comparison")
    parser.add_argument(
        "--query",
        default="How does the login flow verify a user's password?",
        help="Question to ask both systems",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Ollama model for both systems (overrides --normal-model and --graph-model)",
    )
    parser.add_argument(
        "--normal-model",
        default=NORMAL_DEFAULT_MODEL,
        help=f"Ollama model for Normal RAG (default: {NORMAL_DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--graph-model",
        default=GRAPH_DEFAULT_MODEL,
        help=f"Ollama model for Graph RAG (default: {GRAPH_DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=VECTOR_TOP_K,
        help=f"Number of vector-search entry points (default: {VECTOR_TOP_K})",
    )
    args = parser.parse_args()

    normal_model = args.model or args.normal_model
    graph_model = args.model or args.graph_model

    compare(args.query, normal_model, graph_model, args.top_k)


if __name__ == "__main__":
    main()
