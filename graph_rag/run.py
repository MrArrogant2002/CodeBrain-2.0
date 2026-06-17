"""
Graph RAG — run script.

Queries Neo4j using vector search + CALLS edge expansion,
then generates an answer with Ollama.

Run ingest.py first to populate the graph.

Usage:
    python -m graph_rag.run
    python -m graph_rag.run --query "How does login verify a password?"
    python -m graph_rag.run --query "..." --model qwen2.5-coder:7b
    python -m graph_rag.run --top-k 8
"""

import argparse
import logging
import sys

from graph_rag import ollama_client
from graph_rag.config import OLLAMA_MODEL, VECTOR_TOP_K
from graph_rag.graph_store import GraphStore
from graph_rag.hybrid_retriever import HybridRetriever

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

SEP = "-" * 70


def run(query: str, model: str, top_k: int) -> str:
    """Retrieve context from Neo4j and generate an answer."""
    print(f"\n[Graph RAG] Query : {query}")
    print(f"[Graph RAG] Model : {model}")
    print(f"[Graph RAG] Top-K : {top_k}\n")

    try:
        with GraphStore() as store:
            retriever = HybridRetriever(store)
            context = retriever.retrieve(query, top_k=top_k)
    except Exception as exc:
        print(f"ERROR: Could not connect to Neo4j — {exc}")
        print("Make sure Neo4j is running: docker compose up -d neo4j")
        print(
            "Then ingest the codebase:   python -m graph_rag.ingest --path sample_codebase --clear"
        )
        sys.exit(1)

    if not context:
        print(
            "No results returned. Run ingest first: python -m graph_rag.ingest --path sample_codebase --clear"
        )
        sys.exit(1)

    print(SEP)
    print("Retrieved context:")
    print(SEP)
    print(context)
    print(SEP)

    prompt = (
        "You are a code assistant. Use only the provided code context to answer the question.\n"
        "The context is labelled: [VECTOR HIT] = most relevant entity, "
        "[CALLEE] = functions it calls, [CALLER] = functions that call it.\n\n"
        f"=== CODE CONTEXT ===\n{context}\n\n"
        f"=== QUESTION ===\n{query}\n\n"
        "=== ANSWER ==="
    )

    print("\nGenerating answer ...\n")
    answer = ollama_client.ask(prompt, model=model)

    print(SEP)
    print("Answer:")
    print(SEP)
    print(answer)
    print(SEP)
    return answer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Graph RAG — query the Neo4j code graph"
    )
    parser.add_argument(
        "--query", default="", help="Question to ask (omit for interactive mode)"
    )
    parser.add_argument(
        "--model",
        default=OLLAMA_MODEL,
        help=f"Ollama model name (default: {OLLAMA_MODEL})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=VECTOR_TOP_K,
        help=f"Vector search entry points (default: {VECTOR_TOP_K})",
    )
    args = parser.parse_args()

    query = args.query.strip()
    if not query:
        print("Graph RAG — Interactive mode  (Ctrl+C to quit)\n")
        try:
            while True:
                query = input("Your question: ").strip()
                if query:
                    run(query, args.model, args.top_k)
                    print()
        except KeyboardInterrupt:
            print("\nBye.")
    else:
        run(query, args.model, args.top_k)


if __name__ == "__main__":
    main()
