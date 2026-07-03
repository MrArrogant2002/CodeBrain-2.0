"""
Normal RAG — run script.

Parses a codebase (Python or C), builds a FAISS index, retrieves the top-k
most similar chunks for a query, and generates an answer with Ollama.

Usage:
    python -m normal_rag.run
    python -m normal_rag.run --query "How does login verify a password?"
    python -m normal_rag.run --query "..." --model qwen2.5-coder:7b
    python -m normal_rag.run --top-k 8
    python -m normal_rag.run --path BMS_Source_Code --query "How are cell voltages read?"
"""

import argparse
import logging
import sys

from sentence_transformers import SentenceTransformer

from normal_rag import ollama_client
from normal_rag.code_parser import embed_chunks, parse_directory
from normal_rag.config import EMBEDDING_MODEL, OLLAMA_MODEL, VECTOR_TOP_K
from normal_rag.vector_store import VectorStore

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

DEFAULT_CODEBASE_PATH = "sample_codebase"
SEP = "-" * 70


def build_context(
    query: str, store: VectorStore, embed_model: SentenceTransformer, top_k: int
) -> str:
    """Retrieve top-k chunks and format them as a context string."""
    q_vec = embed_model.encode(query).tolist()
    hits = store.search(q_vec, top_k)

    sections = []
    for rank, (chunk, _dist) in enumerate(hits, 1):
        header = f"[{rank}] {chunk.chunk_type.upper()}: {chunk.name}  ({chunk.file_path}:{chunk.start_line})"
        sections.append(f"{header}\n{chunk.source_code}")
    return "\n\n".join(sections)


def run(query: str, model: str, top_k: int, path: str = DEFAULT_CODEBASE_PATH) -> str:
    """Full pipeline: parse → embed → search → generate."""
    print(f"\n[Normal RAG] Parsing and embedding {path}/ ...")
    chunks = parse_directory(path)
    if not chunks:
        print(f"ERROR: No code chunks found in '{path}'. Run from the project root.")
        sys.exit(1)

    chunks = embed_chunks(chunks)
    store = VectorStore(chunks)
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"\n[Normal RAG] Query : {query}")
    print(f"[Normal RAG] Model : {model}")
    print(f"[Normal RAG] Top-K : {top_k}\n")

    context = build_context(query, store, embed_model, top_k)

    print(SEP)
    print("Retrieved context:")
    print(SEP)
    print(context)
    print(SEP)

    prompt = (
        "You are a code assistant. Use only the provided code context to answer the question.\n\n"
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
        description="Normal RAG — query the sample codebase"
    )
    parser.add_argument(
        "--query", default="", help="Question to ask (omit for interactive mode)"
    )
    parser.add_argument(
        "--path",
        default=DEFAULT_CODEBASE_PATH,
        help=f"Directory of source files to query (default: {DEFAULT_CODEBASE_PATH})",
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
        help=f"Number of chunks to retrieve (default: {VECTOR_TOP_K})",
    )
    args = parser.parse_args()

    query = args.query.strip()
    if not query:
        print("Normal RAG — Interactive mode  (Ctrl+C to quit)\n")
        try:
            while True:
                query = input("Your question: ").strip()
                if query:
                    run(query, args.model, args.top_k, args.path)
                    print()
        except KeyboardInterrupt:
            print("\nBye.")
    else:
        run(query, args.model, args.top_k, args.path)


if __name__ == "__main__":
    main()
