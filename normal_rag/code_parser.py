"""
Parses Python source files into embeddable code chunks using tree-sitter.
Extracts top-level functions, classes, and methods.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import tree_sitter_python as tspython
from sentence_transformers import SentenceTransformer
from tree_sitter import Language, Parser

from normal_rag.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_PY_LANGUAGE = Language(tspython.language())


@dataclass
class CodeChunk:
    name: str
    chunk_type: str  # "function" | "class"
    source_code: str
    file_path: str
    start_line: int
    embedding: List[float] = field(default_factory=list)


def parse_directory(directory: str) -> List[CodeChunk]:
    """Walk a directory and extract all functions and classes from .py files."""
    root = Path(directory)
    chunks: List[CodeChunk] = []
    parser = Parser(_PY_LANGUAGE)

    for py_file in sorted(root.rglob("*.py")):
        try:
            source = py_file.read_bytes()
            tree = parser.parse(source)
            file_chunks = _extract_chunks(tree.root_node, source, str(py_file))
            chunks.extend(file_chunks)
            logger.debug("Parsed %s: %d chunks", py_file.name, len(file_chunks))
        except Exception as exc:
            logger.warning("Skipping %s: %s", py_file, exc)

    logger.info("Extracted %d chunks from %s", len(chunks), directory)
    return chunks


def embed_chunks(chunks: List[CodeChunk]) -> List[CodeChunk]:
    """Embed each chunk's source code in-place using sentence-transformers."""
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [f"{c.chunk_type} {c.name}\n{c.source_code}" for c in chunks]
    vectors = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    for chunk, vec in zip(chunks, vectors):
        chunk.embedding = vec.tolist()
    return chunks


# ── helpers ────────────────────────────────────────────────────────────────────


def _extract_chunks(root_node, source: bytes, file_path: str) -> List[CodeChunk]:
    chunks: List[CodeChunk] = []
    for node in root_node.children:
        if node.type == "function_definition":
            chunks.append(_make_chunk(node, source, file_path, "function"))
        elif node.type == "class_definition":
            chunks.append(_make_chunk(node, source, file_path, "class"))
            body = node.child_by_field_name("body")
            if body:
                for child in body.children:
                    if child.type == "function_definition":
                        chunks.append(_make_chunk(child, source, file_path, "function"))
    return chunks


def _make_chunk(node, source: bytes, file_path: str, chunk_type: str) -> CodeChunk:
    name_node = node.child_by_field_name("name")
    name = (
        source[name_node.start_byte : name_node.end_byte].decode("utf-8")
        if name_node
        else "<unknown>"
    )
    src = source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
    return CodeChunk(
        name=name,
        chunk_type=chunk_type,
        source_code=src,
        file_path=file_path,
        start_line=node.start_point[0] + 1,
    )
