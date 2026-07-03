"""
Parses Python and C source files into embeddable code chunks using tree-sitter.
Extracts functions, classes, and named type definitions (structs/enums/typedefs).
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import tree_sitter_c as tsc
import tree_sitter_python as tspython
from sentence_transformers import SentenceTransformer
from tree_sitter import Language, Parser

from normal_rag.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_PY_LANGUAGE = Language(tspython.language())
_C_LANGUAGE = Language(tsc.language())

_LANGUAGE_BY_SUFFIX = {
    ".py": _PY_LANGUAGE,
    ".c": _C_LANGUAGE,
    ".h": _C_LANGUAGE,
}

# Build output and tooling directories — never source to ingest
_SKIP_DIRS = {
    "Debug",
    "Release",
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    "build",
    "dist",
}


@dataclass
class CodeChunk:
    name: str
    chunk_type: str  # "function" | "class" | "struct" | "enum" | "typedef"
    source_code: str
    file_path: str  # relative to the parsed root
    start_line: int
    embedding: List[float] = field(default_factory=list)


def parse_directory(directory: str) -> List[CodeChunk]:
    """Walk a directory and extract all chunks from .py, .c, and .h files."""
    root = Path(directory)
    chunks: List[CodeChunk] = []
    parsers = {suffix: Parser(lang) for suffix, lang in _LANGUAGE_BY_SUFFIX.items()}

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in parsers:
            continue
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        rel_path = path.relative_to(root).as_posix()
        try:
            source = path.read_bytes()
            tree = parsers[path.suffix].parse(source)
            if path.suffix == ".py":
                file_chunks = _extract_python(tree.root_node, source, rel_path)
            else:
                file_chunks = _extract_c(tree.root_node, source, rel_path)
            chunks.extend(file_chunks)
            logger.debug("Parsed %s: %d chunks", rel_path, len(file_chunks))
        except Exception as exc:
            logger.warning("Skipping %s: %s", path, exc)

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


# ── Python extraction ──────────────────────────────────────────────────────────


def _extract_python(root_node, source: bytes, file_path: str) -> List[CodeChunk]:
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


# ── C extraction ───────────────────────────────────────────────────────────────

_C_TYPE_KIND = {
    "struct_specifier": "struct",
    "union_specifier": "struct",
    "enum_specifier": "enum",
}


def _extract_c(root_node, source: bytes, file_path: str) -> List[CodeChunk]:
    """Collect function definitions and named type definitions.

    Recurses through preprocessor blocks (#ifdef/#if) since C files routinely
    wrap definitions in them. Prototypes in headers are plain declarations and
    are deliberately skipped — only definitions become chunks.
    """
    chunks: List[CodeChunk] = []

    def visit(node) -> None:
        if node.type == "function_definition":
            chunk = _parse_c_function(node, source, file_path)
            if chunk:
                chunks.append(chunk)
            return
        if node.type == "type_definition":
            chunk = _parse_c_typedef(node, source, file_path)
            if chunk:
                chunks.append(chunk)
            return
        if node.type in _C_TYPE_KIND:
            chunk = _parse_c_named_type(node, source, file_path)
            if chunk:
                chunks.append(chunk)
            return
        for child in node.children:
            visit(child)

    visit(root_node)
    return chunks


def _parse_c_function(node, source: bytes, file_path: str) -> Optional[CodeChunk]:
    name = _c_declarator_name(node.child_by_field_name("declarator"), source)
    if not name:
        return None
    src = source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
    return CodeChunk(
        name=name,
        chunk_type="function",
        source_code=src,
        file_path=file_path,
        start_line=node.start_point[0] + 1,
    )


def _c_declarator_name(node, source: bytes) -> str:
    """Descend through pointer/parenthesized declarators to the function name."""
    while node is not None:
        if node.type == "function_declarator":
            inner = node.child_by_field_name("declarator")
            if inner is not None and inner.type == "identifier":
                return source[inner.start_byte : inner.end_byte].decode("utf-8")
            node = inner
        elif node.type in ("pointer_declarator", "parenthesized_declarator"):
            node = node.child_by_field_name("declarator") or next(
                (c for c in node.children if c.is_named), None
            )
        elif node.type == "identifier":
            return source[node.start_byte : node.end_byte].decode("utf-8")
        else:
            return ""
    return ""


def _parse_c_typedef(node, source: bytes, file_path: str) -> Optional[CodeChunk]:
    decl = node.child_by_field_name("declarator")
    if decl is None or decl.type != "type_identifier":
        return None  # function-pointer typedefs etc. — skip
    name = source[decl.start_byte : decl.end_byte].decode("utf-8")
    type_node = node.child_by_field_name("type")
    kind = _C_TYPE_KIND.get(type_node.type, "typedef") if type_node else "typedef"
    src = source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
    return CodeChunk(
        name=name,
        chunk_type=kind,
        source_code=src,
        file_path=file_path,
        start_line=node.start_point[0] + 1,
    )


def _parse_c_named_type(node, source: bytes, file_path: str) -> Optional[CodeChunk]:
    name_node = node.child_by_field_name("name")
    body = node.child_by_field_name("body")
    if name_node is None or body is None:
        return None  # forward reference like `struct foo *p` — not a definition
    name = source[name_node.start_byte : name_node.end_byte].decode("utf-8")
    src = source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
    return CodeChunk(
        name=name,
        chunk_type=_C_TYPE_KIND[node.type],
        source_code=src,
        file_path=file_path,
        start_line=node.start_point[0] + 1,
    )
