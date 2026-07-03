"""
Parses Python and C source files into graph-ready entities using tree-sitter.
Extracts functions, classes, structs/typedefs, and the CALLS relationships
between them. Entity identity is the qualified name (file_path::name) so that
same-named functions in different files — common in C — stay distinct.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import tree_sitter_c as tsc
import tree_sitter_python as tspython
from sentence_transformers import SentenceTransformer
from tree_sitter import Language, Parser

from graph_rag.config import EMBEDDING_MODEL

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
class ParsedEntity:
    name: str
    entity_type: str  # "function" | "class" | "struct" | "enum" | "typedef"
    source_code: str
    file_path: str  # relative to the ingested root
    start_line: int
    calls: List[str] = field(default_factory=list)  # names this entity calls
    embedding: List[float] = field(default_factory=list)

    @property
    def qname(self) -> str:
        """Graph identity — keeps same-named entities in different files distinct."""
        return f"{self.file_path}::{self.name}"


def parse_directory(directory: str) -> List[ParsedEntity]:
    """Walk a directory and extract all entities from .py, .c, and .h files."""
    root = Path(directory)
    entities: List[ParsedEntity] = []
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
                file_entities = _extract_python(tree.root_node, source, rel_path)
            else:
                file_entities = _extract_c(tree.root_node, source, rel_path)
            entities.extend(file_entities)
            logger.debug("Parsed %s: %d entities", rel_path, len(file_entities))
        except Exception as exc:
            logger.warning("Skipping %s: %s", path, exc)

    logger.info("Extracted %d entities from %s", len(entities), directory)
    return entities


def embed_entities(entities: List[ParsedEntity]) -> List[ParsedEntity]:
    """Populate the embedding field on each entity in-place."""
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [f"{e.entity_type} {e.name}\n{e.source_code}" for e in entities]
    vectors = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    for entity, vec in zip(entities, vectors):
        entity.embedding = vec.tolist()
    return entities


# ── Python extraction ──────────────────────────────────────────────────────────


def _extract_python(root_node, source: bytes, file_path: str) -> List[ParsedEntity]:
    entities: List[ParsedEntity] = []
    for node in root_node.children:
        if node.type == "function_definition":
            entities.append(_parse_py_function(node, source, file_path))
        elif node.type == "class_definition":
            entities.append(_parse_py_class(node, source, file_path))
            body = node.child_by_field_name("body")
            if body:
                for child in body.children:
                    if child.type == "function_definition":
                        entities.append(_parse_py_function(child, source, file_path))
    return entities


def _parse_py_function(node, source: bytes, file_path: str) -> ParsedEntity:
    name_node = node.child_by_field_name("name")
    name = (
        source[name_node.start_byte : name_node.end_byte].decode("utf-8")
        if name_node
        else "<unknown>"
    )
    src = source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
    return ParsedEntity(
        name=name,
        entity_type="function",
        source_code=src,
        file_path=file_path,
        start_line=node.start_point[0] + 1,
        calls=_extract_py_calls(node, source),
    )


def _parse_py_class(node, source: bytes, file_path: str) -> ParsedEntity:
    name_node = node.child_by_field_name("name")
    name = (
        source[name_node.start_byte : name_node.end_byte].decode("utf-8")
        if name_node
        else "<unknown>"
    )
    src = source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
    return ParsedEntity(
        name=name,
        entity_type="class",
        source_code=src,
        file_path=file_path,
        start_line=node.start_point[0] + 1,
    )


def _extract_py_calls(node, source: bytes) -> List[str]:
    """Depth-first walk to collect all called names inside a node."""
    called: List[str] = []

    def walk(n) -> None:
        if n.type == "call":
            func_node = n.child_by_field_name("function")
            if func_node:
                if func_node.type == "identifier":
                    called.append(
                        source[func_node.start_byte : func_node.end_byte].decode(
                            "utf-8"
                        )
                    )
                elif func_node.type == "attribute":
                    attr = func_node.child_by_field_name("attribute")
                    if attr:
                        called.append(
                            source[attr.start_byte : attr.end_byte].decode("utf-8")
                        )
        for child in n.children:
            walk(child)

    walk(node)
    return list(dict.fromkeys(called))  # deduplicate, preserve order


# ── C extraction ───────────────────────────────────────────────────────────────

_C_TYPE_KIND = {
    "struct_specifier": "struct",
    "union_specifier": "struct",
    "enum_specifier": "enum",
}


def _extract_c(root_node, source: bytes, file_path: str) -> List[ParsedEntity]:
    """Collect function definitions and named type definitions.

    Recurses through preprocessor blocks (#ifdef/#if) since C files routinely
    wrap definitions in them. Prototypes in headers are plain declarations and
    are deliberately skipped — only definitions become entities.
    """
    entities: List[ParsedEntity] = []

    def visit(node) -> None:
        if node.type == "function_definition":
            entity = _parse_c_function(node, source, file_path)
            if entity:
                entities.append(entity)
            return
        if node.type == "type_definition":
            entity = _parse_c_typedef(node, source, file_path)
            if entity:
                entities.append(entity)
            return
        if node.type in _C_TYPE_KIND:
            entity = _parse_c_named_type(node, source, file_path)
            if entity:
                entities.append(entity)
            return
        for child in node.children:
            visit(child)

    visit(root_node)
    return entities


def _parse_c_function(node, source: bytes, file_path: str) -> Optional[ParsedEntity]:
    name = _c_declarator_name(node.child_by_field_name("declarator"), source)
    if not name:
        return None
    src = source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
    return ParsedEntity(
        name=name,
        entity_type="function",
        source_code=src,
        file_path=file_path,
        start_line=node.start_point[0] + 1,
        calls=_extract_c_calls(node, source),
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


def _parse_c_typedef(node, source: bytes, file_path: str) -> Optional[ParsedEntity]:
    decl = node.child_by_field_name("declarator")
    if decl is None or decl.type != "type_identifier":
        return None  # function-pointer typedefs etc. — skip
    name = source[decl.start_byte : decl.end_byte].decode("utf-8")
    type_node = node.child_by_field_name("type")
    kind = _C_TYPE_KIND.get(type_node.type, "typedef") if type_node else "typedef"
    src = source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
    return ParsedEntity(
        name=name,
        entity_type=kind,
        source_code=src,
        file_path=file_path,
        start_line=node.start_point[0] + 1,
    )


def _parse_c_named_type(node, source: bytes, file_path: str) -> Optional[ParsedEntity]:
    name_node = node.child_by_field_name("name")
    body = node.child_by_field_name("body")
    if name_node is None or body is None:
        return None  # forward reference like `struct foo *p` — not a definition
    name = source[name_node.start_byte : name_node.end_byte].decode("utf-8")
    src = source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
    return ParsedEntity(
        name=name,
        entity_type=_C_TYPE_KIND[node.type],
        source_code=src,
        file_path=file_path,
        start_line=node.start_point[0] + 1,
    )


def _extract_c_calls(node, source: bytes) -> List[str]:
    """Collect direct call names inside a node (function-pointer calls excluded)."""
    called: List[str] = []

    def walk(n) -> None:
        if n.type == "call_expression":
            func_node = n.child_by_field_name("function")
            if func_node is not None and func_node.type == "identifier":
                called.append(
                    source[func_node.start_byte : func_node.end_byte].decode("utf-8")
                )
        for child in n.children:
            walk(child)

    walk(node)
    return list(dict.fromkeys(called))
