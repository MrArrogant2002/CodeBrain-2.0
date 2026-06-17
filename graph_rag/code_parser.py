"""
Parses Python source files into graph-ready entities using tree-sitter.
Extracts functions, classes, and the CALLS relationships between them.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import tree_sitter_python as tspython
from sentence_transformers import SentenceTransformer
from tree_sitter import Language, Parser

from graph_rag.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_PY_LANGUAGE = Language(tspython.language())


@dataclass
class ParsedEntity:
    name: str
    entity_type: str  # "function" | "class"
    source_code: str
    file_path: str
    start_line: int
    calls: List[str] = field(default_factory=list)  # names this entity calls
    embedding: List[float] = field(default_factory=list)


def parse_directory(directory: str) -> List[ParsedEntity]:
    """Walk a directory and extract all entities from .py files."""
    root = Path(directory)
    entities: List[ParsedEntity] = []
    parser = Parser(_PY_LANGUAGE)

    for py_file in sorted(root.rglob("*.py")):
        try:
            source = py_file.read_bytes()
            tree = parser.parse(source)
            file_entities = _extract_entities(tree.root_node, source, str(py_file))
            entities.extend(file_entities)
            logger.debug("Parsed %s: %d entities", py_file.name, len(file_entities))
        except Exception as exc:
            logger.warning("Skipping %s: %s", py_file, exc)

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


# ── helpers ────────────────────────────────────────────────────────────────────


def _extract_entities(root_node, source: bytes, file_path: str) -> List[ParsedEntity]:
    entities: List[ParsedEntity] = []
    for node in root_node.children:
        if node.type == "function_definition":
            entities.append(_parse_function(node, source, file_path))
        elif node.type == "class_definition":
            entities.append(_parse_class(node, source, file_path))
            body = node.child_by_field_name("body")
            if body:
                for child in body.children:
                    if child.type == "function_definition":
                        entities.append(_parse_function(child, source, file_path))
    return entities


def _parse_function(node, source: bytes, file_path: str) -> ParsedEntity:
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
        calls=_extract_calls(node, source),
    )


def _parse_class(node, source: bytes, file_path: str) -> ParsedEntity:
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


def _extract_calls(node, source: bytes) -> List[str]:
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
