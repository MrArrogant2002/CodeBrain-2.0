"""
Neo4j graph store for Graph RAG.
Handles schema creation, node/edge ingestion, vector search, and graph traversal.

Node identity is `qname` (file_path::name) — bare names collide across files in
C codebases, where every module can define its own `static` helper with the
same name. CALLS edges are resolved from bare call names before ingestion,
preferring a target in the caller's own file (approximates C static linkage).
"""

import logging
from typing import Any, Dict, List

from neo4j import GraphDatabase

from graph_rag.code_parser import ParsedEntity
from graph_rag.config import (
    EMBEDDING_DIM,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    VECTOR_INDEX_NAME,
    VECTOR_TOP_K,
)

logger = logging.getLogger(__name__)

_NODE_BATCH = 200  # rows per UNWIND write (embeddings make rows heavy)
_EDGE_BATCH = 2000


class GraphStore:
    def __init__(self) -> None:
        self._driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> "GraphStore":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ── Schema ─────────────────────────────────────────────────────────────────

    def create_schema(self) -> None:
        with self._driver.session() as s:
            # identity moved from name to qname — drop the legacy constraint
            s.run("DROP CONSTRAINT code_entity_name IF EXISTS")
            s.run(
                "CREATE CONSTRAINT code_entity_qname IF NOT EXISTS "
                "FOR (n:CodeEntity) REQUIRE n.qname IS UNIQUE"
            )
            s.run(
                f"CREATE VECTOR INDEX `{VECTOR_INDEX_NAME}` IF NOT EXISTS "
                f"FOR (n:CodeEntity) ON (n.embedding) "
                f"OPTIONS {{indexConfig: {{`vector.dimensions`: {EMBEDDING_DIM}, "
                f"`vector.similarity_function`: 'cosine'}}}}"
            )
        logger.info("Schema ready")

    def clear(self) -> None:
        with self._driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")
        logger.info("Graph cleared")

    # ── Ingestion ──────────────────────────────────────────────────────────────

    def build_graph(self, entities: List[ParsedEntity]) -> None:
        rows = [
            {
                "qname": e.qname,
                "name": e.name,
                "entity_type": e.entity_type,
                "source_code": e.source_code,
                "file_path": e.file_path,
                "start_line": e.start_line,
                "embedding": e.embedding,
            }
            for e in entities
        ]
        edges = _resolve_calls(entities)

        with self._driver.session() as s:
            for i in range(0, len(rows), _NODE_BATCH):
                s.run(
                    """
                    UNWIND $rows AS row
                    MERGE (n:CodeEntity {qname: row.qname})
                    SET n.name        = row.name,
                        n.entity_type = row.entity_type,
                        n.source_code = row.source_code,
                        n.file_path   = row.file_path,
                        n.start_line  = row.start_line,
                        n.embedding   = row.embedding
                    """,
                    rows=rows[i : i + _NODE_BATCH],
                )
            for i in range(0, len(edges), _EDGE_BATCH):
                s.run(
                    """
                    UNWIND $pairs AS p
                    MATCH (a:CodeEntity {qname: p.src})
                    MATCH (b:CodeEntity {qname: p.dst})
                    MERGE (a)-[:CALLS]->(b)
                    """,
                    pairs=edges[i : i + _EDGE_BATCH],
                )
        logger.info(
            "Graph built: %d entities, %d CALLS edges", len(entities), len(edges)
        )

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def vector_search(
        self, query_embedding: List[float], top_k: int = VECTOR_TOP_K
    ) -> List[Dict[str, Any]]:
        with self._driver.session() as s:
            result = s.run(
                f"CALL db.index.vector.queryNodes('{VECTOR_INDEX_NAME}', $top_k, $embedding) "
                "YIELD node, score "
                "RETURN node.qname AS qname, node.name AS name, "
                "node.entity_type AS entity_type, node.source_code AS source_code, "
                "node.file_path AS file_path, score",
                top_k=top_k,
                embedding=query_embedding,
            )
            return [dict(r) for r in result]

    def get_callees(self, qname: str) -> List[Dict[str, Any]]:
        with self._driver.session() as s:
            result = s.run(
                "MATCH (n:CodeEntity {qname: $qname})-[:CALLS]->(c:CodeEntity) "
                "RETURN c.qname AS qname, c.name AS name, "
                "c.entity_type AS entity_type, c.source_code AS source_code, "
                "c.file_path AS file_path",
                qname=qname,
            )
            return [dict(r) for r in result]

    def get_callers(self, qname: str) -> List[Dict[str, Any]]:
        with self._driver.session() as s:
            result = s.run(
                "MATCH (c:CodeEntity)-[:CALLS]->(n:CodeEntity {qname: $qname}) "
                "RETURN c.qname AS qname, c.name AS name, "
                "c.entity_type AS entity_type, c.source_code AS source_code, "
                "c.file_path AS file_path",
                qname=qname,
            )
            return [dict(r) for r in result]


# ── helpers ────────────────────────────────────────────────────────────────────


def _resolve_calls(entities: List[ParsedEntity]) -> List[Dict[str, str]]:
    """Resolve bare callee names to node identities.

    A callee name defined in the caller's own file wins (C static semantics);
    otherwise every definition of that name gets an edge, since the true
    target cannot be known without a full preprocessor + linker pass.
    Names with no definition in the corpus (stdlib, vendor SDK externals)
    are dropped.
    """
    by_name: Dict[str, List[ParsedEntity]] = {}
    for e in entities:
        by_name.setdefault(e.name, []).append(e)

    pairs: List[Dict[str, str]] = []
    seen: set = set()
    for e in entities:
        for callee in e.calls:
            candidates = by_name.get(callee)
            if not candidates:
                continue
            same_file = [c for c in candidates if c.file_path == e.file_path]
            for target in same_file or candidates:
                key = (e.qname, target.qname)
                if key not in seen:
                    seen.add(key)
                    pairs.append({"src": e.qname, "dst": target.qname})
    return pairs
