"""
Neo4j graph store for Graph RAG.
Handles schema creation, node/edge ingestion, vector search, and graph traversal.
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
            s.run(
                "CREATE CONSTRAINT code_entity_name IF NOT EXISTS "
                "FOR (n:CodeEntity) REQUIRE n.name IS UNIQUE"
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
        with self._driver.session() as s:
            for e in entities:
                s.run(
                    """
                    MERGE (n:CodeEntity {name: $name})
                    SET n.entity_type = $entity_type,
                        n.source_code  = $source_code,
                        n.file_path    = $file_path,
                        n.start_line   = $start_line,
                        n.embedding    = $embedding
                    """,
                    name=e.name,
                    entity_type=e.entity_type,
                    source_code=e.source_code,
                    file_path=e.file_path,
                    start_line=e.start_line,
                    embedding=e.embedding,
                )
            for e in entities:
                for callee in e.calls:
                    s.run(
                        """
                        MATCH (a:CodeEntity {name: $caller})
                        MATCH (b:CodeEntity {name: $callee})
                        MERGE (a)-[:CALLS]->(b)
                        """,
                        caller=e.name,
                        callee=callee,
                    )
        logger.info("Graph built: %d entities", len(entities))

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def vector_search(
        self, query_embedding: List[float], top_k: int = VECTOR_TOP_K
    ) -> List[Dict[str, Any]]:
        with self._driver.session() as s:
            result = s.run(
                f"CALL db.index.vector.queryNodes('{VECTOR_INDEX_NAME}', $top_k, $embedding) "
                "YIELD node, score "
                "RETURN node.name AS name, node.entity_type AS entity_type, "
                "node.source_code AS source_code, node.file_path AS file_path, score",
                top_k=top_k,
                embedding=query_embedding,
            )
            return [dict(r) for r in result]

    def get_callees(self, name: str) -> List[Dict[str, Any]]:
        with self._driver.session() as s:
            result = s.run(
                "MATCH (n:CodeEntity {name: $name})-[:CALLS]->(c:CodeEntity) "
                "RETURN c.name AS name, c.entity_type AS entity_type, "
                "c.source_code AS source_code, c.file_path AS file_path",
                name=name,
            )
            return [dict(r) for r in result]

    def get_callers(self, name: str) -> List[Dict[str, Any]]:
        with self._driver.session() as s:
            result = s.run(
                "MATCH (c:CodeEntity)-[:CALLS]->(n:CodeEntity {name: $name}) "
                "RETURN c.name AS name, c.entity_type AS entity_type, "
                "c.source_code AS source_code, c.file_path AS file_path",
                name=name,
            )
            return [dict(r) for r in result]
