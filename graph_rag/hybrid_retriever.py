"""
Hybrid retriever: vector search (entry points) + CALLS edge expansion.
Returns a formatted context string ready to be passed to the LLM.
"""

import logging
from typing import List

from sentence_transformers import SentenceTransformer

from graph_rag.config import EMBEDDING_MODEL, GRAPH_HOP_DEPTH, VECTOR_TOP_K
from graph_rag.graph_store import GraphStore

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(self, store: GraphStore) -> None:
        self._store = store
        self._model = SentenceTransformer(EMBEDDING_MODEL)

    def retrieve(self, query: str, top_k: int = VECTOR_TOP_K) -> str:
        """
        1. Embed query -> vector search -> top_k entry points
        2. Expand GRAPH_HOP_DEPTH hops via CALLS edges (callees + callers)
        3. Format and return a single context string
        """
        query_vec = self._model.encode(query).tolist()
        hits = self._store.vector_search(query_vec, top_k=top_k)

        if not hits:
            logger.warning("Vector search returned no results")
            return ""

        seen: set = set()
        sections: List[str] = []
        frontier: List[str] = []

        for hit in hits:
            if hit["qname"] not in seen:
                seen.add(hit["qname"])
                sections.append(_format_entity(hit, label="[VECTOR HIT]"))
                frontier.append(hit["qname"])

        for _hop in range(GRAPH_HOP_DEPTH):
            next_frontier: List[str] = []
            for qname in frontier:
                for callee in self._store.get_callees(qname):
                    if callee["qname"] not in seen:
                        seen.add(callee["qname"])
                        sections.append(_format_entity(callee, label="[CALLEE]"))
                        next_frontier.append(callee["qname"])

                for caller in self._store.get_callers(qname):
                    if caller["qname"] not in seen:
                        seen.add(caller["qname"])
                        sections.append(_format_entity(caller, label="[CALLER]"))
                        next_frontier.append(caller["qname"])
            frontier = next_frontier
            if not frontier:
                break

        logger.info("Retrieved %d entities (%d vector hits)", len(seen), len(hits))
        return "\n\n".join(sections)


def _format_entity(entity: dict, label: str = "") -> str:
    header = (
        f"{label} {entity['entity_type'].upper()}: {entity['name']}"
        f"  ({entity['file_path']})"
    )
    return f"{header}\n{'-' * len(header)}\n{entity['source_code']}"
