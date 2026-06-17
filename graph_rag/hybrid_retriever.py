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
        2. Expand each entry point via CALLS edges (callees + callers)
        3. Format and return a single context string
        """
        query_vec = self._model.encode(query).tolist()
        hits = self._store.vector_search(query_vec, top_k=top_k)

        if not hits:
            logger.warning("Vector search returned no results")
            return ""

        seen: set = set()
        sections: List[str] = []

        for hit in hits:
            name = hit["name"]
            if name not in seen:
                seen.add(name)
                sections.append(_format_entity(hit, label="[VECTOR HIT]"))

            if GRAPH_HOP_DEPTH >= 1:
                for callee in self._store.get_callees(name):
                    if callee["name"] not in seen:
                        seen.add(callee["name"])
                        sections.append(_format_entity(callee, label="[CALLEE]"))

                for caller in self._store.get_callers(name):
                    if caller["name"] not in seen:
                        seen.add(caller["name"])
                        sections.append(_format_entity(caller, label="[CALLER]"))

        logger.info("Retrieved %d entities (%d vector hits)", len(seen), len(hits))
        return "\n\n".join(sections)


def _format_entity(entity: dict, label: str = "") -> str:
    header = (
        f"{label} {entity['entity_type'].upper()}: {entity['name']}"
        f"  ({entity['file_path']})"
    )
    return f"{header}\n{'-' * len(header)}\n{entity['source_code']}"
