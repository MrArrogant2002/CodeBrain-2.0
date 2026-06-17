"""
FAISS in-memory vector store for Normal RAG.
Builds a flat L2 index from chunk embeddings and returns top-k matches.
"""

from typing import List, Tuple

import faiss
import numpy as np

from normal_rag.code_parser import CodeChunk


class VectorStore:
    def __init__(self, chunks: List[CodeChunk]) -> None:
        self._chunks = chunks
        vecs = np.array([c.embedding for c in chunks], dtype="float32")
        self._index = faiss.IndexFlatL2(vecs.shape[1])
        self._index.add(vecs)

    def search(
        self, query_embedding: List[float], top_k: int
    ) -> List[Tuple[CodeChunk, float]]:
        """Return (chunk, distance) pairs for the top_k nearest chunks."""
        q = np.array([query_embedding], dtype="float32")
        distances, indices = self._index.search(q, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                results.append((self._chunks[idx], float(dist)))
        return results
