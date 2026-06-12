import faiss
import numpy as np

class VectorStore:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)
        self.data = []

    def add(self, embeddings, chunks):
        self.index.add(np.array(embeddings).astype("float32"))
        self.data.extend(chunks)

    def search(self, query_embedding, k=3):
        D, I = self.index.search(query_embedding, k)
        results = [self.data[idx] for idx in I[0]]
        return results
