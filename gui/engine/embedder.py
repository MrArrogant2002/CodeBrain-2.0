
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_embeddings(chunks):
    texts = [c["code"] for c in chunks]
    embeddings = model.encode(texts)
    return embeddings
