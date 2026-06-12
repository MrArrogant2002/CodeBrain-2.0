
from gui.engine.repo_loader import load_c_files
from gui.engine.embedder import generate_embeddings
from gui.engine.vector_store import VectorStore
from sentence_transformers import SentenceTransformer
from gui.engine.llm_reasoner import ask_llm

repo_path = "gui/sample_bms_repo"

chunks = load_c_files(repo_path)
embeddings = generate_embeddings(chunks)

dimension = len(embeddings[0])
vector_db = VectorStore(dimension)
vector_db.add(embeddings, chunks)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def query_system(question):
    q_embed = embed_model.encode([question])
    results = vector_db.search(q_embed)

    context = ""
    for r in results:
        context += f"File: {r['file']}\n{r['code']}\n\n"

    prompt = f"Question: {question}\nCode:\n{context}"
    return ask_llm(prompt)

if __name__ == "__main__":
    # Test the system
    question = "How does the battery charging work?"
    print(f"\n[Query] {question}\n")
    response = query_system(question)
    print(f"[Response]\n{response}")
