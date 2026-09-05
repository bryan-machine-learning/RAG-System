from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
col = chromadb.PersistentClient(path="chroma_db").get_collection("docs")

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

def retrieve(question, k=5):
    emb = model.encode([QUERY_PREFIX + question], normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=emb, n_results=k)
    return [
        {"text": d, "source": m["source"], "page": m["page"], "distance": dist}
        for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]

if __name__ == "__main__":
    questions = [
        "What is retrieval-augmented generation?",
        "How do rerankers improve retrieval quality?",
        "What are common failure modes of RAG systems?",
    ]
    for q in questions:
        print(f"\n=== {q}")
        for r in retrieve(q, k=3):
            print(f"{r['source']} p.{r['page']} ({r['distance']:.3f})\n{r['text'][:200]}\n")