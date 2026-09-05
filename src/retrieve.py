from sentence_transformers import SentenceTransformer
import chromadb
from sentence_transformers import CrossEncoder

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


_reranker = None

def retrieve_reranked(question, k=5, candidates=20):
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("BAAI/bge-reranker-base")
    cands = retrieve(question, k=candidates)
    scores = _reranker.predict([(question, c["text"]) for c in cands])
    ranked = sorted(zip(cands, scores), key=lambda x: x[1], reverse=True)
    out = []
    for c, s in ranked[:k]:
        c = dict(c)
        c["score"] = float(s)
        out.append(c)
    return out