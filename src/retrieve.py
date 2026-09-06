from sentence_transformers import SentenceTransformer
import chromadb
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
import re

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

_bm25 = None
_chunks = None

def _tok(text):
    return re.findall(r"[a-z0-9]+", text.lower())

def _load_bm25():
    global _bm25, _chunks
    if _bm25 is None:
        data = col.get(include=["documents", "metadatas"])
        _chunks = [
            {"text": d, "source": m["source"], "page": m["page"]}
            for d, m in zip(data["documents"], data["metadatas"])
        ]
        _bm25 = BM25Okapi([_tok(c["text"]) for c in _chunks])
    return _bm25, _chunks

def retrieve_hybrid(question, k=5, pool=50, rrf_k=60):
    bm25, chunks = _load_bm25()
    dense = retrieve(question, k=pool)
    scores = bm25.get_scores(_tok(question))
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:pool]

    fused = {}
    def key(c):
        return (c["source"], c["page"], c["text"][:80])
    for rank, c in enumerate(dense, 1):
        fused.setdefault(key(c), [c, 0.0])[1] += 1 / (rrf_k + rank)
    for rank, i in enumerate(top_idx, 1):
        c = chunks[i]
        fused.setdefault(key(c), [c, 0.0])[1] += 1 / (rrf_k + rank)
    ranked = sorted(fused.values(), key=lambda x: x[1], reverse=True)
    return [dict(c) for c, _ in ranked[:k]]


_reranker = None

def retrieve_reranked(question, k=5, candidates=20, hybrid=False):
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("BAAI/bge-reranker-base")
    cands = (retrieve_hybrid if hybrid else retrieve)(question, k=candidates)
    scores = _reranker.predict([(question, c["text"]) for c in cands])
    ranked = sorted(zip(cands, scores), key=lambda x: x[1], reverse=True)
    out = []
    for c, s in ranked[:k]:
        c = dict(c)
        c["score"] = float(s)
        out.append(c)
    return out