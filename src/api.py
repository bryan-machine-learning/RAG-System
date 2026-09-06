from fastapi import FastAPI
from pydantic import BaseModel

from generate import answer

app = FastAPI(title="RAG over arXiv papers")


class Query(BaseModel):
    question: str
    k: int = 5


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(q: Query):
    result = answer(q.question, k=q.k, rerank=True, hybrid=True)
    return {
        "answer": result["answer"],
        "sources": [{"source": s["source"], "page": s["page"]} for s in result["sources"]],
    }