import os
from openai import OpenAI
from retrieve import retrieve, retrieve_hybrid, retrieve_reranked, retrieve_reranked

BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
API_KEY = os.getenv("LLM_API_KEY", "ollama")
MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = """You answer questions using ONLY the numbered context passages provided.
Rules:
- Cite the passages you used like [1] or [2][3] after each claim.
- If the passages do not contain the answer, reply exactly: "I don't know based on the provided documents."
- Do not use outside knowledge. Be concise.
Only answer the exact question asked. If the passages merely contain similar-looking examples and don't address that question, refuse."""

def build_context(chunks):
    return "\n\n".join(
        f"[{i}] ({c['source']} p.{c['page']})\n{c['text']}"
        for i, c in enumerate(chunks, start=1)
    )

def answer(question, k=5, rerank=False, hybrid=False):
    if rerank:
        chunks = retrieve_reranked(question, k=k, hybrid=hybrid)
    elif hybrid:
        chunks = retrieve_hybrid(question, k=k)
    else:
        chunks = retrieve(question, k=k)

    user_msg = f"Context:\n{build_context(chunks)}\n\nQuestion: {question}"
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    return {"answer": resp.choices[0].message.content, "sources": chunks}