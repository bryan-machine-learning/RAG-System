import os
from openai import OpenAI
from retrieve import retrieve

BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
API_KEY = os.getenv("LLM_API_KEY", "ollama")
MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = """You answer questions using ONLY the numbered context passages provided.
Rules:
- Cite the passages you used like [1] or [2][3] after each claim.
- If the passages do not contain the answer, reply exactly: "I don't know based on the provided documents."
- Do not use outside knowledge. Be concise."""

def build_context(chunks):
    return "\n\n".join(
        f"[{i}] ({c['source']} p.{c['page']})\n{c['text']}"
        for i, c in enumerate(chunks, start=1)
    )

def answer(question, k=5):
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

if __name__ == "__main__":
    questions = [
        "How do rerankers improve retrieval quality?",
        "What are common failure modes of RAG systems?",
        "What is the capital of France?",   # should say "I don't know"
    ]
    for q in questions:
        result = answer(q)
        print(f"\n=== {q}\n{result['answer']}\n")
        for i, s in enumerate(result["sources"], start=1):
            print(f"  [{i}] {s['source']} p.{s['page']}")