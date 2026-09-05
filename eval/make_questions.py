import json
import os
import random
from collections import Counter
from pathlib import Path

import chromadb
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
N_QUESTIONS = 50
MAX_PER_PAPER = 3

client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
    api_key=os.getenv("LLM_API_KEY", "ollama"),
)
MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")

PROMPT = """Below is a passage from a research paper. Write ONE specific question that can be answered using ONLY this passage.
Rules:
- Paraphrase. Do not reuse long phrases from the passage.
- The question must make sense on its own (never say "in this passage" or "the paper").
- If the passage is a reference list, table of numbers, or has no real content, reply exactly: SKIP
Reply with only the question.

Passage:
{text}"""

def is_junk(text, page):
    low = text.lower()
    return (
        len(text) < 500
        or page == 1
        or low.count("et al") > 3
        or "doi" in low
        or "arxiv:" in low
    )

def main():
    col = chromadb.PersistentClient(path=str(ROOT / "chroma_db")).get_collection("docs")
    data = col.get(include=["documents", "metadatas"])
    items = list(zip(data["documents"], data["metadatas"]))
    random.seed(42)
    random.shuffle(items)

    per_paper = Counter()
    out_path = ROOT / "eval" / "questions_draft.jsonl"
    kept = 0
    with out_path.open("w", encoding="utf-8") as f:
        for text, meta in items:
            if kept >= N_QUESTIONS:
                break
            src, page = meta["source"], meta["page"]
            if is_junk(text, page) or per_paper[src] >= MAX_PER_PAPER:
                continue
            resp = client.chat.completions.create(
                model=MODEL,
                temperature=0.3,
                messages=[{"role": "user", "content": PROMPT.format(text=text)}],
            )
            q = resp.choices[0].message.content.strip().strip('"')
            if "SKIP" in q or not q.endswith("?"):
                continue
            per_paper[src] += 1
            kept += 1
            row = {
                "question": q,
                "gold": [{"source": src, "page": page}],
                "answerable": True,
                "passage": text[:300].replace("\n", " "),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{kept}. {q}")

if __name__ == "__main__":
    main()