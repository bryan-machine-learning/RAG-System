import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import chromadb
from retrieve import retrieve

col = chromadb.PersistentClient(path=str(ROOT / "chroma_db")).get_collection("docs")
lines = (ROOT / "eval" / "questions.jsonl").read_text(encoding="utf-8").splitlines()
for line in lines:
    if not line.strip():
        continue
    q = json.loads(line)
    if not q["answerable"]:
        continue
    gold = {(g["source"], g["page"]) for g in q["gold"]}
    res = retrieve(q["question"], k=10)
    if any((r["source"], r["page"]) in gold for r in res):
        continue
    for s, p in gold:
        n = len(col.get(where={"$and": [{"source": s}, {"page": p}]})["ids"])
        print(f"MISS | {s} p.{p} has {n} chunks | {q['question'][:90]}")