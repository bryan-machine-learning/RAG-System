import argparse
import json
from re import search
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from retrieve import retrieve, retrieve_reranked

KS = [1, 3, 5, 10]

def load(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(l) for l in lines if l.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="baseline")
    ap.add_argument("--rerank", action="store_true", help="use reranking")
    ap.add_argument("--generate", action="store_true", help="also test answers (slow)")
    args = ap.parse_args()

    qs = load(ROOT / "eval" / "questions.jsonl")
    answerable = [q for q in qs if q["answerable"]]
    unanswerable = [q for q in qs if not q["answerable"]]

    hits = {k: 0 for k in KS}
    rr_total = 0.0
    misses = []
    for q in answerable:
        gold = {(g["source"], g["page"]) for g in q["gold"]}
        search = retrieve_reranked if args.rerank else retrieve
        results = search(q["question"], k=max(KS))

        rank = next(
            (i for i, r in enumerate(results, 1) if (r["source"], r["page"]) in gold),
            None,
        )
        if rank:
            rr_total += 1 / rank
            for k in KS:
                if rank <= k:
                    hits[k] += 1
        else:
            misses.append(q["question"])

    n = len(answerable)
    summary = {f"recall@{k}": round(hits[k] / n, 3) for k in KS}
    summary["mrr"] = round(rr_total / n, 3)
    summary["n_answerable"] = n

    if args.generate:
        from generate import answer
        def refused(q):
            return "i don't know" in answer(q["question"])["answer"].lower()
        summary["wrong_refusal_rate"] = round(sum(refused(q) for q in answerable) / n, 3)
        summary["correct_refusal_rate"] = round(
            sum(refused(q) for q in unanswerable) / max(len(unanswerable), 1), 3
        )

    print(f"\n=== {args.name} ===")
    for k, v in summary.items():
        print(f"{k:22} {v}")
    print(f"\nRetrieval misses ({len(misses)}):")
    for m in misses:
        print(" -", m)

    out = ROOT / "eval" / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name}.json").write_text(
        json.dumps({"summary": summary, "misses": misses}, indent=2), encoding="utf-8"
    )

if __name__ == "__main__":
    main()