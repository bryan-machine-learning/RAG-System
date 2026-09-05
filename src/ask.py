from generate import answer

print("Ask a question about your papers (type 'quit' to exit).")
while True:
    q = input("\n> ").strip()
    if q.lower() in ("quit", "exit", ""):
        break
    result = answer(q, rerank=True)
    print("\n" + result["answer"] + "\n")
    for i, s in enumerate(result["sources"], 1):
        print(f"  [{i}] {s['source']} p.{s['page']}")