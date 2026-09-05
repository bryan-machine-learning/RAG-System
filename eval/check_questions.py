import json
from pathlib import Path

path = Path(__file__).resolve().parent / "questions.jsonl"
for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
    if not line.strip():
        continue
    try:
        json.loads(line)
    except json.JSONDecodeError as e:
        print(f"line {i}: {e}")
        print("   ", line[:160])