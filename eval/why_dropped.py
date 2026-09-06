import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from ingest import RAW_DIR, load_pdf, chunk_text, is_junk, looks_like_template

TARGETS = [("2406.03790v2.pdf", 12), ("2509.21336v1.pdf", 9)]

for src, page in TARGETS:
    for page_num, text in load_pdf(RAW_DIR / src):
        if page_num != page:
            continue
        print(f"\n##### {src} p.{page}")
        for i, chunk in enumerate(chunk_text(text)):
            low = chunk.lower()
            reasons = []
            if len(chunk.strip()) < 200:
                reasons.append(f"short({len(chunk.strip())})")
            if low.count("et al") >= 3:
                reasons.append(f"et_al({low.count('et al')})")
            if is_junk(chunk) and not reasons:
                reasons.append("citation-year rule")
            if looks_like_template(chunk):
                reasons.append("template")
            status = "DROPPED " + ",".join(reasons) if reasons else "kept"
            print(f"\n[{i}] {status}")
            print(chunk[:250].replace("\n", " "))