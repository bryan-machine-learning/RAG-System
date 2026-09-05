import time
from pathlib import Path

import arxiv
import requests

QUERY = 'abs:"retrieval-augmented generation"'
MAX_PAPERS = 40
OUT_DIR = Path("data/raw")
HEADERS = {"User-Agent": "rag-portfolio-project (personal learning project)"}

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = arxiv.Client(page_size=20, delay_seconds=3, num_retries=3)
    search = arxiv.Search(
        query=QUERY,
        max_results=MAX_PAPERS,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    for paper in client.results(search):
        path = OUT_DIR / f"{paper.get_short_id()}.pdf"
        if path.exists() and path.stat().st_size > 10_000:
            continue
        url = paper.pdf_url.replace("http://", "https://")
        try:
            resp = requests.get(url, headers=HEADERS, timeout=60)
            resp.raise_for_status()
            if not resp.content.startswith(b"%PDF"):
                raise ValueError("response was not a PDF")
            path.write_bytes(resp.content)
            print(f"Downloaded: {paper.title[:70]}")
        except Exception as e:
            print(f"Failed {path.name}: {e}")
        time.sleep(3)

if __name__ == "__main__":
    main()