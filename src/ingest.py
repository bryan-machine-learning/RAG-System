import os
import re
from pathlib import Path

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = str(ROOT / "chroma_db")
RAW_DIR = ROOT / "data" / "raw"

CHUNK_SIZE = 800   # characters
OVERLAP = 150

# Filters are optional and OFF by default (they showed no clear gain on my eval).
# Turn on with:  set USE_FILTERS=1
USE_FILTERS = os.getenv("USE_FILTERS", "0") == "1"

REF_MARKERS = ("proceedings", "conference on", "journal of", "arxiv preprint",
               "[online]", "available:", "pp.", "association for computational linguistics")
TEMPLATE_MARKERS = ("example:", "output:", '"reasoning"', "question:", "answer:", "content:")


def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def is_junk(chunk):
    """Bibliography entries and trivially short fragments."""
    low = chunk.lower()
    markers = sum(low.count(m) for m in REF_MARKERS)
    years = len(re.findall(r"\b(?:19|20)\d{2}\b", chunk))
    bracket_nums = len(re.findall(r"\[\d{1,3}\]", chunk))
    return (
        len(chunk.strip()) < 80
        or markers >= 2
        or (bracket_nums >= 3 and years >= 2)
    )


def looks_like_template(chunk):
    """Prompt templates and few-shot examples that look like facts but aren't."""
    low = chunk.lower()
    return sum(m in low for m in TEMPLATE_MARKERS) >= 3


def load_pdf(path):
    reader = PdfReader(path)
    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            yield page_num, text


def main():
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    client = chromadb.PersistentClient(path=DB_PATH)
    col = client.get_or_create_collection("docs", metadata={"hnsw:space": "cosine"})

    total_kept = total_junk = total_template = 0
    for pdf in sorted(RAW_DIR.glob("*.pdf")):
        ids, docs, metas = [], [], []
        for page_num, text in load_pdf(pdf):
            for i, chunk in enumerate(chunk_text(text)):
                if USE_FILTERS and is_junk(chunk):
                    total_junk += 1
                    continue
                if USE_FILTERS and looks_like_template(chunk):
                    total_template += 1
                    continue
                ids.append(f"{pdf.stem}-p{page_num}-c{i}")
                docs.append(chunk)
                metas.append({"source": pdf.name, "page": page_num})
        if docs:
            embeddings = model.encode(docs, normalize_embeddings=True).tolist()
            col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
        total_kept += len(docs)
        print(f"Ingested {pdf.name}: {len(docs)} chunks")

    print(f"\nFilters on: {USE_FILTERS} | kept {total_kept} chunks | "
          f"dropped as junk: {total_junk} | dropped as template: {total_template}")


if __name__ == "__main__":
    main()