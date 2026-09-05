from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

CHUNK_SIZE = 800   # characters
OVERLAP = 150

def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks

def load_pdf(path):
    reader = PdfReader(path)
    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            yield page_num, text

def main():
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    client = chromadb.PersistentClient(path="chroma_db")
    col = client.get_or_create_collection("docs", metadata={"hnsw:space": "cosine"})

    for pdf in Path("data/raw").glob("*.pdf"):
        ids, docs, metas = [], [], []
        for page_num, text in load_pdf(pdf):
            for i, chunk in enumerate(chunk_text(text)):
                ids.append(f"{pdf.stem}-p{page_num}-c{i}")
                docs.append(chunk)
                metas.append({"source": pdf.name, "page": page_num})
        if docs:
            embeddings = model.encode(docs, normalize_embeddings=True).tolist()
            col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
        print(f"Ingested {pdf.name}: {len(docs)} chunks")

if __name__ == "__main__":
    main()