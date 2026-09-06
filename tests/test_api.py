from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(
    not (ROOT / "chroma_db").exists(), reason="vector index not built (run src/ingest.py)"
)


def test_health():
    from fastapi.testclient import TestClient
    from api import app

    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ask_rejects_missing_question():
    from fastapi.testclient import TestClient
    from api import app

    client = TestClient(app)
    r = client.post("/ask", json={})
    assert r.status_code == 422