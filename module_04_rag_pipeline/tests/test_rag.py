from fastapi.testclient import TestClient

from src.chunking import fixed_chunks, sentence_chunks
from src.main import app


def test_fixed_chunks_overlap() -> None:
    chunks = fixed_chunks(" ".join(str(index) for index in range(20)), "test", 10, 2)
    assert len(chunks) == 3
    assert chunks[0].text.split()[-2:] == chunks[1].text.split()[:2]


def test_sentence_chunks_preserve_source() -> None:
    chunks = sentence_chunks("One sentence. Another sentence.", "filing.txt", 2)
    assert chunks
    assert all(chunk.source == "filing.txt" for chunk in chunks)


def test_health_endpoint() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_query_validation() -> None:
    response = TestClient(app).post("/query", json={"question": "", "session_id": "test"})
    assert response.status_code == 422
