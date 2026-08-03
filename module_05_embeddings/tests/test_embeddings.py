from fastapi.testclient import TestClient

from src.main import app
from src.vector_store.vector_db import DocumentIndex


def test_index_returns_ranked_documents() -> None:
    index = DocumentIndex([("a", "profit increased"), ("b", "interest rates fell")])
    results = index.search("higher profit", 2)
    assert results[0].document_id == "a"
    assert results[0].score >= results[1].score


def test_health_endpoint() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["documents"] > 0


def test_similar_docs_validation() -> None:
    response = TestClient(app).post(
        "/similar-docs", json={"document_text": "earnings improved", "top_k": 0}
    )
    assert response.status_code == 422
