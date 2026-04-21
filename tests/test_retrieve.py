from fastapi.testclient import TestClient

from mediforme_chatbot_rag.main import app


def test_retrieve_returns_empty_stub() -> None:
    client = TestClient(app)
    resp = client.post(
        "/retrieve",
        json={"query": "타이레놀 먹으면 안 되는 사람", "top_k": 5},
    )
    assert resp.status_code == 200
    assert resp.json() == {"chunks": []}


def test_retrieve_validates_top_k_bounds() -> None:
    client = TestClient(app)
    resp = client.post("/retrieve", json={"query": "test", "top_k": 0})
    assert resp.status_code == 422
