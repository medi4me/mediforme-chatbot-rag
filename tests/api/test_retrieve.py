"""/retrieve 엔드포인트 테스트

- TestClient + dependency_overrides 로 RetrievalService 모킹
- HTTP 응답 형식·검증·필터 전달 확인
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mediforme_chatbot_rag.api.retrieve import get_retrieval_service
from mediforme_chatbot_rag.ingestion.chunker import Chunk
from mediforme_chatbot_rag.main import app


class _FakeService:
    """
    호출 인자를 기록하고 미리 준비한 결과를 돌려주는 가짜 RetrievalService
    """

    def __init__(self, results: list[tuple[Chunk, float]]) -> None:
        self.results = results
        self.last_call: dict[str, Any] = {}

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        drug_id: str | None = None,
        category: str | None = None,
    ) -> list[tuple[Chunk, float]]:
        self.last_call = {
            "query": query,
            "top_k": top_k,
            "drug_id": drug_id,
            "category": category,
        }
        return self.results


@pytest.fixture
def fake_service() -> _FakeService:
    chunk = Chunk(
        text="Do not use if you have severe hepatic impairment.",
        section="contraindications",
        drug_name="TYLENOL",
    )
    return _FakeService(results=[(chunk, 0.92)])


@pytest.fixture
def client(fake_service: _FakeService) -> Iterator[TestClient]:
    app.dependency_overrides[get_retrieval_service] = lambda: fake_service
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def test_endpoint_returns_chunks(client: TestClient, fake_service: _FakeService) -> None:
    resp = client.post(
        "/retrieve",
        json={"query": "타이레놀 먹으면 안 되는 사람", "top_k": 5},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["chunks"]) == 1
    chunk = body["chunks"][0]
    assert chunk["text"] == fake_service.results[0][0].text
    assert chunk["drug_name"] == "TYLENOL"
    assert chunk["section"] == "contraindications"
    assert chunk["source"] == "fda_label"
    assert chunk["similarity"] == pytest.approx(0.92)


def test_endpoint_passes_filters_to_service(client: TestClient, fake_service: _FakeService) -> None:
    client.post(
        "/retrieve",
        json={
            "query": "warnings",
            "top_k": 3,
            "drug_id": "tylenol",
            "category": "warnings",
        },
    )

    assert fake_service.last_call == {
        "query": "warnings",
        "top_k": 3,
        "drug_id": "tylenol",
        "category": "warnings",
    }


def test_endpoint_validates_top_k_lower_bound(client: TestClient) -> None:
    resp = client.post("/retrieve", json={"query": "test", "top_k": 0})
    assert resp.status_code == 422


def test_endpoint_validates_top_k_upper_bound(client: TestClient) -> None:
    resp = client.post("/retrieve", json={"query": "test", "top_k": 21})
    assert resp.status_code == 422


def test_endpoint_returns_empty_when_service_returns_nothing() -> None:
    empty_service = _FakeService(results=[])
    app.dependency_overrides[get_retrieval_service] = lambda: empty_service
    try:
        client = TestClient(app)
        resp = client.post("/retrieve", json={"query": "no match", "top_k": 5})
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    assert resp.json() == {"chunks": []}
