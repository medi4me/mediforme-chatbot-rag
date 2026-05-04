"""RetrievalService 단위 테스트

- 실제 FaissIndex + 가짜 임베더로 검색·필터 동작 검증
- 인덱스가 비어 있을 때, drug_id / category 로 좁힐 때 모두 커버
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mediforme_chatbot_rag.ingestion.chunker import Chunk
from mediforme_chatbot_rag.ingestion.embedder import Embedder
from mediforme_chatbot_rag.ingestion.index import FaissIndex
from mediforme_chatbot_rag.services.retrieval import RetrievalService


class _FixedSentencesModel:
    """
    주어진 벡터를 텍스트 수만큼 복제해 돌려주는 가짜 SentenceTransformer
    """

    def __init__(self, vector: list[float]) -> None:
        self._vector = np.array(vector, dtype=np.float32)

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> Any:
        return np.tile(self._vector, (len(sentences), 1))


def _embedder_with(query_vector: list[float]) -> Embedder:
    return Embedder(model=_FixedSentencesModel(query_vector))


def _build_index() -> FaissIndex:
    index = FaissIndex(dim=4)
    chunks = [
        Chunk(text="Tylenol pain relief", section="indications", drug_name="TYLENOL"),
        Chunk(text="Tylenol contraindications", section="contraindications", drug_name="TYLENOL"),
        Chunk(text="Aspirin pain relief", section="indications", drug_name="ASPIRIN"),
        Chunk(text="Aspirin warnings", section="warnings", drug_name="ASPIRIN"),
    ]
    embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.9, 0.1, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.9, 0.1, 0.0],
    ]
    index.add(chunks, embeddings)
    return index


def test_retrieve_returns_empty_when_index_is_none() -> None:
    service = RetrievalService(embedder=_embedder_with([1.0, 0.0, 0.0, 0.0]), index=None)

    assert service.retrieve("query", top_k=5) == []


def test_retrieve_returns_empty_when_index_is_empty() -> None:
    empty_index = FaissIndex(dim=4)
    service = RetrievalService(
        embedder=_embedder_with([1.0, 0.0, 0.0, 0.0]),
        index=empty_index,
    )

    assert service.retrieve("query", top_k=5) == []


def test_retrieve_returns_top_k_closest_chunks() -> None:
    service = RetrievalService(
        embedder=_embedder_with([1.0, 0.0, 0.0, 0.0]),
        index=_build_index(),
    )

    results = service.retrieve("tylenol pain", top_k=2)

    assert len(results) == 2
    assert results[0][0].drug_name == "TYLENOL"
    assert results[0][0].section == "indications"
    assert results[0][1] >= results[1][1]


def test_retrieve_drug_id_filter_is_case_insensitive() -> None:
    service = RetrievalService(
        embedder=_embedder_with([0.0, 1.0, 0.0, 0.0]),
        index=_build_index(),
    )

    results = service.retrieve("aspirin", top_k=5, drug_id="aspirin")

    assert len(results) == 2
    assert all(chunk.drug_name == "ASPIRIN" for chunk, _ in results)


def test_retrieve_category_filter_matches_section() -> None:
    service = RetrievalService(
        embedder=_embedder_with([1.0, 0.0, 0.0, 0.0]),
        index=_build_index(),
    )

    results = service.retrieve("query", top_k=5, category="indications")

    sections = [chunk.section for chunk, _ in results]
    assert sections
    assert all(s == "indications" for s in sections)


def test_retrieve_combines_drug_id_and_category() -> None:
    service = RetrievalService(
        embedder=_embedder_with([1.0, 0.0, 0.0, 0.0]),
        index=_build_index(),
    )

    results = service.retrieve("query", top_k=5, drug_id="TYLENOL", category="contraindications")

    assert len(results) == 1
    assert results[0][0].drug_name == "TYLENOL"
    assert results[0][0].section == "contraindications"


def test_retrieve_filter_with_no_match_returns_empty() -> None:
    service = RetrievalService(
        embedder=_embedder_with([1.0, 0.0, 0.0, 0.0]),
        index=_build_index(),
    )

    assert service.retrieve("query", top_k=5, drug_id="NOT_EXIST") == []
