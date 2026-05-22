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


def test_retrieve_drug_id_matches_generic_when_drug_name_is_brand() -> None:
    """
    drug_name 이 brand 명("PAIN RELIEVER") 인 청크를 generic("acetaminophen") 으로 검색해도 매칭
    """
    index = FaissIndex(dim=4)
    chunks = [
        Chunk(
            text="warning text",
            section="warnings",
            drug_name="PAIN RELIEVER EXTRA STRENGTH",
            brand_name="PAIN RELIEVER EXTRA STRENGTH",
            generic_name="ACETAMINOPHEN",
        ),
    ]
    embeddings = [[1.0, 0.0, 0.0, 0.0]]
    index.add(chunks, embeddings)

    service = RetrievalService(
        embedder=_embedder_with([1.0, 0.0, 0.0, 0.0]),
        index=index,
    )

    results = service.retrieve("liver damage", top_k=5, drug_id="acetaminophen")

    assert len(results) == 1
    assert results[0][0].drug_name == "PAIN RELIEVER EXTRA STRENGTH"


def test_retrieve_drug_id_matches_brand_when_drug_name_is_generic() -> None:
    """
    drug_name 이 generic("ibuprofen") 인 청크를 brand("ADVIL") 로 검색해도 매칭
    """
    index = FaissIndex(dim=4)
    chunks = [
        Chunk(
            text="dosage text",
            section="dosage",
            drug_name="ibuprofen",
            brand_name="ADVIL",
            generic_name="ibuprofen",
        ),
    ]
    embeddings = [[1.0, 0.0, 0.0, 0.0]]
    index.add(chunks, embeddings)

    service = RetrievalService(
        embedder=_embedder_with([1.0, 0.0, 0.0, 0.0]),
        index=index,
    )

    results = service.retrieve("kidney", top_k=5, drug_id="advil")

    assert len(results) == 1


def test_retrieve_drug_id_filter_excludes_unrelated_chunks() -> None:
    """
    drug_id 가 어느 필드와도 매칭 안 되면 결과 비어있어야 함
    """
    index = FaissIndex(dim=4)
    chunks = [
        Chunk(
            text="text",
            section="warnings",
            drug_name="ASPIRIN",
            brand_name="ASPIRIN",
            generic_name="aspirin",
        ),
    ]
    embeddings = [[1.0, 0.0, 0.0, 0.0]]
    index.add(chunks, embeddings)

    service = RetrievalService(
        embedder=_embedder_with([1.0, 0.0, 0.0, 0.0]),
        index=index,
    )

    results = service.retrieve("query", top_k=5, drug_id="ibuprofen")

    assert results == []


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


def test_retrieve_drug_id_finds_low_similarity_chunk_outside_oversample_window() -> None:
    """
    drug_id 대상 청크가 유사도 낮아 oversample 창 밖이어도 필터로 반환되어야 함
    (post-filter 가 좁은 후보 창에 걸려 빈 결과가 되던 문제 회귀 방지)
    """
    index = FaissIndex(dim=4)
    # 쿼리([1,0,0,0])에 매우 가까운 ASPIRIN 청크 6개 + 유사도 0 인 TYLENOL 청크 1개
    chunks = [Chunk(text=f"aspirin {i}", section="warnings", drug_name="ASPIRIN") for i in range(6)]
    chunks.append(Chunk(text="tylenol", section="indications", drug_name="TYLENOL"))
    embeddings = [[1.0, 0.0, 0.0, 0.0]] * 6 + [[0.0, 0.0, 0.0, 1.0]]
    index.add(chunks, embeddings)

    service = RetrievalService(
        embedder=_embedder_with([1.0, 0.0, 0.0, 0.0]),
        index=index,
    )

    # top_k=1 → 기존 oversample(5) 창에는 ASPIRIN 만 들어와 TYLENOL 이 누락되던 케이스
    results = service.retrieve("pain", top_k=1, drug_id="tylenol")

    assert len(results) == 1
    assert results[0][0].drug_name == "TYLENOL"


def test_retrieve_korean_drug_id_matches_english_generic_via_alias() -> None:
    """
    한국어 성분/브랜드 drug_id 가 영어 generic 청크에 alias 로 매칭 (Phase 2)
    """
    index = FaissIndex(dim=4)
    chunks = [
        Chunk(
            text="weight management",
            section="indications",
            drug_name="WEGOVY",
            brand_name="WEGOVY",
            generic_name="SEMAGLUTIDE",
        ),
    ]
    index.add(chunks, [[1.0, 0.0, 0.0, 0.0]])

    service = RetrievalService(
        embedder=_embedder_with([1.0, 0.0, 0.0, 0.0]),
        index=index,
    )

    # 한국어 브랜드(위고비) / 성분(세마글루티드) → semaglutide 로 확장돼 매칭
    assert len(service.retrieve("부작용", top_k=5, drug_id="위고비")) == 1
    assert len(service.retrieve("부작용", top_k=5, drug_id="세마글루티드")) == 1
    # 매핑 없는 한국어는 여전히 매칭 안 됨
    assert service.retrieve("부작용", top_k=5, drug_id="존재하지않는약") == []
