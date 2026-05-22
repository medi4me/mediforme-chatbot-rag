"""RAG 검색 서비스

- 임베더 + FAISS 인덱스를 묶어 /retrieve 핸들러의 단일 진입점 제공
- post-filter 로 drug_id / category 적용 후 top_k 결과 반환
"""

from __future__ import annotations

from pathlib import Path

from mediforme_chatbot_rag.core.config import get_settings
from mediforme_chatbot_rag.core.drug_aliases import expand_drug_id
from mediforme_chatbot_rag.ingestion.chunker import Chunk
from mediforme_chatbot_rag.ingestion.embedder import Embedder
from mediforme_chatbot_rag.ingestion.index import (
    CHUNKS_FILENAME,
    INDEX_FILENAME,
    FaissIndex,
)

# 필터 적용 후 결과가 부족하지 않도록 oversample 후 top_k 자르기
_OVERSAMPLE_FACTOR = 5


class RetrievalService:
    """
    임베더 + FAISS 인덱스 묶음 검색 서비스
    """

    def __init__(self, embedder: Embedder, index: FaissIndex | None) -> None:
        self._embedder = embedder
        self._index = index

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        drug_id: str | None = None,
        category: str | None = None,
    ) -> list[tuple[Chunk, float]]:
        if self._index is None or len(self._index) == 0:
            return []

        query_embedding = self._embedder.embed([query])[0]
        # 필터가 있으면 후보를 인덱스 전체로 확장해 해당 약·섹션 청크가 좁은 oversample
        # 창 밖으로 밀려 빈 결과가 되는 문제를 피함 (filter-then-rank). IndexFlatIP 는
        # 어차피 전수 계산이라 후보 수를 키워도 비용 차이는 미미하다
        if drug_id is not None or category is not None:
            search_k = len(self._index)
        else:
            search_k = top_k * _OVERSAMPLE_FACTOR
        candidates = self._index.search(query_embedding, search_k)

        filtered = [
            (chunk, score)
            for chunk, score in candidates
            if _matches_filters(chunk, drug_id=drug_id, category=category)
        ]
        return filtered[:top_k]


def load_retrieval_service() -> RetrievalService:
    """
    설정의 INDEX_DIR 에서 인덱스를 로드해 RetrievalService 생성

    - 인덱스 파일이 아직 없으면 index=None 으로 만들어 빈 응답을 돌리도록 함
    """
    settings = get_settings()
    embedder = Embedder()
    index_path = Path(settings.index_dir)
    if (index_path / INDEX_FILENAME).exists() and (index_path / CHUNKS_FILENAME).exists():
        index: FaissIndex | None = FaissIndex.load(index_path)
    else:
        index = None
    return RetrievalService(embedder=embedder, index=index)


def _matches_filters(
    chunk: Chunk,
    *,
    drug_id: str | None,
    category: str | None,
) -> bool:
    if drug_id is not None:
        # 한국어 성분/브랜드 drug_id 는 영어 generic 으로도 확장해 매칭 (cross-lingual)
        needles = expand_drug_id(drug_id)
        candidates = [
            c.lower() for c in (chunk.drug_name, chunk.brand_name, chunk.generic_name) if c
        ]
        if not any(needle in cand for needle in needles for cand in candidates):
            return False
    return not (category is not None and chunk.section != category)
