"""검색 평가 지표 (순수 함수)

- Recall@K: 정답 청크가 top-K 안에 있는지 (0 또는 1)
- MRR: 첫 정답의 1/순위 (없으면 0)
- 매칭은 drug_name 부분 일치 (대소문자 무시) + 선택적 section 부분 일치
"""

from __future__ import annotations

from mediforme_chatbot_rag.ingestion.chunker import Chunk


def is_match(
    chunk: Chunk,
    *,
    expected_drug_names: list[str],
    expected_sections: list[str] | None = None,
) -> bool:
    """
    청크가 기대 정답에 부합하는지 판정

    - drug_name 은 expected_drug_names 중 하나라도 부분 일치(case-insensitive)
    - expected_sections 가 주어지면 section 도 같은 방식으로 부분 일치 필요
    """
    name_lower = chunk.drug_name.lower()
    name_match = any(exp.lower() in name_lower for exp in expected_drug_names)
    if not name_match:
        return False
    if expected_sections:
        section_lower = chunk.section.lower()
        if not any(exp.lower() in section_lower for exp in expected_sections):
            return False
    return True


def recall_at_k(
    ranked_chunks: list[Chunk],
    *,
    expected_drug_names: list[str],
    expected_sections: list[str] | None = None,
    k: int,
) -> float:
    """
    top-K 안에 정답이 1개 이상 있으면 1.0, 아니면 0.0
    """
    for chunk in ranked_chunks[:k]:
        if is_match(
            chunk,
            expected_drug_names=expected_drug_names,
            expected_sections=expected_sections,
        ):
            return 1.0
    return 0.0


def reciprocal_rank(
    ranked_chunks: list[Chunk],
    *,
    expected_drug_names: list[str],
    expected_sections: list[str] | None = None,
) -> float:
    """
    첫 정답의 1/순위. 정답이 없으면 0.0
    """
    for idx, chunk in enumerate(ranked_chunks, start=1):
        if is_match(
            chunk,
            expected_drug_names=expected_drug_names,
            expected_sections=expected_sections,
        ):
            return 1.0 / idx
    return 0.0
