"""평가 지표 단위 테스트

- is_match: drug_name / section substring 매칭 동작 검증
- recall_at_k / reciprocal_rank: 가짜 ranked 리스트로 계산이 맞는지 검증
"""

from __future__ import annotations

import pytest

from mediforme_chatbot_rag.eval.metrics import is_match, recall_at_k, reciprocal_rank
from mediforme_chatbot_rag.ingestion.chunker import Chunk


def _chunk(drug_name: str, section: str = "warnings") -> Chunk:
    return Chunk(text="t", section=section, drug_name=drug_name)


def test_is_match_drug_name_substring_case_insensitive() -> None:
    chunk = _chunk(drug_name="어린이타이레놀산160밀리그램")

    assert is_match(chunk, expected_drug_names=["타이레놀"]) is True
    assert is_match(chunk, expected_drug_names=["TYLENOL"]) is False


def test_is_match_brand_substring_works_for_english() -> None:
    chunk = _chunk(drug_name="WEGOVY")

    assert is_match(chunk, expected_drug_names=["wegovy"]) is True
    assert is_match(chunk, expected_drug_names=["semaglutide"]) is False


def test_is_match_returns_false_when_no_drug_name_overlap() -> None:
    chunk = _chunk(drug_name="ASPIRIN")

    assert is_match(chunk, expected_drug_names=["tylenol", "ibuprofen"]) is False


def test_is_match_section_filter_required_when_provided() -> None:
    chunk = _chunk(drug_name="WEGOVY", section="warnings")

    assert (
        is_match(
            chunk,
            expected_drug_names=["wegovy"],
            expected_sections=["warnings"],
        )
        is True
    )
    assert (
        is_match(
            chunk,
            expected_drug_names=["wegovy"],
            expected_sections=["contraindications"],
        )
        is False
    )


def test_recall_at_k_returns_one_when_match_inside_top_k() -> None:
    ranked = [_chunk("ASPIRIN"), _chunk("WEGOVY"), _chunk("TYLENOL")]

    assert recall_at_k(ranked, expected_drug_names=["wegovy"], k=2) == 1.0
    assert recall_at_k(ranked, expected_drug_names=["tylenol"], k=2) == 0.0
    assert recall_at_k(ranked, expected_drug_names=["tylenol"], k=3) == 1.0


def test_recall_at_k_returns_zero_when_no_match_in_index() -> None:
    ranked = [_chunk("A"), _chunk("B"), _chunk("C")]

    assert recall_at_k(ranked, expected_drug_names=["wegovy"], k=10) == 0.0


def test_recall_at_k_with_empty_ranked_returns_zero() -> None:
    assert recall_at_k([], expected_drug_names=["wegovy"], k=5) == 0.0


def test_recall_at_k_respects_section_filter() -> None:
    ranked = [
        _chunk("WEGOVY", section="warnings"),
        _chunk("WEGOVY", section="contraindications"),
    ]

    assert (
        recall_at_k(
            ranked,
            expected_drug_names=["wegovy"],
            expected_sections=["contraindications"],
            k=1,
        )
        == 0.0
    )
    assert (
        recall_at_k(
            ranked,
            expected_drug_names=["wegovy"],
            expected_sections=["contraindications"],
            k=2,
        )
        == 1.0
    )


def test_reciprocal_rank_returns_one_for_first_position() -> None:
    ranked = [_chunk("WEGOVY"), _chunk("ASPIRIN")]

    assert reciprocal_rank(ranked, expected_drug_names=["wegovy"]) == 1.0


def test_reciprocal_rank_returns_one_third_for_third_position() -> None:
    ranked = [_chunk("A"), _chunk("B"), _chunk("WEGOVY"), _chunk("D")]

    assert reciprocal_rank(ranked, expected_drug_names=["wegovy"]) == pytest.approx(1 / 3)


def test_reciprocal_rank_returns_zero_when_no_match() -> None:
    ranked = [_chunk("A"), _chunk("B")]

    assert reciprocal_rank(ranked, expected_drug_names=["wegovy"]) == 0.0
