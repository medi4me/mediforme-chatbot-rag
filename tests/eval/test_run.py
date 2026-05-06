"""평가 러너 보조 함수 단위 테스트

- 실패 케이스 추출, 분할 표 포맷, 실패 섹션 마크다운 동작 검증
"""

from __future__ import annotations

from typing import Any

from mediforme_chatbot_rag.eval.run import (
    _build_failure_section,
    _build_split_table,
    _failure_entries,
    _format_split_row,
    _percentile,
)
from mediforme_chatbot_rag.ingestion.chunker import Chunk


def _result(
    *,
    qid: str,
    r5: float,
    r10: float,
    rr: float,
    drug_id: str | None = None,
    ranked: list[Chunk] | None = None,
    expected_drug_names: list[str] | None = None,
    query_text: str = "test",
) -> dict[str, Any]:
    return {
        "query": {
            "id": qid,
            "query": query_text,
            "expected_drug_names": expected_drug_names or ["test"],
        },
        "ranked": ranked or [],
        "r5": r5,
        "r10": r10,
        "rr": rr,
        "latency_ms": 12.0,
        "drug_id_used": drug_id,
    }


def test_failure_entries_filters_by_r10_zero() -> None:
    results = [
        _result(qid="a", r5=0, r10=1, rr=0.5),
        _result(qid="b", r5=0, r10=0, rr=0),
        _result(qid="c", r5=1, r10=1, rr=1),
        _result(qid="d", r5=0, r10=0, rr=0),
    ]

    failures = _failure_entries(results)

    assert [f["query"]["id"] for f in failures] == ["b", "d"]


def test_failure_entries_returns_empty_when_no_failures() -> None:
    results = [_result(qid="a", r5=1, r10=1, rr=1)]

    assert _failure_entries(results) == []


def test_format_split_row_handles_empty_group() -> None:
    row = _format_split_row("ko", [])

    assert "0" in row
    assert "-" in row


def test_format_split_row_averages_metrics() -> None:
    items = [
        _result(qid="a", r5=1.0, r10=1.0, rr=1.0),
        _result(qid="b", r5=0.0, r10=1.0, rr=0.5),
    ]

    row = _format_split_row("ko", items)

    assert "ko" in row
    assert "2" in row
    assert "0.500" in row  # avg r5
    assert "1.000" in row  # avg r10
    assert "0.750" in row  # avg rr


def test_build_split_table_includes_all_groups_and_overall() -> None:
    groups = {
        "ko": [_result(qid="k1", r5=0, r10=1, rr=0.5)],
        "en": [_result(qid="e1", r5=1, r10=1, rr=1)],
    }

    lines = _build_split_table("언어별", groups)

    assert "### 언어별" in lines
    joined = "\n".join(lines)
    assert "| ko |" in joined
    assert "| en |" in joined
    assert "**전체**" in joined


def test_build_failure_section_no_failures() -> None:
    lines = _build_failure_section([])

    joined = "\n".join(lines)
    assert "(없음)" in joined


def test_build_failure_section_includes_query_metadata() -> None:
    chunk = Chunk(text="t", section="warnings", drug_name="ASPIRIN", source="fda_label")
    failures = [
        _result(
            qid="ko-pain-001",
            r5=0,
            r10=0,
            rr=0,
            drug_id="타이레놀",
            ranked=[chunk],
            expected_drug_names=["타이레놀"],
            query_text="타이레놀 부작용",
        )
    ]

    lines = _build_failure_section(failures)
    joined = "\n".join(lines)

    assert "ko-pain-001" in joined
    assert "타이레놀 부작용" in joined
    assert "['타이레놀']" in joined or "['타이레놀']" in joined
    assert "drug_id 필터 적용" in joined
    assert "ASPIRIN" in joined
    assert "warnings" in joined


def test_percentile_basics() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]

    assert _percentile(values, 0.0) == 1.0
    assert _percentile(values, 1.0) == 5.0
    assert _percentile([], 0.5) == 0.0
