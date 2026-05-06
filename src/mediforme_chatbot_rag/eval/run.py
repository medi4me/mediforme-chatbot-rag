"""검색 성능 평가 러너

- queries.yaml + FAISS 인덱스 로드 → 각 쿼리 검색 → Recall@K · MRR · Latency 집계
- 언어 / 카테고리 / 난이도별 분할 점수 출력
- --use-drug-id-filter 옵션으로 정규화 검색 모드 측정
- 실패 쿼리(Recall@10 == 0) 자동 추출 + 리포트 별도 섹션
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import yaml

from mediforme_chatbot_rag.eval.metrics import recall_at_k, reciprocal_rank
from mediforme_chatbot_rag.ingestion.embedder import Embedder
from mediforme_chatbot_rag.ingestion.index import FaissIndex
from mediforme_chatbot_rag.services.retrieval import RetrievalService


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mediforme_chatbot_rag.eval.run",
        description="검색 성능 평가 — Recall@K · MRR · Latency · 분할 점수",
    )
    parser.add_argument("--queries", type=Path, required=True, help="queries.yaml 경로")
    parser.add_argument("--index", type=Path, required=True, help="FAISS 인덱스 디렉터리")
    parser.add_argument("--top-k", type=int, default=10, help="검색 top-K (기본 10)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval/results.md"),
        help="마크다운 리포트 출력 경로",
    )
    parser.add_argument(
        "--use-drug-id-filter",
        action="store_true",
        help="expected_drug_id 가 있는 쿼리는 drug_id 필터를 적용해 검색",
    )
    return parser.parse_args(argv)


def _load_queries(path: Path) -> list[dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path} 의 최상위는 list 여야 함")
    return raw


def _evaluate_query(
    query: dict[str, Any],
    *,
    service: RetrievalService,
    top_k: int,
    use_drug_id_filter: bool,
) -> dict[str, Any]:
    """
    쿼리 1개를 실행하고 결과 dict 를 돌려준다
    """
    drug_id = query.get("expected_drug_id") if use_drug_id_filter else None

    start = time.perf_counter()
    chunks_with_scores = service.retrieve(
        query["query"],
        top_k=top_k,
        drug_id=drug_id,
    )
    latency_ms = (time.perf_counter() - start) * 1000.0

    ranked = [c for c, _ in chunks_with_scores]
    expected_names = query["expected_drug_names"]
    expected_sections = query.get("expected_sections")

    r5 = recall_at_k(
        ranked,
        expected_drug_names=expected_names,
        expected_sections=expected_sections,
        k=5,
    )
    r10 = recall_at_k(
        ranked,
        expected_drug_names=expected_names,
        expected_sections=expected_sections,
        k=10,
    )
    rr = reciprocal_rank(
        ranked,
        expected_drug_names=expected_names,
        expected_sections=expected_sections,
    )

    return {
        "query": query,
        "ranked": ranked,
        "r5": r5,
        "r10": r10,
        "rr": rr,
        "latency_ms": latency_ms,
        "drug_id_used": drug_id,
    }


def _format_split_row(label: str, items: list[dict[str, Any]]) -> str:
    if not items:
        return f"| {label} | 0 | - | - | - |"
    n = len(items)
    avg_r5 = sum(it["r5"] for it in items) / n
    avg_r10 = sum(it["r10"] for it in items) / n
    avg_mrr = sum(it["rr"] for it in items) / n
    return f"| {label} | {n} | {avg_r5:.3f} | {avg_r10:.3f} | {avg_mrr:.3f} |"


def _build_split_table(
    title: str,
    groups: dict[str, list[dict[str, Any]]],
) -> list[str]:
    all_items = [item for items in groups.values() for item in items]
    lines = [
        f"### {title}",
        "",
        "| 분할 | N | Recall@5 | Recall@10 | MRR |",
        "|---|---|---|---|---|",
    ]
    for key in sorted(groups.keys()):
        lines.append(_format_split_row(key, groups[key]))
    lines.append(_format_split_row("**전체**", all_items))
    lines.append("")
    return lines


def _failure_entries(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Recall@10 == 0 인 쿼리만 골라 반환
    """
    return [r for r in results if r["r10"] == 0.0]


def _build_failure_section(failures: list[dict[str, Any]]) -> list[str]:
    if not failures:
        return ["## 실패 케이스 (Recall@10 == 0)", "", "(없음)", ""]
    lines = ["## 실패 케이스 (Recall@10 == 0)", ""]
    for r in failures:
        q = r["query"]
        lines.append(f"### {q['id']} — `{q['query']}`")
        lines.append(f"- expected_drug_names: {q['expected_drug_names']}")
        if r["drug_id_used"]:
            lines.append(f"- drug_id 필터 적용: `{r['drug_id_used']}`")
        lines.append("- top-3 결과:")
        for i, chunk in enumerate(r["ranked"][:3], start=1):
            lines.append(f"  {i}. `{chunk.drug_name}` / `{chunk.section}` (source={chunk.source})")
        lines.append("")
    return lines


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = min(len(sorted_vals) - 1, round(p * (len(sorted_vals) - 1)))
    return sorted_vals[idx]


def _build_report(
    results: list[dict[str, Any]],
    latencies_ms: list[float],
    *,
    top_k: int,
    use_drug_id_filter: bool,
) -> str:
    filter_label = "ON" if use_drug_id_filter else "OFF"
    lines = [
        "# 검색 성능 평가 결과",
        "",
        f"top-K = {top_k}, N = {len(results)}, drug_id 필터 = {filter_label}",
        "",
        "## 지표",
        "",
    ]

    by_lang: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_difficulty: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        q = r["query"]
        by_lang[q.get("language", "unknown")].append(r)
        by_category[q.get("category", "unknown")].append(r)
        by_difficulty[q.get("difficulty", "unknown")].append(r)

    lines.extend(_build_split_table("언어별", dict(by_lang)))
    lines.extend(_build_split_table("카테고리별", dict(by_category)))
    lines.extend(_build_split_table("난이도별", dict(by_difficulty)))

    lines.append("## Latency (검색 단일 호출)")
    lines.append("")
    if latencies_ms:
        p50 = median(latencies_ms)
        p95 = _percentile(latencies_ms, 0.95)
        lines.append(f"- p50: {p50:.1f} ms")
        lines.append(f"- p95: {p95:.1f} ms")
    lines.append("")

    lines.extend(_build_failure_section(_failure_entries(results)))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    queries = _load_queries(args.queries)
    index = FaissIndex.load(args.index)
    embedder = Embedder()

    # 모델 사전 로드 — Latency 측정에서 첫 호출 모델 로딩 시간 제거
    embedder.embed(["warmup"])

    service = RetrievalService(embedder=embedder, index=index)

    results: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    for q in queries:
        result = _evaluate_query(
            q,
            service=service,
            top_k=args.top_k,
            use_drug_id_filter=args.use_drug_id_filter,
        )
        results.append(result)
        latencies_ms.append(result["latency_ms"])

    report = _build_report(
        results,
        latencies_ms,
        top_k=args.top_k,
        use_drug_id_filter=args.use_drug_id_filter,
    )
    print(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report + "\n", encoding="utf-8")
    print(f"\n결과 리포트 저장: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
