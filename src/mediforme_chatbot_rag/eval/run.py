"""검색 성능 평가 러너

- queries.yaml + FAISS 인덱스 로드 → 각 쿼리 검색 → Recall@K · MRR · Latency 집계
- 언어별(ko/en) 분할 점수 출력
- stdout 표 + 마크다운 리포트(파일) 동시 출력
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mediforme_chatbot_rag.eval.run",
        description="검색 성능 평가 — Recall@K · MRR · Latency · 언어별 분할",
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
    return parser.parse_args(argv)


def _load_queries(path: Path) -> list[dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path} 의 최상위는 list 여야 함")
    return raw


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = min(len(sorted_vals) - 1, round(p * (len(sorted_vals) - 1)))
    return sorted_vals[idx]


def _format_split_row(label: str, items: list[dict[str, float]]) -> str:
    if not items:
        return f"| {label} | 0 | - | - | - |"
    n = len(items)
    avg_r5 = sum(it["r5"] for it in items) / n
    avg_r10 = sum(it["r10"] for it in items) / n
    avg_mrr = sum(it["rr"] for it in items) / n
    return f"| {label} | {n} | {avg_r5:.3f} | {avg_r10:.3f} | {avg_mrr:.3f} |"


def _build_report(
    results_by_lang: dict[str, list[dict[str, float]]],
    latencies_ms: list[float],
    *,
    top_k: int,
) -> str:
    all_results = [item for items in results_by_lang.values() for item in items]
    lines = [
        "# 검색 성능 평가 결과",
        "",
        f"top-K = {top_k}, N = {len(all_results)}",
        "",
        "## 지표",
        "",
        "| 분할 | N | Recall@5 | Recall@10 | MRR |",
        "|---|---|---|---|---|",
    ]
    for lang in sorted(results_by_lang.keys()):
        lines.append(_format_split_row(lang, results_by_lang[lang]))
    lines.append(_format_split_row("**전체**", all_results))
    lines.append("")
    lines.append("## Latency (검색 단일 호출)")
    lines.append("")
    if latencies_ms:
        p50 = median(latencies_ms)
        p95 = _percentile(latencies_ms, 0.95)
        lines.append(f"- p50: {p50:.1f} ms")
        lines.append(f"- p95: {p95:.1f} ms")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    queries = _load_queries(args.queries)
    index = FaissIndex.load(args.index)
    embedder = Embedder()

    # 모델 사전 로드 — Latency 측정에서 첫 호출 모델 로딩 시간 제거
    embedder.embed(["warmup"])

    results_by_lang: dict[str, list[dict[str, float]]] = defaultdict(list)
    latencies_ms: list[float] = []

    for q in queries:
        query_text = q["query"]
        expected_names = q["expected_drug_names"]
        expected_sections = q.get("expected_sections")
        language = q.get("language", "unknown")

        start = time.perf_counter()
        query_emb = embedder.embed([query_text])[0]
        chunks_with_scores = index.search(query_emb, top_k=args.top_k)
        latency_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(latency_ms)

        ranked = [c for c, _ in chunks_with_scores]
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
        results_by_lang[language].append({"r5": r5, "r10": r10, "rr": rr, "latency_ms": latency_ms})

    report = _build_report(results_by_lang, latencies_ms, top_k=args.top_k)
    print(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report + "\n", encoding="utf-8")
    print(f"\n결과 리포트 저장: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
