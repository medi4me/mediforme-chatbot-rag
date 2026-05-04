"""ingest 오케스트레이터 CLI

- FDA / MFDS 페처를 묶어 fetch → chunk → embed → FAISS 인덱스 저장까지 한 번에 실행
- 약 단발(--fda-drug / --mfds-drug) 또는 파일 기반(--fda-drugs-file / --mfds-drugs-file)
- 한 약의 fetch 실패는 로그 후 다음으로 진행, 청크 0 이면 abort
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

from mediforme_chatbot_rag.core.config import get_settings
from mediforme_chatbot_rag.ingestion.chunker import Chunk, chunk_label
from mediforme_chatbot_rag.ingestion.embedder import Embedder
from mediforme_chatbot_rag.ingestion.fda_fetcher import FdaFetchError
from mediforme_chatbot_rag.ingestion.fda_fetcher import fetch_label as fetch_fda
from mediforme_chatbot_rag.ingestion.index import FaissIndex
from mediforme_chatbot_rag.ingestion.mfds_fetcher import MfdsFetchError
from mediforme_chatbot_rag.ingestion.mfds_fetcher import fetch_label as fetch_mfds


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mediforme_chatbot_rag.ingestion.run",
        description="FDA / MFDS 라벨을 fetch → chunk → embed → FAISS 인덱스로 저장",
    )
    parser.add_argument(
        "--fda-drug",
        action="append",
        default=[],
        help="FDA 약 단건 (반복 가능)",
    )
    parser.add_argument(
        "--mfds-drug",
        action="append",
        default=[],
        help="MFDS 약 단건 (반복 가능)",
    )
    parser.add_argument(
        "--fda-drugs-file",
        type=Path,
        default=None,
        help="FDA 약 리스트 파일 (한 줄에 하나, # 주석)",
    )
    parser.add_argument(
        "--mfds-drugs-file",
        type=Path,
        default=None,
        help="MFDS 약 리스트 파일 (한 줄에 하나, # 주석)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="인덱스 저장 디렉터리 (기본: Settings.index_dir)",
    )
    return parser.parse_args(argv)


def _load_drug_list(path: Path | None) -> list[str]:
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"drug list 파일이 없음: {path}")
    drugs: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        drugs.append(line)
    return drugs


async def _collect_chunks_fda(drugs: list[str]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for drug in drugs:
        try:
            label = await fetch_fda(drug)
        except FdaFetchError as exc:
            print(f"[FDA] {drug} 실패: {exc}", file=sys.stderr)
            continue
        added = chunk_label(label, source="fda_label")
        chunks.extend(added)
        print(f"[FDA] {drug} → {len(added)} chunks")
    return chunks


async def _collect_chunks_mfds(drugs: list[str]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for drug in drugs:
        try:
            label = await fetch_mfds(drug)
        except MfdsFetchError as exc:
            print(f"[MFDS] {drug} 실패: {exc}", file=sys.stderr)
            continue
        added = chunk_label(label, source="mfds_label")
        chunks.extend(added)
        print(f"[MFDS] {drug} → {len(added)} chunks")
    return chunks


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    output: Path = args.output or Path(settings.index_dir)

    fda_drugs = list(args.fda_drug) + _load_drug_list(args.fda_drugs_file)
    mfds_drugs = list(args.mfds_drug) + _load_drug_list(args.mfds_drugs_file)

    if not fda_drugs and not mfds_drugs:
        print(
            "ingest 대상이 없음. --fda-drug/--mfds-drug 또는 "
            "--fda-drugs-file/--mfds-drugs-file 지정 필요",
            file=sys.stderr,
        )
        return 2

    print(f"FDA {len(fda_drugs)}개 + MFDS {len(mfds_drugs)}개 ingest 시작")

    chunks: list[Chunk] = []
    chunks.extend(await _collect_chunks_fda(fda_drugs))
    chunks.extend(await _collect_chunks_mfds(mfds_drugs))

    if not chunks:
        print("수집된 청크가 없어 인덱스 저장 안 함", file=sys.stderr)
        return 1

    print(f"총 {len(chunks)} 청크 수집. 임베딩 시작...")

    embedder = Embedder()
    embed_start = time.monotonic()
    embeddings = embedder.embed([c.text for c in chunks])
    embed_seconds = time.monotonic() - embed_start
    print(f"임베딩 완료 ({embed_seconds:.1f}s)")

    index = FaissIndex(dim=settings.embedding_dimensions)
    index.add(chunks, embeddings)
    index.save(output)
    print(f"인덱스 저장 완료: {output} (청크 {len(index)}개)")

    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
