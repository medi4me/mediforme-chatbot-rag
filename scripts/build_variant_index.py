"""ablation 비교용 변형 인덱스 빌더

- 기존 data/index/chunks.json 의 코퍼스를 고정하고 모델·헤더만 바꿔 재임베딩
- 헤더 효과 측정을 위해 --strip-header 로 청크 text 첫 줄 약 이름 헤더 제거
- OOM/중단 대비로 샤드 단위 임베딩 후 .npy 체크포인트 저장 (재실행 시 이어서)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from mediforme_chatbot_rag.ingestion.chunker import Chunk
from mediforme_chatbot_rag.ingestion.embedder import Embedder
from mediforme_chatbot_rag.ingestion.index import FaissIndex

SRC = Path("data/index/chunks.json")


def strip_header(text: str) -> str:
    parts = text.split("\n\n", 1)
    if len(parts) == 2 and parts[0].startswith("["):
        return parts[1]
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--shard", type=int, default=2000)
    ap.add_argument("--strip-header", action="store_true")
    ap.add_argument("--checkpoint", type=Path, required=True)
    args = ap.parse_args()

    data = json.loads(SRC.read_text(encoding="utf-8"))
    chunks: list[Chunk] = []
    for c in data:
        text = strip_header(c["text"]) if args.strip_header else c["text"]
        chunks.append(
            Chunk(
                text=text,
                section=c["section"],
                drug_name=c["drug_name"],
                brand_name=c.get("brand_name", ""),
                generic_name=c.get("generic_name", ""),
                source=c.get("source", "fda_label"),
            )
        )
    n = len(chunks)
    args.checkpoint.mkdir(parents=True, exist_ok=True)
    embedder = Embedder(model_name=args.model, batch_size=args.batch_size)

    print(
        f"[build] model={args.model} n={n} shard={args.shard} "
        f"batch={args.batch_size} strip_header={args.strip_header}",
        flush=True,
    )
    t0 = time.time()
    for start in range(0, n, args.shard):
        shard_path = args.checkpoint / f"shard_{start:06d}.npy"
        if shard_path.exists():
            print(f"[skip] shard {start} 이미 있음", flush=True)
            continue
        texts = [c.text for c in chunks[start : start + args.shard]]
        vecs = embedder.embed(texts)
        np.save(shard_path, np.asarray(vecs, dtype=np.float32))
        done = min(start + args.shard, n)
        el = time.time() - t0
        eta = el / done * (n - done) / 60 if done else 0
        print(
            f"[shard] {done}/{n} ({100 * done / n:.0f}%) elapsed {el / 60:.1f}min eta {eta:.1f}min",
            flush=True,
        )

    print("[assemble] 샤드 로드", flush=True)
    all_vecs = [
        np.load(args.checkpoint / f"shard_{start:06d}.npy") for start in range(0, n, args.shard)
    ]
    embeddings = np.concatenate(all_vecs, axis=0)
    if embeddings.shape[0] != n:
        raise ValueError(f"임베딩 수 불일치: {embeddings.shape[0]} != {n}")
    index = FaissIndex(dim=args.dim)
    index.add(chunks, [list(map(float, row)) for row in embeddings])
    index.save(args.output)
    print(f"[done] 저장 완료: {args.output} ({len(index)} 청크)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
