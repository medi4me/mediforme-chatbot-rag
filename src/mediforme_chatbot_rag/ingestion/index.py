"""FAISS 인덱스 빌더

- IndexFlatIP + L2 정규화로 코사인 유사도 검색
- Chunk 메타는 JSON 파일로 별도 저장
- save / load 로 디스크 영속화
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from mediforme_chatbot_rag.ingestion.chunker import Chunk

INDEX_FILENAME = "index.faiss"
CHUNKS_FILENAME = "chunks.json"


class FaissIndex:
    """
    FAISS IndexFlatIP 기반 코사인 유사도 인덱스
    """

    def __init__(self, dim: int) -> None:
        import faiss

        self._dim = dim
        self._index: Any = faiss.IndexFlatIP(dim)
        self._chunks: list[Chunk] = []

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def dim(self) -> int:
        return self._dim

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks 와 embeddings 길이가 다름: {len(chunks)} vs {len(embeddings)}"
            )
        if not chunks:
            return

        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.shape[1] != self._dim:
            raise ValueError(
                f"임베딩 차원 불일치: 인덱스 dim={self._dim}, 입력 dim={vectors.shape[1]}"
            )
        _normalize_in_place(vectors)
        self._index.add(vectors)
        self._chunks.extend(chunks)

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[tuple[Chunk, float]]:
        if not self._chunks or top_k <= 0:
            return []

        query = np.asarray([query_embedding], dtype=np.float32)
        if query.shape[1] != self._dim:
            raise ValueError(
                f"쿼리 임베딩 차원 불일치: 인덱스 dim={self._dim}, 쿼리 dim={query.shape[1]}"
            )
        _normalize_in_place(query)

        k = min(top_k, len(self._chunks))
        scores, indices = self._index.search(query, k)
        results: list[tuple[Chunk, float]] = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx < 0:
                continue
            results.append((self._chunks[int(idx)], float(score)))
        return results

    def save(self, path: Path) -> None:
        import faiss

        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / INDEX_FILENAME))
        chunks_data = [c.model_dump() for c in self._chunks]
        (path / CHUNKS_FILENAME).write_text(
            json.dumps(chunks_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> FaissIndex:
        import faiss

        loaded_index = faiss.read_index(str(path / INDEX_FILENAME))
        chunks_data = json.loads((path / CHUNKS_FILENAME).read_text(encoding="utf-8"))

        instance = cls.__new__(cls)
        instance._dim = int(loaded_index.d)
        instance._index = loaded_index
        instance._chunks = [Chunk(**c) for c in chunks_data]
        return instance


def _normalize_in_place(vectors: Any) -> None:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors /= norms
