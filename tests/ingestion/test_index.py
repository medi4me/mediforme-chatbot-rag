"""FAISS 인덱스 단위 테스트

- 작은 차원(4) 으로 add → search → save → load 라운드트립 검증
- 입력 검증(차원·길이 불일치) 케이스 포함
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mediforme_chatbot_rag.ingestion.chunker import Chunk
from mediforme_chatbot_rag.ingestion.index import FaissIndex


def _chunk(text: str, section: str = "warnings", drug_name: str = "TYLENOL") -> Chunk:
    return Chunk(text=text, section=section, drug_name=drug_name)


def test_empty_index_has_zero_length() -> None:
    index = FaissIndex(dim=4)
    assert len(index) == 0


def test_add_extends_length() -> None:
    index = FaissIndex(dim=4)
    chunks = [_chunk("hello"), _chunk("world")]
    embeddings = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]

    index.add(chunks, embeddings)

    assert len(index) == 2


def test_add_empty_lists_does_nothing() -> None:
    index = FaissIndex(dim=4)
    index.add([], [])
    assert len(index) == 0


def test_add_length_mismatch_raises() -> None:
    index = FaissIndex(dim=4)
    with pytest.raises(ValueError, match="길이가 다름"):
        index.add([_chunk("a")], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])


def test_add_dim_mismatch_raises() -> None:
    index = FaissIndex(dim=4)
    with pytest.raises(ValueError, match="차원 불일치"):
        index.add([_chunk("a")], [[1.0, 0.0]])


def test_search_returns_closest_chunk_first() -> None:
    index = FaissIndex(dim=4)
    chunks = [_chunk("first"), _chunk("second"), _chunk("third")]
    embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    index.add(chunks, embeddings)

    results = index.search([0.9, 0.1, 0.0, 0.0], top_k=2)

    assert len(results) == 2
    assert results[0][0].text == "first"
    assert results[0][1] >= results[1][1]


def test_search_empty_index_returns_empty() -> None:
    index = FaissIndex(dim=4)
    assert index.search([1.0, 0.0, 0.0, 0.0], top_k=5) == []


def test_search_top_k_zero_returns_empty() -> None:
    index = FaissIndex(dim=4)
    index.add([_chunk("a")], [[1.0, 0.0, 0.0, 0.0]])
    assert index.search([1.0, 0.0, 0.0, 0.0], top_k=0) == []


def test_search_dim_mismatch_raises() -> None:
    index = FaissIndex(dim=4)
    index.add([_chunk("a")], [[1.0, 0.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match="차원 불일치"):
        index.search([1.0, 0.0], top_k=1)


def test_save_load_roundtrip(tmp_path: Path) -> None:
    index = FaissIndex(dim=4)
    chunks = [
        _chunk("first", section="indications"),
        _chunk("second", section="warnings"),
    ]
    embeddings = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    index.add(chunks, embeddings)
    index.save(tmp_path)

    loaded = FaissIndex.load(tmp_path)

    assert len(loaded) == 2
    assert loaded.dim == 4

    results = loaded.search([1.0, 0.0, 0.0, 0.0], top_k=1)
    assert results[0][0].text == "first"
    assert results[0][0].section == "indications"
    assert results[0][0].drug_name == "TYLENOL"


def test_save_creates_index_and_chunks_files(tmp_path: Path) -> None:
    index = FaissIndex(dim=4)
    index.add([_chunk("a")], [[1.0, 0.0, 0.0, 0.0]])
    index.save(tmp_path)

    assert (tmp_path / "index.faiss").exists()
    assert (tmp_path / "chunks.json").exists()
