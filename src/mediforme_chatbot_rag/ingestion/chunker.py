"""FDA 라벨 섹션 청커

FdaLabel 을 검색 단위 청크 리스트로 변환한다
기본 전략은 섹션 1개 = 청크 1개이며, 너무 긴 섹션은 문장 경계로 분할하고
너무 짧은 섹션은 스킵
"""

from __future__ import annotations

import re
from typing import Protocol

from pydantic import BaseModel

MAX_CHUNK_CHARS = 2000
MIN_CHUNK_CHARS = 50

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


class _LabelLike(Protocol):
    """
    FdaLabel / MfdsLabel 처럼 drug_name 과 sections 를 가진 라벨 구조
    """

    drug_name: str
    sections: dict[str, list[str]]


class Chunk(BaseModel):
    """
    검색 단위 텍스트 청크
    """

    text: str
    section: str
    drug_name: str
    source: str = "fda_label"


def chunk_label(label: _LabelLike, *, source: str = "fda_label") -> list[Chunk]:
    """
    라벨의 섹션을 검색 단위 청크 리스트로 변환

    - source 는 청크의 출처 식별자 (예: fda_label / mfds_label)
    """
    chunks: list[Chunk] = []
    for section, parts in label.sections.items():
        text = "\n".join(part.strip() for part in parts if part.strip())
        if len(text) < MIN_CHUNK_CHARS:
            continue
        for piece in _split_if_too_long(text, MAX_CHUNK_CHARS):
            chunks.append(
                Chunk(
                    text=piece,
                    section=section,
                    drug_name=label.drug_name,
                    source=source,
                )
            )
    return chunks


def _split_if_too_long(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    sentences = _SENTENCE_BOUNDARY.split(text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        addition = len(sentence) + (1 if current else 0)
        if current and current_len + addition > max_chars:
            chunks.append(" ".join(current))
            current = [sentence]
            current_len = len(sentence)
        else:
            current.append(sentence)
            current_len += addition
    if current:
        chunks.append(" ".join(current))
    return chunks
