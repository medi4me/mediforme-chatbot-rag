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
    brand_name: str = ""
    generic_name: str = ""
    source: str = "fda_label"


def chunk_label(label: _LabelLike, *, source: str = "fda_label") -> list[Chunk]:
    """
    라벨의 섹션을 검색 단위 청크 리스트로 변환

    - source 는 청크의 출처 식별자 (예: fda_label / mfds_label)
    - 청크 텍스트 첫 줄에 약 이름 헤더를 prepend 해 의미 매칭 표면적을 넓힘
    """
    header = _build_header(label)
    body_max = max(MIN_CHUNK_CHARS, MAX_CHUNK_CHARS - len(header) - 2)
    brand = getattr(label, "brand_name", "") or ""
    generic = getattr(label, "generic_name", "") or ""
    chunks: list[Chunk] = []
    for section, parts in label.sections.items():
        text = "\n".join(part.strip() for part in parts if part.strip())
        if len(text) < MIN_CHUNK_CHARS:
            continue
        for piece in _split_if_too_long(text, body_max):
            chunks.append(
                Chunk(
                    text=f"{header}\n\n{piece}",
                    section=section,
                    drug_name=label.drug_name,
                    brand_name=brand,
                    generic_name=generic,
                    source=source,
                )
            )
    return chunks


def _build_header(label: _LabelLike) -> str:
    """
    drug_name 과 (있으면) brand_name / generic_name 을 합친 헤더 라인 생성
    """
    primary = label.drug_name
    aliases: list[str] = []
    for attr in ("brand_name", "generic_name"):
        candidate = getattr(label, attr, "") or ""
        if not candidate:
            continue
        if candidate.lower() == primary.lower():
            continue
        if any(candidate.lower() == a.lower() for a in aliases):
            continue
        aliases.append(candidate)
    if aliases:
        return f"[{primary} / {' / '.join(aliases)}]"
    return f"[{primary}]"


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
