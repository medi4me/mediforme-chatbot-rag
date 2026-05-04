"""FDA 라벨 청커 단위 테스트."""

from __future__ import annotations

from mediforme_chatbot_rag.ingestion.chunker import (
    MAX_CHUNK_CHARS,
    Chunk,
    chunk_label,
)
from mediforme_chatbot_rag.ingestion.fda_fetcher import FdaLabel


def _label(sections: dict[str, list[str]], drug_name: str = "TYLENOL") -> FdaLabel:
    return FdaLabel(
        drug_name=drug_name,
        sections=sections,
        set_id="abcd-1234",
        effective_time="20240101",
    )


def test_each_section_becomes_one_chunk_when_short_enough() -> None:
    label = _label(
        {
            "indications_and_usage": [
                "For temporary relief of minor aches and headaches due to colds."
            ],
            "contraindications": [
                "Do not use if you have severe hepatic impairment or alcohol use disorder."
            ],
            "warnings": ["Consult a doctor if symptoms persist for more than seven days."],
        }
    )

    chunks = chunk_label(label)

    assert len(chunks) == 3
    assert [c.section for c in chunks] == [
        "indications_and_usage",
        "contraindications",
        "warnings",
    ]
    for c in chunks:
        assert isinstance(c, Chunk)
        assert c.drug_name == "TYLENOL"
        assert c.source == "fda_label"


def test_short_section_is_skipped() -> None:
    label = _label(
        {
            "tiny": ["None"],
            "indications_and_usage": [
                "For temporary relief of minor aches and headaches due to colds and flu."
            ],
        }
    )

    chunks = chunk_label(label)

    sections = [c.section for c in chunks]
    assert "tiny" not in sections
    assert "indications_and_usage" in sections


def test_long_section_is_split_by_sentence_boundary() -> None:
    sentence = (
        "This drug is used for relief of pain and reduction of fever in adults and children. "
    )
    long_text = sentence * 40

    label = _label({"warnings": [long_text]})

    chunks = chunk_label(label)

    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= MAX_CHUNK_CHARS
        assert c.section == "warnings"


def test_empty_parts_are_filtered_out() -> None:
    label = _label(
        {
            "indications_and_usage": [
                "",
                "  ",
                "For temporary relief of minor aches and headaches due to colds.",
            ]
        }
    )

    chunks = chunk_label(label)

    assert len(chunks) == 1
    assert "For temporary relief" in chunks[0].text
    assert "\n" not in chunks[0].text


def test_multi_part_section_is_joined_with_newline() -> None:
    label = _label(
        {
            "warnings": [
                "Consult a doctor if symptoms persist for more than seven days.",
                "Stop use and ask a doctor if pain or fever gets worse.",
            ]
        }
    )

    chunks = chunk_label(label)

    assert len(chunks) == 1
    assert "\n" in chunks[0].text
    assert "Consult a doctor" in chunks[0].text
    assert "Stop use" in chunks[0].text


def test_drug_name_propagates_to_chunks() -> None:
    label = _label(
        {
            "indications_and_usage": [
                "For temporary relief of minor aches and headaches due to colds."
            ]
        },
        drug_name="ACETAMINOPHEN",
    )

    chunks = chunk_label(label)

    assert all(c.drug_name == "ACETAMINOPHEN" for c in chunks)
