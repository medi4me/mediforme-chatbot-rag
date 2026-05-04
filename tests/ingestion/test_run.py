"""ingest CLI 단위 테스트

- 페처·임베더 모킹으로 오케스트레이션 흐름 검증
- 실 API 호출 통합 테스트는 @pytest.mark.integration 으로 분리
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mediforme_chatbot_rag.core import config
from mediforme_chatbot_rag.ingestion import run
from mediforme_chatbot_rag.ingestion.fda_fetcher import FdaFetchError, FdaLabel
from mediforme_chatbot_rag.ingestion.index import CHUNKS_FILENAME, INDEX_FILENAME, FaissIndex
from mediforme_chatbot_rag.ingestion.mfds_fetcher import MfdsFetchError, MfdsLabel


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    config.get_settings.cache_clear()


def _fda_label(name: str = "TYLENOL") -> FdaLabel:
    return FdaLabel(
        drug_name=name,
        sections={"warnings": ["Consult a doctor if symptoms persist for more than seven days."]},
        set_id="test-set-id",
        effective_time="20240101",
    )


def _mfds_label(name: str = "타이레놀정500밀리그람") -> MfdsLabel:
    return MfdsLabel(
        drug_name=name,
        sections={
            "효능효과": [
                "이 약은 해열, 진통, 두통, 치통, 근육통의 일시적 완화에 사용합니다. "
                "자세한 정보는 의사 또는 약사에게 상의하시기 바랍니다."
            ]
        },
        item_seq="199101247",
        permit_date="",
    )


class _FixedModel:
    """
    sentence_transformers 인터페이스 흉내, 모든 입력에 같은 벡터를 돌려주는 가짜
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> Any:
        return np.tile(np.arange(self.dim, dtype=np.float32), (len(sentences), 1))


@pytest.fixture
def _patch_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Embedder 의 sentence_transformers 로더를 가짜로 교체해 실모델 다운로드 차단
    """
    from mediforme_chatbot_rag.ingestion import embedder as embedder_mod

    def fake_loader(_name: str) -> _FixedModel:
        return _FixedModel(config.get_settings().embedding_dimensions)

    monkeypatch.setattr(embedder_mod, "_load_sentence_transformer", fake_loader)


def test_load_drug_list_skips_comments_and_blanks(tmp_path: Path) -> None:
    file = tmp_path / "drugs.txt"
    file.write_text(
        "# comment\ntylenol\n\n  ibuprofen  \n# another\naspirin\n",
        encoding="utf-8",
    )

    drugs = run._load_drug_list(file)

    assert drugs == ["tylenol", "ibuprofen", "aspirin"]


def test_load_drug_list_returns_empty_when_path_none() -> None:
    assert run._load_drug_list(None) == []


def test_load_drug_list_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run._load_drug_list(tmp_path / "nonexistent.txt")


async def test_collect_chunks_fda_continues_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_fetch(query: str, **_: Any) -> FdaLabel:
        if query == "bad":
            raise FdaFetchError("not found")
        return _fda_label(query.upper())

    monkeypatch.setattr(run, "fetch_fda", fake_fetch)

    chunks = await run._collect_chunks_fda(["tylenol", "bad", "ibuprofen"])

    drug_names = {c.drug_name for c in chunks}
    assert drug_names == {"TYLENOL", "IBUPROFEN"}


async def test_collect_chunks_mfds_continues_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_fetch(query: str, **_: Any) -> MfdsLabel:
        if query == "없는약":
            raise MfdsFetchError("itemName 없음")
        return _mfds_label(query)

    monkeypatch.setattr(run, "fetch_mfds", fake_fetch)

    chunks = await run._collect_chunks_mfds(["타이레놀", "없는약", "게보린"])

    drug_names = {c.drug_name for c in chunks}
    assert drug_names == {"타이레놀", "게보린"}


def test_main_no_input_returns_exit_code_2() -> None:
    assert run.main([]) == 2


def test_main_all_failures_returns_exit_code_1(
    monkeypatch: pytest.MonkeyPatch,
    _patch_embedder: None,
) -> None:
    async def fail_fda(_query: str, **_: Any) -> FdaLabel:
        raise FdaFetchError("fail")

    monkeypatch.setattr(run, "fetch_fda", fail_fda)

    assert run.main(["--fda-drug", "tylenol"]) == 1


def test_main_end_to_end_writes_index(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    _patch_embedder: None,
) -> None:
    async def fake_fda(query: str, **_: Any) -> FdaLabel:
        return _fda_label(query.upper())

    async def fake_mfds(query: str, **_: Any) -> MfdsLabel:
        return _mfds_label(query)

    monkeypatch.setattr(run, "fetch_fda", fake_fda)
    monkeypatch.setattr(run, "fetch_mfds", fake_mfds)

    output = tmp_path / "index"
    rc = run.main(
        [
            "--fda-drug",
            "tylenol",
            "--mfds-drug",
            "타이레놀",
            "--output",
            str(output),
        ]
    )

    assert rc == 0
    assert (output / INDEX_FILENAME).exists()
    assert (output / CHUNKS_FILENAME).exists()

    loaded = FaissIndex.load(output)
    sources = {c.source for c, _ in loaded.search([1.0] * loaded.dim, top_k=10)}
    assert sources == {"fda_label", "mfds_label"}


def test_main_drugs_file_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    _patch_embedder: None,
) -> None:
    drugs_file = tmp_path / "fda.txt"
    drugs_file.write_text("# header\ntylenol\nibuprofen\n", encoding="utf-8")

    async def fake_fda(query: str, **_: Any) -> FdaLabel:
        return _fda_label(query.upper())

    monkeypatch.setattr(run, "fetch_fda", fake_fda)

    rc = run.main(
        [
            "--fda-drugs-file",
            str(drugs_file),
            "--output",
            str(tmp_path / "index"),
        ]
    )

    assert rc == 0


@pytest.mark.integration
def test_main_real_apis(tmp_path: Path) -> None:
    """
    실 FDA + MFDS API + 실 SBERT 모델 호출

    - pytest -m integration + MFDS_API_KEY 환경변수 필요
    """
    output = tmp_path / "index"
    rc = run.main(
        [
            "--fda-drug",
            "tylenol",
            "--mfds-drug",
            "타이레놀",
            "--output",
            str(output),
        ]
    )
    assert rc == 0
    assert (output / INDEX_FILENAME).exists()
