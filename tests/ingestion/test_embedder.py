"""SBERT 임베더 단위 테스트

- 모델 인스턴스를 주입해 실모델 다운로드 없이 인터페이스 검증
- 실 모델 호출은 @pytest.mark.integration 마커로 분리
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from mediforme_chatbot_rag.core import config
from mediforme_chatbot_rag.ingestion.embedder import Embedder


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    config.get_settings.cache_clear()


class _FakeModel:
    """
    encode 호출을 기록하고 고정 길이 임베딩을 돌려주는 테스트용 가짜
    """

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.calls: list[dict[str, Any]] = []

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> Any:
        self.calls.append(
            {
                "sentences": list(sentences),
                "batch_size": batch_size,
                "convert_to_numpy": convert_to_numpy,
                "show_progress_bar": show_progress_bar,
            }
        )
        return np.array(
            [[float(i) + 0.1 * j for j in range(self.dim)] for i in range(len(sentences))]
        )


def test_embed_empty_input_returns_empty_list() -> None:
    embedder = Embedder(model=_FakeModel())

    assert embedder.embed([]) == []


def test_embed_returns_list_of_lists_of_floats() -> None:
    fake = _FakeModel(dim=4)
    embedder = Embedder(model=fake)

    result = embedder.embed(["hello", "world"])

    assert len(result) == 2
    assert all(isinstance(row, list) for row in result)
    assert all(isinstance(v, float) for row in result for v in row)
    assert all(len(row) == 4 for row in result)


def test_embed_passes_batch_size_to_model() -> None:
    fake = _FakeModel()
    embedder = Embedder(model=fake, batch_size=8)

    embedder.embed(["a", "b", "c"])

    assert fake.calls[0]["batch_size"] == 8
    assert fake.calls[0]["sentences"] == ["a", "b", "c"]


def test_constructor_overrides_default_model_name() -> None:
    embedder = Embedder(model_name="custom-model", model=_FakeModel())

    assert embedder._model_name == "custom-model"


def test_lazy_load_only_on_first_embed_call(monkeypatch: pytest.MonkeyPatch) -> None:
    load_count = {"n": 0}

    def fake_loader(_model_name: str) -> Any:
        load_count["n"] += 1
        return _FakeModel()

    monkeypatch.setattr(
        "mediforme_chatbot_rag.ingestion.embedder._load_sentence_transformer",
        fake_loader,
    )

    embedder = Embedder()
    assert load_count["n"] == 0

    embedder.embed(["x"])
    assert load_count["n"] == 1

    embedder.embed(["y"])
    assert load_count["n"] == 1


def test_injected_model_skips_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    load_count = {"n": 0}

    def fake_loader(_model_name: str) -> Any:
        load_count["n"] += 1
        return _FakeModel()

    monkeypatch.setattr(
        "mediforme_chatbot_rag.ingestion.embedder._load_sentence_transformer",
        fake_loader,
    )

    embedder = Embedder(model=_FakeModel())
    embedder.embed(["x"])

    assert load_count["n"] == 0


@pytest.mark.integration
def test_real_sbert_model_embeds_text() -> None:
    """
    실 paraphrase-multilingual-mpnet-base-v2 모델로 임베딩

    - 첫 실행 시 모델 다운로드 발생 (~470MB)
    - `pytest -m integration` 으로만 실행
    """
    embedder = Embedder()

    result = embedder.embed(["타이레놀이 뭐예요?", "What is acetaminophen?"])

    assert len(result) == 2
    assert len(result[0]) == 768
    assert len(result[1]) == 768
