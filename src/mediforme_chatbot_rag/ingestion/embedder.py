"""SBERT 다국어 임베더

- Sentence-Transformers 다국어 모델로 텍스트를 벡터로 임베딩 진행
- 모델은 첫 embed 호출 시 lazy load
"""

from __future__ import annotations

from threading import Lock
from typing import TYPE_CHECKING, Any, Protocol, cast

from mediforme_chatbot_rag.core.config import get_settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class _EncoderProtocol(Protocol):
    """
    SentenceTransformer 의 encode 시그니처를 흉내 내는 최소 인터페이스
    """

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> Any: ...


class Embedder:
    """
    SBERT 기반 다국어 임베더
    """

    def __init__(
        self,
        *,
        model_name: str | None = None,
        batch_size: int | None = None,
        model: _EncoderProtocol | None = None,
    ) -> None:
        settings = get_settings()
        self._model_name = model_name or settings.embedding_model_name
        self._batch_size = batch_size or settings.embedding_batch_size
        self._model: _EncoderProtocol | None = model
        self._load_lock = Lock()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [list(map(float, row)) for row in embeddings]

    def _get_model(self) -> _EncoderProtocol:
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    self._model = cast(
                        _EncoderProtocol,
                        _load_sentence_transformer(self._model_name),
                    )
        model = self._model
        assert model is not None
        return model


_MAX_SEQ_LENGTH_CAP = 1024


def _load_sentence_transformer(model_name: str) -> SentenceTransformer:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    # BGE-M3 처럼 default max_seq_length 가 8192 인 모델은 attention 단계에서
    # seq² 비례로 OOM 이 나기 쉬워 합리적인 값으로 캡
    model.max_seq_length = min(model.max_seq_length, _MAX_SEQ_LENGTH_CAP)
    return model  # type: ignore[no-any-return]
