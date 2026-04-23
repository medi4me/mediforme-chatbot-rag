"""openFDA 페처 단위 테스트."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from mediforme_chatbot_rag.core import config
from mediforme_chatbot_rag.ingestion.fda_fetcher import (
    FdaFetchError,
    FdaLabel,
    fetch_label,
)

_BASE = "https://api.fda.gov/drug/label.json"


def _label_response(
    brand: str | None = "TYLENOL",
    generic: str | None = "ACETAMINOPHEN",
) -> dict[str, Any]:
    openfda: dict[str, list[str]] = {}
    if brand is not None:
        openfda["brand_name"] = [brand]
    if generic is not None:
        openfda["generic_name"] = [generic]

    return {
        "meta": {},
        "results": [
            {
                "set_id": "abcd-1234",
                "effective_time": "20240101",
                "openfda": openfda,
                "indications_and_usage": ["For temporary relief of minor aches."],
                "contraindications": ["Do not use if you have severe hepatic impairment."],
                "warnings": ["Consult a doctor if symptoms persist."],
            }
        ],
    }


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    config.get_settings.cache_clear()


@pytest.fixture
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    from mediforme_chatbot_rag.ingestion import fda_fetcher as mod

    async def _skip(_seconds: float) -> None:
        return None

    monkeypatch.setattr(mod.asyncio, "sleep", _skip)


async def test_fetch_label_happy_path(respx_mock: respx.Router) -> None:
    respx_mock.get(_BASE).mock(return_value=httpx.Response(200, json=_label_response()))

    label = await fetch_label("tylenol")

    assert isinstance(label, FdaLabel)
    assert label.drug_name == "TYLENOL"
    assert label.set_id == "abcd-1234"
    assert label.effective_time == "20240101"
    assert "indications_and_usage" in label.sections
    assert "contraindications" in label.sections
    assert "openfda" not in label.sections


async def test_fetch_label_uses_generic_when_brand_missing(
    respx_mock: respx.Router,
) -> None:
    respx_mock.get(_BASE).mock(return_value=httpx.Response(200, json=_label_response(brand=None)))

    label = await fetch_label("acetaminophen")

    assert label.drug_name == "ACETAMINOPHEN"


async def test_fetch_label_no_results_raises(respx_mock: respx.Router) -> None:
    respx_mock.get(_BASE).mock(return_value=httpx.Response(200, json={"results": []}))

    with pytest.raises(FdaFetchError, match="라벨이 없음"):
        await fetch_label("nonexistent")


async def test_fetch_label_404_raises(respx_mock: respx.Router) -> None:
    respx_mock.get(_BASE).mock(
        return_value=httpx.Response(404, json={"error": {"code": "NOT_FOUND"}})
    )

    with pytest.raises(FdaFetchError):
        await fetch_label("nonexistent")


async def test_fetch_label_retries_on_429_then_succeeds(
    respx_mock: respx.Router,
    _no_sleep: None,
) -> None:
    route = respx_mock.get(_BASE).mock(
        side_effect=[
            httpx.Response(429),
            httpx.Response(429),
            httpx.Response(200, json=_label_response()),
        ]
    )

    label = await fetch_label("tylenol")

    assert route.call_count == 3
    assert label.drug_name == "TYLENOL"


async def test_fetch_label_retries_exhausted_on_5xx(
    respx_mock: respx.Router,
    _no_sleep: None,
) -> None:
    respx_mock.get(_BASE).mock(return_value=httpx.Response(500))

    with pytest.raises(FdaFetchError, match="재시도 초과"):
        await fetch_label("tylenol")


async def test_fetch_label_attaches_api_key_when_configured(
    respx_mock: respx.Router,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENFDA_API_KEY", "test-key")
    config.get_settings.cache_clear()

    route = respx_mock.get(_BASE).mock(return_value=httpx.Response(200, json=_label_response()))

    await fetch_label("tylenol")

    assert "api_key=test-key" in str(route.calls.last.request.url)


@pytest.mark.integration
async def test_fetch_label_real_openfda() -> None:
    """실 openFDA 호출 — `pytest -m integration` 으로만 실행."""
    label = await fetch_label("tylenol")
    assert label.drug_name.lower() in {"tylenol", "acetaminophen"}
    assert label.sections
