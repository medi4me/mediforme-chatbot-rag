"""MFDS 페처 단위 테스트

- e약은요 응답 정상/에러 경로와 응답 wrapper 두 형식 모두 검증
- 실 MFDS 호출은 @pytest.mark.integration 마커로 분리
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from mediforme_chatbot_rag.core import config
from mediforme_chatbot_rag.ingestion.mfds_fetcher import (
    MfdsFetchError,
    MfdsLabel,
    fetch_label,
)

_BASE = "https://apis.data.go.kr/1471000/DrugPrdtPrmsnInfoService06/getDrugPrdtPermitDtlInq03"


def _label_payload(item_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {
        "itemName": "타이레놀정500밀리그람",
        "itemSeq": "199101247",
        "entpName": "한국얀센(주)",
        "efcyQesitm": "이 약은 해열, 진통, 두통에 사용합니다.",
        "useMethodQesitm": "성인은 1회 1정씩, 1일 3-4회 복용합니다.",
        "atpnWarnQesitm": "심한 간장애 환자는 사용하지 마십시오.",
        "atpnQesitm": "다른 의약품과 동시 복용시 주의하십시오.",
        "intrcQesitm": "와파린과 병용시 출혈 위험이 증가할 수 있습니다.",
        "seQesitm": "드물게 발진, 메스꺼움이 나타날 수 있습니다.",
        "depositMethodQesitm": "직사광선을 피해 보관하십시오.",
    }
    if item_overrides:
        item.update(item_overrides)
    return {
        "header": {"resultCode": "00", "resultMsg": "NORMAL SERVICE."},
        "body": {
            "pageNo": 1,
            "totalCount": 1,
            "numOfRows": 10,
            "items": [item],
        },
    }


@pytest.fixture(autouse=True)
def _set_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MFDS_API_KEY", "test-mfds-key")
    config.get_settings.cache_clear()


@pytest.fixture
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    from mediforme_chatbot_rag.ingestion import mfds_fetcher as mod

    async def _skip(_seconds: float) -> None:
        return None

    monkeypatch.setattr(mod.asyncio, "sleep", _skip)


async def test_fetch_label_happy_path(respx_mock: respx.Router) -> None:
    respx_mock.get(_BASE).mock(return_value=httpx.Response(200, json=_label_payload()))

    label = await fetch_label("타이레놀정500밀리그람")

    assert isinstance(label, MfdsLabel)
    assert label.drug_name == "타이레놀정500밀리그람"
    assert label.item_seq == "199101247"
    assert "효능효과" in label.sections
    assert "용법용량" in label.sections
    assert "부작용" in label.sections
    assert label.sections["효능효과"] == ["이 약은 해열, 진통, 두통에 사용합니다."]


async def test_fetch_label_handles_response_wrapper(respx_mock: respx.Router) -> None:
    payload = {"response": _label_payload()}
    respx_mock.get(_BASE).mock(return_value=httpx.Response(200, json=payload))

    label = await fetch_label("타이레놀")

    assert label.drug_name == "타이레놀정500밀리그람"


async def test_fetch_label_no_items_raises(respx_mock: respx.Router) -> None:
    payload = {"header": {}, "body": {"items": []}}
    respx_mock.get(_BASE).mock(return_value=httpx.Response(200, json=payload))

    with pytest.raises(MfdsFetchError, match="없음"):
        await fetch_label("존재하지않는약")


async def test_fetch_label_missing_item_name_raises(respx_mock: respx.Router) -> None:
    payload = _label_payload(item_overrides={"itemName": ""})
    respx_mock.get(_BASE).mock(return_value=httpx.Response(200, json=payload))

    with pytest.raises(MfdsFetchError, match="itemName"):
        await fetch_label("test")


async def test_fetch_label_without_api_key_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MFDS_API_KEY", "")
    config.get_settings.cache_clear()

    with pytest.raises(MfdsFetchError, match="MFDS_API_KEY"):
        await fetch_label("test")


async def test_fetch_label_retries_on_429_then_succeeds(
    respx_mock: respx.Router,
    _no_sleep: None,
) -> None:
    route = respx_mock.get(_BASE).mock(
        side_effect=[
            httpx.Response(429),
            httpx.Response(429),
            httpx.Response(200, json=_label_payload()),
        ]
    )

    label = await fetch_label("타이레놀")

    assert route.call_count == 3
    assert label.drug_name == "타이레놀정500밀리그람"


async def test_fetch_label_retries_exhausted_on_5xx(
    respx_mock: respx.Router,
    _no_sleep: None,
) -> None:
    respx_mock.get(_BASE).mock(return_value=httpx.Response(500))

    with pytest.raises(MfdsFetchError, match="재시도 초과"):
        await fetch_label("타이레놀")


async def test_fetch_label_passes_service_key_in_params(respx_mock: respx.Router) -> None:
    route = respx_mock.get(_BASE).mock(return_value=httpx.Response(200, json=_label_payload()))

    await fetch_label("타이레놀")

    assert "serviceKey=test-mfds-key" in str(route.calls.last.request.url)


async def test_fetch_label_skips_blank_section_fields(respx_mock: respx.Router) -> None:
    payload = _label_payload(item_overrides={"intrcQesitm": "", "seQesitm": "   "})
    respx_mock.get(_BASE).mock(return_value=httpx.Response(200, json=payload))

    label = await fetch_label("타이레놀")

    assert "상호작용" not in label.sections
    assert "부작용" not in label.sections
    assert "효능효과" in label.sections


@pytest.mark.integration
async def test_fetch_label_real_mfds() -> None:
    """
    실 MFDS 호출

    - pytest -m integration + 실제 MFDS_API_KEY 환경변수 필요
    """
    label = await fetch_label("타이레놀")
    assert label.drug_name
    assert label.sections
