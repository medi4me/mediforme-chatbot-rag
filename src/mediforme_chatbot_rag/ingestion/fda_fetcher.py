"""
openFDA 라벨 페처
- openFDA /drug/label.json API를 호출해 약품 라벨을 FdaLabel로 매핑
- 네트워크 계층만 담당하며, 섹션 파싱·청킹은 후속 모듈에서 수행
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from pydantic import BaseModel, Field

from mediforme_chatbot_rag.core.config import get_settings

_MAX_RETRIES = 3
_BACKOFF_BASE_SECONDS = 1.0
_REQUEST_TIMEOUT_SECONDS = 10.0
_LABEL_META_KEYS = frozenset(
    {
        "openfda",
        "set_id",
        "effective_time",
        "id",
        "version",
        "spl_id",
        "spl_set_id",
        "spl_product_data_elements",
    }
)


class FdaLabel(BaseModel):
    """openFDA 라벨 1건의 정규화된 표현."""

    drug_name: str
    sections: dict[str, list[str]] = Field(default_factory=dict)
    set_id: str
    effective_time: str


class FdaFetchError(Exception):
    """openFDA 호출 실패."""


async def fetch_label(
    query: str,
    *,
    client: httpx.AsyncClient | None = None,
) -> FdaLabel:
    """brand_name 또는 generic_name 으로 openFDA 라벨을 1건 가져온다."""
    settings = get_settings()
    params: dict[str, Any] = {
        "search": (f'openfda.brand_name:"{query}" OR openfda.generic_name:"{query}"'),
        "limit": 1,
    }
    if settings.openfda_api_key:
        params["api_key"] = settings.openfda_api_key

    url = f"{settings.openfda_base_url}/drug/label.json"

    if client is None:
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECONDS) as owned:
            response = await _get_with_retry(owned, url, params)
    else:
        response = await _get_with_retry(client, url, params)

    payload = response.json()
    results = payload.get("results") or []
    if not results:
        raise FdaFetchError(f"openFDA 에 '{query}' 라벨이 없음")

    return _to_label(results[0])


async def _get_with_retry(
    client: httpx.AsyncClient,
    url: str,
    params: dict[str, Any],
) -> httpx.Response:
    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = await client.get(url, params=params)
        except httpx.HTTPError as exc:
            if attempt < _MAX_RETRIES:
                await asyncio.sleep(_BACKOFF_BASE_SECONDS * (2**attempt))
                continue
            raise FdaFetchError(f"openFDA 호출 실패: {exc}") from exc

        status = response.status_code
        if status == 429 or 500 <= status < 600:
            if attempt < _MAX_RETRIES:
                await asyncio.sleep(_BACKOFF_BASE_SECONDS * (2**attempt))
                continue
            raise FdaFetchError(f"openFDA 재시도 초과: status={status}")

        if status == 404:
            raise FdaFetchError(f"openFDA 404 (params={params})")

        response.raise_for_status()
        return response

    raise FdaFetchError("openFDA 페처가 예상치 못한 상태로 종료됨")


def _to_label(result: dict[str, Any]) -> FdaLabel:
    openfda = result.get("openfda") or {}
    brand = (openfda.get("brand_name") or [None])[0]
    generic = (openfda.get("generic_name") or [None])[0]
    drug_name = brand or generic
    if drug_name is None:
        raise FdaFetchError("라벨에 brand_name·generic_name 모두 없음")

    set_id = _coerce_str(result.get("set_id"))
    effective_time = _coerce_str(result.get("effective_time"))

    sections: dict[str, list[str]] = {}
    for key, value in result.items():
        if key in _LABEL_META_KEYS:
            continue
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            sections[key] = value

    return FdaLabel(
        drug_name=drug_name,
        sections=sections,
        set_id=set_id,
        effective_time=effective_time,
    )


def _coerce_str(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value and isinstance(value[0], str):
        return value[0]
    return ""
