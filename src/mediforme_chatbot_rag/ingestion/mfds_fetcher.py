"""MFDS 의약품 첨부문서 페처

- 식약처 의약품안전나라 e약은요 API (getDrugPrdtPermitDtlInq03) 호출
- MfdsLabel 모델로 정규화 (sections 는 효능효과 / 용법용량 / 주의사항 / 부작용 등)
- 네트워크 계층만 담당하며 청킹·임베딩은 후속 모듈에서 수행
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from pydantic import BaseModel, Field

from mediforme_chatbot_rag.core.config import get_settings

_ENDPOINT = "getDrugPrdtPermitDtlInq03"
_MAX_RETRIES = 3
_BACKOFF_BASE_SECONDS = 1.0
_REQUEST_TIMEOUT_SECONDS = 10.0

# e약은요 응답의 한글 친화 필드 → 한국어 섹션 키 매핑
_SECTION_FIELDS: dict[str, str] = {
    "efcyQesitm": "효능효과",
    "useMethodQesitm": "용법용량",
    "atpnWarnQesitm": "경고",
    "atpnQesitm": "사용상의 주의사항",
    "intrcQesitm": "상호작용",
    "seQesitm": "부작용",
    "depositMethodQesitm": "저장방법",
}


class MfdsLabel(BaseModel):
    """
    MFDS 의약품 첨부문서 1건의 정규화된 표현
    """

    drug_name: str
    sections: dict[str, list[str]] = Field(default_factory=dict)
    item_seq: str
    permit_date: str = ""


class MfdsFetchError(Exception):
    """
    MFDS API 호출 실패
    """


async def fetch_label(
    query: str,
    *,
    client: httpx.AsyncClient | None = None,
) -> MfdsLabel:
    """
    제품명(itemName) 으로 MFDS 의약품 첨부문서 1건 가져오기
    """
    settings = get_settings()
    if not settings.mfds_api_key:
        raise MfdsFetchError("MFDS_API_KEY 가 설정되지 않음")

    params: dict[str, Any] = {
        "serviceKey": settings.mfds_api_key,
        "itemName": query,
        "type": "json",
        "numOfRows": 1,
        "pageNo": 1,
    }

    url = f"{settings.mfds_base_url}/{_ENDPOINT}"

    if client is None:
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECONDS) as owned:
            response = await _get_with_retry(owned, url, params)
    else:
        response = await _get_with_retry(client, url, params)

    payload = response.json()
    items = _extract_items(payload)
    if not items:
        raise MfdsFetchError(f"MFDS 에 '{query}' 첨부문서가 없음")

    return _to_label(items[0])


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
            raise MfdsFetchError(f"MFDS 호출 실패: {exc}") from exc

        status = response.status_code
        if status == 429 or 500 <= status < 600:
            if attempt < _MAX_RETRIES:
                await asyncio.sleep(_BACKOFF_BASE_SECONDS * (2**attempt))
                continue
            raise MfdsFetchError(f"MFDS 재시도 초과: status={status}")

        response.raise_for_status()
        return response

    raise MfdsFetchError("MFDS 페처가 예상치 못한 상태로 종료됨")


def _extract_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    # 공공데이터포털 응답이 {response: {body: ...}} 또는 {body: ...} 형태로 옴
    body = payload.get("body")
    if body is None:
        body = (payload.get("response") or {}).get("body") or {}

    raw_items = body.get("items") or []
    if isinstance(raw_items, dict):
        # items 가 단일 객체로 올 때 (단건 응답 일부 케이스)
        raw_items = [raw_items]
    return [item for item in raw_items if isinstance(item, dict)]


def _to_label(item: dict[str, Any]) -> MfdsLabel:
    drug_name = item.get("itemName") or ""
    if not drug_name:
        raise MfdsFetchError("응답 itemName 없음")

    sections: dict[str, list[str]] = {}
    for field, korean_key in _SECTION_FIELDS.items():
        value = item.get(field)
        if isinstance(value, str) and value.strip():
            sections[korean_key] = [value]

    return MfdsLabel(
        drug_name=drug_name,
        sections=sections,
        item_seq=str(item.get("itemSeq") or ""),
        permit_date=str(item.get("permitDate") or item.get("openDe") or ""),
    )
