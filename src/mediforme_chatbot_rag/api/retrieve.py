"""/retrieve — 약 ID / 카테고리 / 쿼리 기반 벡터 검색 엔드포인트 (Phase C-3 구현 예정)."""

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()


class RetrieveRequest(BaseModel):
    drug_id: str | None = Field(default=None, description="확정된 약 식별자. 있으면 메타데이터 필터에 사용.")
    category: str | None = Field(default=None, description="질문 카테고리. 라벨 섹션 매핑에 사용.")
    query: str = Field(description="사용자 자연어 질의.")
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievedChunk(BaseModel):
    text: str
    source: str
    drug_name: str
    section: str
    similarity: float


class RetrieveResponse(BaseModel):
    chunks: list[RetrievedChunk]


@router.post("/retrieve", response_model=RetrieveResponse, tags=["retrieval"])
def retrieve(_req: RetrieveRequest) -> RetrieveResponse:
    """TODO(C-3): pgvector 쿼리 + 메타데이터 필터 + top-k 반환."""
    return RetrieveResponse(chunks=[])