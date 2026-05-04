"""/retrieve — 약 ID / 카테고리 / 쿼리 기반 벡터 검색 엔드포인트

- 사용자 자연어 질의를 임베딩해 FAISS 인덱스에서 top-k 청크 검색
- drug_id / category 로 결과 필터링
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from mediforme_chatbot_rag.services.retrieval import RetrievalService

router = APIRouter()


class RetrieveRequest(BaseModel):
    drug_id: str | None = Field(
        default=None, description="확정된 약 식별자. 있으면 메타데이터 필터에 사용."
    )
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


def get_retrieval_service(request: Request) -> RetrievalService:
    service: RetrievalService = request.app.state.retrieval
    return service


@router.post("/retrieve", response_model=RetrieveResponse, tags=["retrieval"])
def retrieve(
    req: RetrieveRequest,
    service: Annotated[RetrievalService, Depends(get_retrieval_service)],
) -> RetrieveResponse:
    results = service.retrieve(
        req.query,
        top_k=req.top_k,
        drug_id=req.drug_id,
        category=req.category,
    )
    return RetrieveResponse(
        chunks=[
            RetrievedChunk(
                text=chunk.text,
                source=chunk.source,
                drug_name=chunk.drug_name,
                section=chunk.section,
                similarity=score,
            )
            for chunk, score in results
        ]
    )
