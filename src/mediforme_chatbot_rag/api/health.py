"""Health check endpoints."""

from fastapi import APIRouter

from mediforme_chatbot_rag import __version__

router = APIRouter()


@router.get("/healthz", tags=["meta"])
def healthz() -> dict[str, str]:
    """서비스 기본 기동 확인 (liveness)."""
    return {"status": "ok", "version": __version__}


@router.get("/readyz", tags=["meta"])
def readyz() -> dict[str, str]:
    """외부 의존성(DB·임베딩) 준비 상태 확인 (readiness).

    TODO(C-2): DB 연결, 임베딩 모델 준비 상태 실제 체크.
    """
    return {"status": "ok"}
