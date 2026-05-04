"""FastAPI 엔트리 포인트

- API 라우터 마운트
- lifespan 에서 RetrievalService 를 startup 시 로드해 app.state 에 보관
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from mediforme_chatbot_rag import __version__
from mediforme_chatbot_rag.api import health, retrieve
from mediforme_chatbot_rag.services.retrieval import load_retrieval_service


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.retrieval = load_retrieval_service()
    yield


app = FastAPI(
    title="Mediforme Chatbot RAG",
    description="FDA·MFDS 라벨 기반 retrieval 서비스",
    version=__version__,
    lifespan=_lifespan,
)

app.include_router(health.router)
app.include_router(retrieve.router)
