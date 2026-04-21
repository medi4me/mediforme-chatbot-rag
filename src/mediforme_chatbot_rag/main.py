"""FastAPI 엔트리 포인트."""

from fastapi import FastAPI

from mediforme_chatbot_rag import __version__
from mediforme_chatbot_rag.api import health, retrieve

app = FastAPI(
    title="Mediforme Chatbot RAG",
    description="FDA·MFDS 라벨 기반 retrieval 서비스",
    version=__version__,
)

app.include_router(health.router)
app.include_router(retrieve.router)