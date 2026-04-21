# mediforme-chatbot-rag

> mediforme Chatbot RAG Service — FastAPI + pgvector

Mediforme 챗봇의 **Retrieval-Augmented Generation** 서비스입니다. 
FDA openFDA 및 MFDS 의약품 라벨을 임베딩하고, Java 챗봇(backend_medi)이 질문 시 근거 청크를 retrieve 해 응답에 주입하도록 돕습니다.

## 설계 배경

전체 설계 문서 및 의사결정은 백엔드 위키에 있습니다.

- [5.6 Chatbot RAG 도입 설계 (Phase B)](https://github.com/medi4me/backend_medi/wiki/5.6-기술-의사결정-(Decisions)-‐-Chatbot-RAG-도입-설계)
- [측정 기반 근거 — A vs B 베이스라인 리포트 (PR #59)](https://github.com/medi4me/backend_medi/pull/59)

## 관련 레포

- [medi4me/backend_medi](https://github.com/medi4me/backend_medi) — Java Spring Boot 백엔드 (이 서비스를 HTTP 호출)
- [medi4me/frontend_medi](https://github.com/medi4me/frontend_medi) — Android 클라이언트
- [medi4me/ai_pill_classifier](https://github.com/medi4me/ai_pill_classifier) — 약 이미지 분류 AI

## 기술 스택

| 영역 | 선택 |
|---|---|
| 언어·런타임 | Python 3.12 |
| 웹 프레임워크 | FastAPI |
| 벡터 DB | Postgres + pgvector |
| 임베딩 | `text-embedding-3-large` (1순위) / BGE-M3 (전환 검토) |
| 패키지 매니저 | uv |
| Lint/Format | ruff |
| 타입 체크 | mypy |
| 테스트 | pytest |
| 컨테이너 | Docker (multi-stage) + docker-compose |

## API 계약

```
POST /retrieve
{
  "drug_id": "tylenol",              // optional
  "category": "contraindications",   // optional
  "query": "타이레놀 먹으면 안 되는 사람",
  "top_k": 5
}

→ 200 OK
{
  "chunks": [
    {
      "text": "Do not use if you have severe hepatic impairment...",
      "source": "fda_label",
      "drug_name": "Tylenol",
      "section": "contraindications",
      "similarity": 0.87
    }
  ]
}
```

## 로컬 개발

### 요구사항

- Python 3.12+
- Docker (pgvector 실행용)
- [uv](https://docs.astral.sh/uv/) (권장, 선택 사항 — `pip` 로도 동작)

### 설정

```bash
# 1) 가상환경·의존성 설치
uv sync

# 2) 환경변수 설정
cp .env.example .env
# .env 에 OPENAI_API_KEY, DATABASE_URL 등 채우기

# 3) pgvector 기동
docker compose up -d db

# 4) 서비스 기동
uv run uvicorn mediforme_chatbot_rag.main:app --reload
```

### 테스트

```bash
uv run pytest
uv run ruff check .
uv run mypy src
```

## 프로젝트 구조

```
src/mediforme_chatbot_rag/
├── main.py            # FastAPI 엔트리 (APIRouter 마운트)
├── core/
│   └── config.py      # 환경변수 설정
└── api/
    ├── health.py      # GET /healthz
    └── retrieve.py    # POST /retrieve (Phase C-3 구현)

tests/                 # pytest
scripts/               # 라벨 ingestion 파이프라인 (Phase C-2)
```

## 현재 상태

- [x] C-0 설계 확정 (위키 5.6)
- [x] C-1 레포·인프라 스켈레톤 (본 커밋)
- [ ] C-2 Ingestion — FDA openFDA 페처 + 섹션 청크 + 임베딩 + 저장
- [ ] C-3 Retrieval — `/retrieve` 엔드포인트
- [ ] C-4 평가 연동 — backend_medi `eval/chatbot-rag/runners/version_c_rag.py`
- [ ] C-5 측정 + A/B/C 3-way 리포트
- [ ] C-6 Java 통합 — `ChatbotRagClient` 포트 + `HttpRagClient` 어댑터

## 라이선스

MIT — [LICENSE](./LICENSE) 참고.
