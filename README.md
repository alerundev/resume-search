# 📄 AI 이력서 검색 시스템 (Resume Search & Ranking)

PDF 이력서를 벡터 DB에 저장하고, 자연어로 원하는 인재를 검색하거나 채용공고에 맞춰 순위를 매기는 서비스입니다.

## ✨ 주요 기능

| 기능 | 설명 |
|------|------|
| **PDF 업로드** | 최대 100개의 PDF 이력서를 드래그&드롭으로 업로드 |
| **자연어 검색** | "Python 5년 경력 백엔드 개발자" 같은 자연어로 유사 이력서 검색 |
| **채용공고 매칭** | 채용공고 전문을 입력하면 이력서를 유사도 순으로 자동 순위 결정 |
| **이력서 관리** | 업로드된 이력서 목록 조회 및 개별/전체 삭제 |

## 🛠 기술 스택

- **Backend**: FastAPI (Python)
- **Vector DB**: Qdrant
- **Embedding Model**: `paraphrase-multilingual-MiniLM-L12-v2` (한국어/영어 지원)
- **PDF 파싱**: pdfplumber + pypdf
- **Frontend**: Vanilla HTML/CSS/JS (서버 내장 제공)

## 🚀 로컬 실행

```bash
# 1. Qdrant 실행 (Docker)
docker run -p 6333:6333 qdrant/qdrant

# 2. 패키지 설치
pip install -r requirements.txt

# 3. 서버 실행
uvicorn main:app --host 0.0.0.0 --port 8000

# 4. 브라우저에서 http://localhost:8000 접속
```

## 🌍 환경변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `QDRANT_HOST` | `localhost` | Qdrant 서버 호스트 |
| `QDRANT_PORT` | `6333` | Qdrant 서버 포트 |
| `QDRANT_API_KEY` | (없음) | Qdrant API 키 (클라우드 환경) |

## 📡 API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| `POST` | `/api/upload` | PDF 이력서 업로드 |
| `GET`  | `/api/resumes` | 이력서 목록 조회 |
| `DELETE` | `/api/resumes/{id}` | 이력서 삭제 |
| `DELETE` | `/api/resumes` | 전체 삭제 |
| `POST` | `/api/search` | 자연어 검색 |
| `POST` | `/api/rank` | 채용공고 기반 순위 |
| `GET`  | `/api/stats` | 통계 |

## 📦 배포 (Cloudtype)

이 서비스는 Cloudtype에 Python + Qdrant 구성으로 배포됩니다.

환경변수 `QDRANT_HOST`를 Cloudtype 내부 hostname(Qdrant 서비스 이름)으로 설정하세요.
