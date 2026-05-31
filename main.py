import os
import io
import uuid
import json
import logging
import threading
from pathlib import Path
from typing import Optional

import pdfplumber
from pypdf import PdfReader
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
)
import numpy as np

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
COLLECTION_NAME = "resumes"
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VECTOR_DIM = 384

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Resume Search & Ranking API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy singletons ──────────────────────────────────────────────────────────
_embedder = None
_embedder_ready = False
_embedder_lock = threading.Lock()
_qdrant: Optional[QdrantClient] = None


def _load_embedder_bg():
    """Load embedding model in background thread."""
    global _embedder, _embedder_ready
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model in background …")
        model = SentenceTransformer(EMBED_MODEL_NAME)
        with _embedder_lock:
            _embedder = model
            _embedder_ready = True
        logger.info("✅ Embedding model loaded.")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")


def get_embedder():
    global _embedder, _embedder_ready
    if not _embedder_ready:
        raise HTTPException(
            status_code=503,
            detail="모델 로딩 중입니다. 잠시 후 다시 시도해주세요. (Embedding model is loading, please retry in a moment.)"
        )
    return _embedder


def get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        kwargs = {"host": QDRANT_HOST, "port": QDRANT_PORT}
        if QDRANT_API_KEY:
            kwargs["api_key"] = QDRANT_API_KEY
        _qdrant = QdrantClient(**kwargs)
        logger.info(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        _ensure_collection(_qdrant)
    return _qdrant


def _ensure_collection(client: QdrantClient):
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        logger.info(f"Created collection '{COLLECTION_NAME}'")


# ── PDF helpers ──────────────────────────────────────────────────────────────
def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Try pdfplumber first, fall back to pypdf."""
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
    except Exception:
        pass

    if not text.strip():
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
        except Exception:
            pass

    return text.strip()


def chunk_text(text: str, max_chars: int = 2000) -> str:
    """Return first max_chars chars as a single representative chunk."""
    return text[:max_chars]


# ── Pydantic models ──────────────────────────────────────────────────────────
class SearchRequest_(BaseModel):
    query: str
    top_k: int = 10


class RankRequest(BaseModel):
    job_description: str
    top_k: int = 20


class DeleteRequest(BaseModel):
    resume_id: str


# ── Startup ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Start background model loading and ensure collection exists."""
    # Load model in background so health check passes immediately
    threading.Thread(target=_load_embedder_bg, daemon=True).start()
    try:
        get_qdrant()
    except Exception as e:
        logger.warning(f"Qdrant init warning (will retry on first request): {e}")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = Path(__file__).parent / "static" / "index.html"
    return html_path.read_text(encoding="utf-8")


@app.get("/health")
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model_ready": _embedder_ready,
        "qdrant_host": QDRANT_HOST,
    }


@app.post("/api/upload")
async def upload_resumes(files: list[UploadFile] = File(...)):
    """Upload up to 100 PDF resumes and store embeddings in Qdrant."""
    if len(files) > 100:
        raise HTTPException(status_code=400, detail="최대 100개의 PDF를 업로드할 수 있습니다.")

    embedder = get_embedder()
    client = get_qdrant()

    results = []
    points = []

    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            results.append({"filename": f.filename, "status": "skipped", "reason": "PDF 파일이 아닙니다."})
            continue

        try:
            file_bytes = await f.read()
            text = extract_text_from_pdf(file_bytes)

            if not text:
                results.append({"filename": f.filename, "status": "error", "reason": "텍스트 추출 실패"})
                continue

            chunk = chunk_text(text)
            vector = embedder.encode(chunk, normalize_embeddings=True).tolist()

            resume_id = str(uuid.uuid4())
            payload = {
                "resume_id": resume_id,
                "filename": f.filename,
                "text": text[:5000],          # store up to 5 000 chars
                "text_preview": text[:300],
                "char_count": len(text),
            }

            points.append(PointStruct(id=resume_id, vector=vector, payload=payload))
            results.append({"filename": f.filename, "status": "success", "resume_id": resume_id})

        except Exception as e:
            logger.exception(f"Error processing {f.filename}")
            results.append({"filename": f.filename, "status": "error", "reason": str(e)})

    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        logger.info(f"Upserted {len(points)} resumes.")

    return {"uploaded": len(points), "results": results}


@app.get("/api/resumes")
async def list_resumes(limit: int = 100, offset: int = 0):
    """List all stored resumes."""
    client = get_qdrant()
    records, next_offset = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=limit,
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )
    return {
        "total": client.count(COLLECTION_NAME).count,
        "resumes": [
            {
                "resume_id": r.payload.get("resume_id"),
                "filename": r.payload.get("filename"),
                "text_preview": r.payload.get("text_preview", ""),
                "char_count": r.payload.get("char_count", 0),
            }
            for r in records
        ],
    }


@app.delete("/api/resumes/{resume_id}")
async def delete_resume(resume_id: str):
    """Delete a resume by ID."""
    client = get_qdrant()
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=[resume_id],
    )
    return {"deleted": resume_id}


@app.delete("/api/resumes")
async def delete_all_resumes():
    """Delete all resumes (reset collection)."""
    client = get_qdrant()
    client.delete_collection(COLLECTION_NAME)
    _ensure_collection(client)
    return {"message": "모든 이력서가 삭제되었습니다."}


@app.post("/api/search")
async def search_resumes(req: SearchRequest_):
    """Natural-language search over stored resumes."""
    embedder = get_embedder()
    client = get_qdrant()

    query_vec = embedder.encode(req.query, normalize_embeddings=True).tolist()
    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vec,
        limit=req.top_k,
        with_payload=True,
    )

    return {
        "query": req.query,
        "results": [
            {
                "rank": i + 1,
                "score": round(float(hit.score), 4),
                "resume_id": hit.payload.get("resume_id"),
                "filename": hit.payload.get("filename"),
                "text_preview": hit.payload.get("text_preview", ""),
            }
            for i, hit in enumerate(hits)
        ],
    }


@app.post("/api/rank")
async def rank_for_job(req: RankRequest):
    """Rank all resumes against a job description."""
    embedder = get_embedder()
    client = get_qdrant()

    total = client.count(COLLECTION_NAME).count
    if total == 0:
        raise HTTPException(status_code=404, detail="저장된 이력서가 없습니다.")

    job_vec = embedder.encode(req.job_description, normalize_embeddings=True).tolist()

    # Fetch all vectors (up to 1000) and score them
    limit = min(total, 1000)
    records, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=limit,
        with_payload=True,
        with_vectors=True,
    )

    scored = []
    job_arr = np.array(job_vec)
    for r in records:
        rv = np.array(r.vector)
        score = float(np.dot(job_arr, rv))  # cosine similarity (vectors normalized)
        scored.append(
            {
                "score": round(score, 4),
                "resume_id": r.payload.get("resume_id"),
                "filename": r.payload.get("filename"),
                "text_preview": r.payload.get("text_preview", ""),
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[: req.top_k]
    for i, item in enumerate(top):
        item["rank"] = i + 1

    return {
        "job_description": req.job_description[:300],
        "total_candidates": total,
        "results": top,
    }


@app.get("/api/stats")
async def stats():
    client = get_qdrant()
    count = client.count(COLLECTION_NAME).count
    return {"total_resumes": count, "collection": COLLECTION_NAME}
