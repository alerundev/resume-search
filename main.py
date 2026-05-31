import os
import io
import uuid
import logging
import threading
from pathlib import Path
from typing import Optional, List

# Ensure writable cache dirs before any HF imports
os.environ.setdefault("HF_HOME", "/app/.cache/huggingface")
os.environ.setdefault("FASTEMBED_CACHE_PATH", "/app/.cache/fastembed")

import pdfplumber
from pypdf import PdfReader
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Embedding provider detection ─────────────────────────────────────────────
# Support both OPENAI_APIKEY (Cloudtype secret name) and OPENAI_API_KEY
OPENAI_API_KEY = os.getenv("OPENAI_APIKEY") or os.getenv("OPENAI_API_KEY") or ""

if OPENAI_API_KEY:
    EMBED_PROVIDER = "openai"
    EMBED_MODEL   = "text-embedding-3-small"
    VECTOR_DIM    = 1536
    logger.info("🔑 OpenAI embedding mode: text-embedding-3-small (1536 dims)")
else:
    EMBED_PROVIDER = "fastembed"
    EMBED_MODEL   = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    VECTOR_DIM    = 384
    logger.info("🤖 fastembed mode: paraphrase-multilingual-MiniLM-L12-v2 (384 dims)")

# ── Config ───────────────────────────────────────────────────────────────────
QDRANT_HOST      = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT      = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY   = os.getenv("QDRANT_API_KEY", None)
COLLECTION_NAME  = "resumes"

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Resume Search & Ranking API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singletons ───────────────────────────────────────────────────────────────
_fastembed_model  = None
_embedder_ready   = False
_embedder_lock    = threading.Lock()
_qdrant: Optional[QdrantClient] = None


# ── OpenAI embedding ─────────────────────────────────────────────────────────
def embed_openai(text: str) -> List[float]:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=text[:8000],
        encoding_format="float",
    )
    v = np.array(resp.data[0].embedding, dtype=np.float32)
    return v.tolist()


# ── fastembed (background load) ──────────────────────────────────────────────
def _load_fastembed_bg():
    global _fastembed_model, _embedder_ready
    try:
        from fastembed import TextEmbedding
        logger.info("Loading fastembed model …")
        model = TextEmbedding(model_name=EMBED_MODEL)
        _ = list(model.embed(["warm up"]))
        with _embedder_lock:
            _fastembed_model = model
            _embedder_ready = True
        logger.info("✅ fastembed model ready.")
    except Exception as e:
        logger.error(f"Failed to load fastembed model: {e}", exc_info=True)


# ── Unified embed ─────────────────────────────────────────────────────────────
def embed_text(text: str) -> List[float]:
    if EMBED_PROVIDER == "openai":
        return embed_openai(text)
    else:
        if not _embedder_ready:
            raise HTTPException(
                status_code=503,
                detail="모델 로딩 중입니다. 잠시 후 다시 시도해주세요.",
            )
        vectors = list(_fastembed_model.embed([text]))
        v = np.array(vectors[0], dtype=np.float32)
        v = v / (np.linalg.norm(v) + 1e-10)
        return v.tolist()


# ── Qdrant ────────────────────────────────────────────────────────────────────
def get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        kwargs: dict = {"host": QDRANT_HOST, "port": QDRANT_PORT}
        if QDRANT_API_KEY:
            kwargs["api_key"] = QDRANT_API_KEY
        _qdrant = QdrantClient(**kwargs)
        logger.info(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        _ensure_collection(_qdrant)
    return _qdrant


def _ensure_collection(client: QdrantClient):
    """Create or recreate collection if vector dim doesn't match."""
    existing = client.get_collections().collections
    existing_names = [c.name for c in existing]

    if COLLECTION_NAME in existing_names:
        info = client.get_collection(COLLECTION_NAME)
        current_dim = info.config.params.vectors.size
        if current_dim != VECTOR_DIM:
            logger.warning(
                f"Collection dim mismatch ({current_dim} vs {VECTOR_DIM}). "
                "Recreating collection — existing data will be lost."
            )
            client.delete_collection(COLLECTION_NAME)
            existing_names.remove(COLLECTION_NAME)

    if COLLECTION_NAME not in existing_names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        logger.info(f"Created collection '{COLLECTION_NAME}' (dim={VECTOR_DIM})")


# ── PDF helpers ───────────────────────────────────────────────────────────────
def extract_text_from_pdf(file_bytes: bytes) -> str:
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


# ── Pydantic models ───────────────────────────────────────────────────────────
class SearchReq(BaseModel):
    query: str
    top_k: int = 10


class RankReq(BaseModel):
    job_description: str
    top_k: int = 20


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    if EMBED_PROVIDER == "fastembed":
        threading.Thread(target=_load_fastembed_bg, daemon=True).start()
    else:
        # OpenAI: mark ready immediately (no local model to load)
        global _embedder_ready
        _embedder_ready = True
        logger.info("✅ OpenAI embedding ready (API calls on demand).")
    try:
        get_qdrant()
    except Exception as e:
        logger.warning(f"Qdrant init warning (will retry): {e}")


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
        "embed_provider": EMBED_PROVIDER,
        "embed_model": EMBED_MODEL,
        "vector_dim": VECTOR_DIM,
        "qdrant_host": QDRANT_HOST,
    }


@app.post("/api/upload")
async def upload_resumes(files: List[UploadFile] = File(...)):
    if len(files) > 100:
        raise HTTPException(status_code=400, detail="최대 100개의 PDF를 업로드할 수 있습니다.")
    if not _embedder_ready:
        raise HTTPException(status_code=503, detail="모델 로딩 중입니다. 잠시 후 다시 시도해주세요.")

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
                results.append({"filename": f.filename, "status": "error", "reason": "텍스트 추출 실패 (스캔 PDF는 지원 안 됩니다)"})
                continue

            vector = embed_text(text[:8000])
            resume_id = str(uuid.uuid4())
            payload = {
                "resume_id": resume_id,
                "filename": f.filename,
                "text": text[:5000],
                "text_preview": text[:300],
                "char_count": len(text),
            }
            points.append(PointStruct(id=resume_id, vector=vector, payload=payload))
            results.append({"filename": f.filename, "status": "success", "resume_id": resume_id})
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Error processing {f.filename}")
            results.append({"filename": f.filename, "status": "error", "reason": str(e)})

    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        logger.info(f"Upserted {len(points)} resumes.")

    return {"uploaded": len(points), "results": results}


@app.get("/api/resumes")
async def list_resumes(limit: int = 100, offset: int = 0):
    client = get_qdrant()
    records, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=limit, offset=offset,
        with_payload=True, with_vectors=False,
    )
    return {
        "total": client.count(COLLECTION_NAME).count,
        "resumes": [
            {
                "resume_id": r.payload.get("resume_id"),
                "filename":  r.payload.get("filename"),
                "text_preview": r.payload.get("text_preview", ""),
                "char_count": r.payload.get("char_count", 0),
            }
            for r in records
        ],
    }


@app.delete("/api/resumes/{resume_id}")
async def delete_resume(resume_id: str):
    get_qdrant().delete(collection_name=COLLECTION_NAME, points_selector=[resume_id])
    return {"deleted": resume_id}


@app.delete("/api/resumes")
async def delete_all_resumes():
    client = get_qdrant()
    client.delete_collection(COLLECTION_NAME)
    _ensure_collection(client)
    return {"message": "모든 이력서가 삭제되었습니다."}


@app.post("/api/search")
async def search_resumes(req: SearchReq):
    client = get_qdrant()
    query_vec = embed_text(req.query)
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
                "filename":  hit.payload.get("filename"),
                "text_preview": hit.payload.get("text_preview", ""),
            }
            for i, hit in enumerate(hits)
        ],
    }


@app.post("/api/rank")
async def rank_for_job(req: RankReq):
    client = get_qdrant()
    total = client.count(COLLECTION_NAME).count
    if total == 0:
        raise HTTPException(status_code=404, detail="저장된 이력서가 없습니다.")

    job_vec  = embed_text(req.job_description)
    job_arr  = np.array(job_vec)
    records, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=min(total, 1000),
        with_payload=True, with_vectors=True,
    )

    scored = []
    for r in records:
        rv    = np.array(r.vector)
        score = float(np.dot(job_arr, rv))
        scored.append({
            "score": round(score, 4),
            "resume_id": r.payload.get("resume_id"),
            "filename":  r.payload.get("filename"),
            "text_preview": r.payload.get("text_preview", ""),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[: req.top_k]
    for i, item in enumerate(top):
        item["rank"] = i + 1

    return {
        "job_description":  req.job_description[:300],
        "total_candidates": total,
        "embed_provider":   EMBED_PROVIDER,
        "results": top,
    }


@app.get("/api/stats")
async def stats():
    count = get_qdrant().count(COLLECTION_NAME).count
    return {
        "total_resumes":  count,
        "collection":     COLLECTION_NAME,
        "embed_provider": EMBED_PROVIDER,
        "embed_model":    EMBED_MODEL,
        "vector_dim":     VECTOR_DIM,
    }
