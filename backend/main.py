"""
main.py — FastAPI application for the Local OCR + RAG System

Endpoints:
  GET  /              → Serves the frontend HTML UI
  POST /upload        → Upload + OCR + embed + store document
  POST /search        → RAG search with metadata filters
  GET  /documents     → List all indexed documents
  DELETE /documents/{doc_id}  → Remove a document
  GET  /stats         → Collection statistics
  GET  /health        → Health check
"""

import os
import sys
import uuid
import logging
import shutil
from datetime import date
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add backend directory to path
sys.path.insert(0, str(Path(__file__).parent))

import ocr_processor
import chunker
import embedder
import vector_store as vs
import rag_engine

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
UPLOADS_DIR = Path("./uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title       = "Local OCR + RAG System",
    description = "Assessment 3 — Bilingual (Bangla + English) document processing pipeline",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ---------------------------------------------------------------------------
# Language detection helper
# ---------------------------------------------------------------------------
def detect_language(text: str) -> str:
    """Detect whether text is Bangla, English, or mixed."""
    if not text.strip():
        return "unknown"
    try:
        from langdetect import detect
        code = detect(text)
        if code == "bn":
            return "bn"
        elif code == "en":
            return "en"
        else:
            # Check for Bangla Unicode characters (U+0980–U+09FF)
            bangla_chars = sum(1 for c in text if '\u0980' <= c <= '\u09FF')
            ratio = bangla_chars / max(len(text), 1)
            if ratio > 0.3:
                return "bn"
            return code
    except Exception:
        bangla_chars = sum(1 for c in text if '\u0980' <= c <= '\u09FF')
        return "bn" if bangla_chars / max(len(text), 1) > 0.1 else "en"


def determine_doc_language(page_texts: list) -> str:
    """Determine document-level language from all pages."""
    all_text = " ".join(page_texts)
    if not all_text.strip():
        return "unknown"

    bangla_chars = sum(1 for c in all_text if '\u0980' <= c <= '\u09FF')
    total_chars  = len([c for c in all_text if c.strip()])

    if total_chars == 0:
        return "unknown"

    ratio = bangla_chars / total_chars
    if ratio > 0.5:
        return "bn"
    elif ratio > 0.1:
        return "mixed"
    else:
        return "en"


# ---------------------------------------------------------------------------
# Serve Frontend
# ---------------------------------------------------------------------------
FRONTEND_PATH = Path(__file__).parent.parent / "frontend" / "index.html"


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    if FRONTEND_PATH.exists():
        return FRONTEND_PATH.read_text(encoding="utf-8")
    return HTMLResponse("<h1>Frontend not found. Place index.html in frontend/</h1>")


# ---------------------------------------------------------------------------
# Health & Stats
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "message": "OCR RAG system is running"}


@app.get("/stats")
async def stats():
    try:
        return vs.get_stats()
    except Exception as e:
        raise HTTPException(500, str(e))


# ---------------------------------------------------------------------------
# Upload Endpoint
# ---------------------------------------------------------------------------
@app.post("/upload")
async def upload_document(
    file      : UploadFile = File(...),
    language  : str        = Form("auto"),   # "auto", "bn", "en", "mixed"
    doc_date  : str        = Form(""),       # override upload date (ISO format)
):
    """
    Upload a PDF or image → run local OCR → chunk → embed → store in Qdrant.

    Processing log is returned in the response so the demo video can show
    exactly what happened locally.
    """
    doc_id    = str(uuid.uuid4())
    filename  = file.filename
    suffix    = Path(filename).suffix.lower()
    log       = []

    # ── Validate file type ────────────────────────────────────────────────────
    allowed = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type: {suffix}. Allowed: {allowed}")

    # ── Save uploaded file ────────────────────────────────────────────────────
    save_path = UPLOADS_DIR / f"{doc_id}{suffix}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    file_size = save_path.stat().st_size
    log.append(f"✅ Saved: {filename} ({file_size:,} bytes) → uploads/{doc_id}{suffix}")
    logger.info(log[-1])

    # ── OCR Extraction ────────────────────────────────────────────────────────
    log.append(f"🔍 Starting OCR extraction (local, no external API)...")
    logger.info(log[-1])

    try:
        page_texts, engine_used = ocr_processor.extract_text_from_file(str(save_path))
    except Exception as e:
        raise HTTPException(500, f"OCR failed: {e}")

    total_chars = sum(len(t) for t in page_texts)
    log.append(
        f"✅ OCR complete: {len(page_texts)} page(s), {total_chars:,} chars extracted "
        f"using [{engine_used}]"
    )
    logger.info(log[-1])

    if total_chars < 10:
        raise HTTPException(422, "OCR extracted no readable text from the document.")

    # ── Language detection ────────────────────────────────────────────────────
    detected_language = determine_doc_language(page_texts)
    final_language    = language if language != "auto" else detected_language
    log.append(f"🌐 Language detection: '{detected_language}' → using '{final_language}'")
    logger.info(log[-1])

    # ── Chunking ──────────────────────────────────────────────────────────────
    log.append("✂️  Chunking text (sentence-aware, Bangla + English)...")
    logger.info(log[-1])

    all_chunks    = []
    chunk_pages   = []

    for page_num, page_text in enumerate(page_texts, start=1):
        page_chunks = chunker.chunk_text(page_text, chunk_size=400, overlap=80)
        all_chunks.extend(page_chunks)
        chunk_pages.extend([page_num] * len(page_chunks))

    log.append(f"✅ Created {len(all_chunks)} chunks across {len(page_texts)} page(s)")
    logger.info(log[-1])

    # ── Embedding ─────────────────────────────────────────────────────────────
    log.append(f"🧠 Embedding {len(all_chunks)} chunks with multilingual-MiniLM...")
    logger.info(log[-1])

    try:
        embeddings = embedder.embed(all_chunks)
    except Exception as e:
        raise HTTPException(500, f"Embedding failed: {e}")

    log.append(f"✅ Embeddings generated: shape {embeddings.shape}")
    logger.info(log[-1])

    # ── Store in Qdrant ───────────────────────────────────────────────────────
    log.append(f"💾 Storing in Qdrant (local vector store)...")
    logger.info(log[-1])

    upload_date = doc_date if doc_date else date.today().isoformat()
    doc_type    = "pdf" if suffix == ".pdf" else "image"

    try:
        n_stored = vs.add_document(
            doc_id      = doc_id,
            filename    = filename,
            doc_type    = doc_type,
            language    = final_language,
            chunks      = all_chunks,
            embeddings  = embeddings,
            page_numbers= chunk_pages,
            upload_date = upload_date,
        )
    except Exception as e:
        raise HTTPException(500, f"Storage failed: {e}")

    log.append(f"✅ Stored {n_stored} vectors in Qdrant collection 'document_chunks'")
    logger.info(log[-1])
    log.append(f"🎉 Done! Document ready for RAG search. doc_id: {doc_id}")

    return {
        "success"      : True,
        "doc_id"       : doc_id,
        "filename"     : filename,
        "pages"        : len(page_texts),
        "chunks"       : len(all_chunks),
        "language"     : final_language,
        "doc_type"     : doc_type,
        "ocr_engine"   : engine_used,
        "upload_date"  : upload_date,
        "processing_log": log,
    }


# ---------------------------------------------------------------------------
# Search Endpoint
# ---------------------------------------------------------------------------
class SearchRequest(BaseModel):
    query     : str
    language  : Optional[str] = None    # "bn", "en", "mixed", "all"
    doc_type  : Optional[str] = None    # "pdf", "image", "all"
    date_from : Optional[str] = None    # ISO date string "2026-01-01"
    date_to   : Optional[str] = None
    doc_id    : Optional[str] = None
    top_k     : int           = 5


@app.post("/search")
async def search(req: SearchRequest):
    """
    Hybrid search: metadata filter + vector similarity + LLM answer.

    Metadata filters (language, doc_type, date range) are applied at the
    Qdrant level — not as a post-filter — so ranking is within the filtered
    subset only.
    """
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty.")

    try:
        result = rag_engine.search_and_answer(
            query     = req.query,
            language  = req.language  if req.language  not in (None, "all", "") else None,
            doc_type  = req.doc_type  if req.doc_type  not in (None, "all", "") else None,
            date_from = req.date_from if req.date_from else None,
            date_to   = req.date_to   if req.date_to   else None,
            doc_id    = req.doc_id    if req.doc_id    else None,
            top_k     = req.top_k,
        )
        return result
    except Exception as e:
        logger.exception(f"Search error: {e}")
        raise HTTPException(500, str(e))


# ---------------------------------------------------------------------------
# Document Management
# ---------------------------------------------------------------------------
@app.get("/documents")
async def list_documents():
    """List all indexed documents."""
    try:
        docs = vs.list_documents()
        return {"documents": docs, "count": len(docs)}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document and all its chunks from the vector store."""
    try:
        n = vs.delete_document(doc_id)
        # Also delete the file from uploads
        for f in UPLOADS_DIR.glob(f"{doc_id}*"):
            f.unlink()
        return {"success": True, "chunks_deleted": n, "doc_id": doc_id}
    except Exception as e:
        raise HTTPException(500, str(e))


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Local OCR + RAG System...")
    logger.info("Frontend: http://localhost:8000")
    logger.info("API docs: http://localhost:8000/docs")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
