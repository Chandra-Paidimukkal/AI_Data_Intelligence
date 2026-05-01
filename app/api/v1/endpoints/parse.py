"""
parse.py — ADE-style parse endpoint.

POST /api/v1/parse
  Upload a file and receive the full ADE-compatible parse result immediately
  (synchronous, no job queue).

GET /api/v1/parse/{doc_id}/chunks
  Return the chunk list for an already-parsed document.

GET /api/v1/parse/{doc_id}/splits
  Return page splits.

GET /api/v1/parse/{doc_id}/grounding
  Return the grounding map.

GET /api/v1/parse/{doc_id}/markdown
  Return the full markdown string.
"""
import os
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.models.document import Document
from app.services.parser import parse_document

router = APIRouter(prefix="/parse", tags=["ADE Parse"])


def _save_upload(upload: UploadFile) -> tuple[str, str, int]:
    doc_id = str(uuid.uuid4())
    ext = Path(upload.filename or "file.bin").suffix
    fname = f"{doc_id}{ext}"
    fpath = os.path.join(settings.UPLOAD_DIR, fname)
    with open(fpath, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    size = os.path.getsize(fpath)
    return doc_id, fpath, size


@router.post("")
async def parse_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload and parse a document immediately.
    Returns full ADE-compatible response:
      markdown, chunks, splits, grounding, metadata.
    """
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    doc_id, fpath, size = _save_upload(file)

    # Persist document record
    doc = Document(
        id=doc_id,
        file_name=file.filename,
        file_path=fpath,
        file_size=size,
        mime_type=file.content_type or "application/octet-stream",
        status="parsing",
    )
    db.add(doc)
    db.commit()

    try:
        parsed = parse_document(fpath, file.content_type or "")
        doc.parsed_data = parsed
        doc.page_count = parsed["metadata"].get("page_count", 1)
        doc.status = "parsed"
        db.commit()
    except Exception as e:
        doc.status = "error"
        doc.error_message = str(e)
        db.commit()
        raise HTTPException(500, f"Parse failed: {e}")

    # Return ADE-style top-level response
    return {
        "id": doc_id,
        "markdown": parsed["markdown"],
        "chunks": parsed["chunks"],
        "splits": parsed["splits"],
        "grounding": parsed["grounding"],
        "metadata": {
            **parsed["metadata"],
            "job_id": doc_id,
        }
    }


@router.get("/{doc_id}/chunks")
async def get_chunks(doc_id: str, db: Session = Depends(get_db)):
    """Return chunk list for a parsed document."""
    doc = _get_parsed_doc(doc_id, db)
    return {
        "id": doc_id,
        "chunks": doc.parsed_data.get("chunks", []),
        "total": len(doc.parsed_data.get("chunks", []))
    }


@router.get("/{doc_id}/splits")
async def get_splits(doc_id: str, db: Session = Depends(get_db)):
    """Return page splits."""
    doc = _get_parsed_doc(doc_id, db)
    return {
        "id": doc_id,
        "splits": doc.parsed_data.get("splits", [])
    }


@router.get("/{doc_id}/grounding")
async def get_grounding(doc_id: str, db: Session = Depends(get_db)):
    """Return the grounding map (chunk_id → bbox + page + type + confidence)."""
    doc = _get_parsed_doc(doc_id, db)
    return {
        "id": doc_id,
        "grounding": doc.parsed_data.get("grounding", {})
    }


@router.get("/{doc_id}/markdown")
async def get_markdown(doc_id: str, db: Session = Depends(get_db)):
    """Return full markdown string."""
    doc = _get_parsed_doc(doc_id, db)
    return {
        "id": doc_id,
        "markdown": doc.parsed_data.get("markdown", "")
    }


@router.get("/{doc_id}/chunk/{chunk_id}")
async def get_single_chunk(doc_id: str, chunk_id: str, db: Session = Depends(get_db)):
    """Return a single chunk by id, with its grounding info."""
    doc = _get_parsed_doc(doc_id, db)
    chunks = doc.parsed_data.get("chunks", [])
    chunk = next((c for c in chunks if c["id"] == chunk_id), None)
    if not chunk:
        raise HTTPException(404, f"Chunk {chunk_id} not found")
    grounding = doc.parsed_data.get("grounding", {}).get(chunk_id, {})
    return {"chunk": chunk, "grounding": grounding}


# ── helpers ─────────────────────────────────────────────────────────────────

def _get_parsed_doc(doc_id: str, db: Session) -> Document:
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(404, "Document not found")
    if doc.status != "parsed" or not doc.parsed_data:
        raise HTTPException(400, f"Document not parsed. Status: {doc.status}")
    return doc


# ── Batch parse ───────────────────────────────────────────────────────────────

class BatchParseRequest(BaseModel):
    document_ids: list[str]


@router.post("/batch", tags=["Documents"])
async def batch_parse(
    req: BatchParseRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Parse multiple already-uploaded documents in the background.

    Pass a list of document_ids (status must be 'uploaded').
    Returns immediately with per-document status.
    Poll GET /api/v1/documents/{id} to check when each reaches 'parsed'.
    """
    if not req.document_ids:
        raise HTTPException(400, "No document_ids provided.")

    # Validate all docs exist and are in a parseable state
    docs_to_parse = []
    results = []
    for doc_id in req.document_ids:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            results.append({"document_id": doc_id, "status": "error", "message": "Not found"})
            continue
        if doc.status == "parsed":
            results.append({"document_id": doc_id, "status": "already_parsed", "message": "Already parsed, skipping"})
            continue
        docs_to_parse.append((doc_id, doc.file_path, doc.file_name))
        doc.status = "parsing"
        results.append({"document_id": doc_id, "status": "queued", "message": "Queued for parsing"})

    db.commit()

    if docs_to_parse:
        background_tasks.add_task(_parse_documents_background, docs_to_parse)

    return {
        "total": len(req.document_ids),
        "queued": len(docs_to_parse),
        "results": results,
        "message": f"Parsing {len(docs_to_parse)} document(s) in background. Poll each document_id for status.",
    }


async def _parse_documents_background(docs: list[tuple[str, str, str]]):
    """Parse a list of (doc_id, file_path, file_name) in the background."""
    from app.core.database import SessionLocal
    db = SessionLocal()
    try:
        for doc_id, fpath, filename in docs:
            doc = db.query(Document).filter(Document.id == doc_id).first()
            if not doc:
                continue
            try:
                parsed = parse_document(fpath)
                doc.parsed_data = parsed
                doc.page_count = parsed["metadata"].get("page_count", 1)
                doc.status = "parsed"
            except Exception as e:
                doc.status = "error"
                doc.error_message = str(e)
            db.commit()
    finally:
        db.close()
