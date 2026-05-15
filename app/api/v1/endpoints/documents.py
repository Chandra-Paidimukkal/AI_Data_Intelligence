"""
documents.py — Document upload, listing, and parsed-data retrieval.
All data is scoped to the authenticated user.
"""
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
import shutil
import uuid
import os
from pathlib import Path

from app.core.database import get_db
from app.core.config import settings
from app.core.auth import get_current_user, get_current_user_optional
from app.core.user_filter import filter_by_user, owned_by
from app.models.document import Document
from app.models.user import User
from app.services.parser import parse_document

router = APIRouter(prefix="/documents", tags=["Documents"])


def save_file(upload: UploadFile) -> tuple:
    doc_id = str(uuid.uuid4())
    ext = Path(upload.filename or "file.bin").suffix
    fname = f"{doc_id}{ext}"
    fpath = os.path.join(settings.UPLOAD_DIR, fname)
    with open(fpath, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    size = os.path.getsize(fpath)
    return doc_id, fpath, size


def _parse_in_background(doc_id: str, file_path: str, mime_type: str):
    from app.core.database import SessionLocal
    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            return
        try:
            doc.status = "parsing"
            db.commit()
            parsed = parse_document(file_path, mime_type)
            doc.parsed_data = parsed
            doc.page_count = parsed["metadata"]["page_count"]
            doc.status = "parsed"
        except Exception as e:
            doc.status = "error"
            doc.error_message = str(e)
        finally:
            db.commit()
    finally:
        db.close()


@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    doc_id, fpath, size = save_file(file)
    doc = Document(
        id=doc_id,
        user_id=current_user.id if current_user else None,
        file_name=file.filename,
        file_path=fpath,
        file_size=size,
        mime_type=file.content_type or "application/octet-stream",
        status="uploaded",
    )
    db.add(doc)
    db.commit()
    background_tasks.add_task(_parse_in_background, doc_id, fpath, file.content_type or "")
    return {"id": doc_id, "file_name": file.filename, "status": "uploaded"}


@router.post("/upload/batch")
async def upload_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    results = []
    for file in files:
        doc_id, fpath, size = save_file(file)
        doc = Document(
            id=doc_id,
            user_id=current_user.id if current_user else None,
            file_name=file.filename,
            file_path=fpath,
            file_size=size,
            mime_type=file.content_type or "application/octet-stream",
            status="uploaded",
        )
        db.add(doc)
        db.commit()
        background_tasks.add_task(_parse_in_background, doc_id, fpath, file.content_type or "")
        results.append({"id": doc_id, "file_name": file.filename, "status": "uploaded"})
    return {"documents": results, "count": len(results)}


@router.get("")
async def list_documents(
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    query = db.query(Document).order_by(Document.created_at.desc())
    query = filter_by_user(query, current_user, Document)
    docs = query.all()
    return {"documents": [
        {
            "id": d.id,
            "file_name": d.file_name,
            "file_path": d.file_path,
            "file_size": d.file_size,
            "mime_type": d.mime_type,
            "page_count": d.page_count,
            "status": d.status,
            "chunk_count": len(d.parsed_data.get("chunks", [])) if d.parsed_data else 0,
            "table_count": d.parsed_data.get("metadata", {}).get("table_count", 0) if d.parsed_data else 0,
            "created_at": d.created_at.isoformat() if d.created_at else None,
        }
        for d in docs
    ]}


@router.get("/{doc_id}")
async def get_document(
    doc_id: str,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(404, "Document not found")
    if not owned_by(current_user, doc):
        raise HTTPException(403, "Access denied")
    meta = doc.parsed_data.get("metadata", {}) if doc.parsed_data else {}
    return {
        "id": doc.id, "file_name": doc.file_name, "file_path": doc.file_path,
        "file_size": doc.file_size,
        "mime_type": doc.mime_type, "page_count": doc.page_count, "status": doc.status,
        "error_message": doc.error_message, "parse_metadata": meta,
        "chunk_count": len(doc.parsed_data.get("chunks", [])) if doc.parsed_data else 0,
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
    }


@router.get("/{doc_id}/parsed")
async def get_parsed(
    doc_id: str,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(404, "Document not found")
    if not owned_by(current_user, doc):
        raise HTTPException(403, "Access denied")
    if doc.status != "parsed":
        raise HTTPException(400, f"Document not yet parsed. Status: {doc.status}")
    return doc.parsed_data


@router.get("/{doc_id}/summary")
async def get_parsed_summary(
    doc_id: str,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(404, "Document not found")
    if not owned_by(current_user, doc):
        raise HTTPException(403, "Access denied")
    if doc.status != "parsed":
        raise HTTPException(400, f"Document not parsed. Status: {doc.status}")
    pd = doc.parsed_data
    chunk_types: dict = {}
    for c in pd.get("chunks", []):
        t = c.get("type", "unknown")
        chunk_types[t] = chunk_types.get(t, 0) + 1
    return {
        "id": doc_id,
        "page_count": pd.get("metadata", {}).get("page_count", 0),
        "chunk_count": len(pd.get("chunks", [])),
        "chunk_type_breakdown": chunk_types,
        "table_count": pd.get("metadata", {}).get("table_count", 0),
        "kv_pair_count": len(pd.get("kv_pairs", [])),
        "section_count": len(pd.get("sections", [])),
        "ocr_used": pd.get("metadata", {}).get("ocr_used", False),
        "version": pd.get("metadata", {}).get("version", ""),
    }


@router.post("/{doc_id}/reparse")
async def reparse_document(
    doc_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(404, "Document not found")
    if not owned_by(current_user, doc):
        raise HTTPException(403, "Access denied")
    doc.status = "uploaded"
    doc.parsed_data = None
    doc.error_message = None
    db.commit()
    background_tasks.add_task(_parse_in_background, doc_id, doc.file_path, doc.mime_type)
    return {"id": doc_id, "status": "reparsing"}


@router.delete("/{doc_id}")
async def delete_document(
    doc_id: str,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(404, "Document not found")
    if not owned_by(current_user, doc):
        raise HTTPException(403, "Access denied")
    try:
        if os.path.exists(doc.file_path):
            os.remove(doc.file_path)
    except Exception:
        pass
    db.delete(doc)
    db.commit()
    return {"deleted": doc_id}
