from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Optional
from app.core.database import get_db
from app.core.auth import get_current_user_optional
from app.core.user_filter import filter_by_user, owned_by
from app.models.job import ExtractionJob
from app.models.user import User

router = APIRouter(prefix="/jobs", tags=["Jobs"])


@router.get("")
async def list_jobs(
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    try:
        query = db.query(ExtractionJob).order_by(ExtractionJob.created_at.desc()).limit(200)
        query = filter_by_user(query, current_user, ExtractionJob)
        jobs = query.all()
        result = []
        for j in jobs:
            try:
                result.append({
                    "job_id": j.id,
                    "document_id": j.document_id,
                    "schema_id": getattr(j, "schema_id", None),
                    "schema_name": getattr(j, "schema_name", None),
                    "batch_id": getattr(j, "batch_id", None),
                    "status": j.status,
                    "provider": j.provider,
                    "model": getattr(j, "model", None),
                    "duration_seconds": getattr(j, "duration_seconds", None),
                    "result": j.result,
                    "created_at": j.created_at.isoformat() if j.created_at else None,
                    "updated_at": j.updated_at.isoformat() if getattr(j, "updated_at", None) else None,
                })
            except Exception:
                pass
        return {"jobs": result}
    except Exception as e:
        # Fallback: raw SQL query if ORM fails (e.g. missing columns)
        from sqlalchemy import text
        try:
            rows = db.execute(text("SELECT id, document_id, status, provider, created_at FROM extraction_jobs ORDER BY created_at DESC LIMIT 200")).fetchall()
            return {"jobs": [
                {"job_id": r[0], "document_id": r[1], "status": r[2], "provider": r[3],
                 "created_at": r[4], "schema_name": None, "batch_id": None, "result": None}
                for r in rows
            ]}
        except Exception:
            return {"jobs": []}


@router.get("/{job_id}")
async def get_job(job_id: str, db: Session = Depends(get_db)):
    job = db.query(ExtractionJob).filter(ExtractionJob.id == job_id).first()
    if not job:
        raise HTTPException(404, "Job not found")
    return {
        "job_id": job.id,
        "document_id": job.document_id,
        "schema_id": getattr(job, "schema_id", None),
        "schema_name": getattr(job, "schema_name", None),
        "status": job.status,
        "provider": job.provider,
        "model": getattr(job, "model", None),
        "result": job.result,
        "confidence": getattr(job, "confidence", None),
        "sources": getattr(job, "sources", None),
        "evidence": getattr(job, "evidence", None),
        "schema_fields": getattr(job, "schema_fields", None),
        "validation": getattr(job, "validation_errors", None),
        "failure_log": getattr(job, "failure_log", None),
        "duration_seconds": getattr(job, "duration_seconds", None),
        "error_message": getattr(job, "error_message", None),
        "created_at": job.created_at.isoformat() if job.created_at else None
    }


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str, db: Session = Depends(get_db)):
    job = db.query(ExtractionJob).filter(ExtractionJob.id == job_id).first()
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status in ("completed", "failed"):
        raise HTTPException(400, f"Cannot cancel job with status: {job.status}")
    job.status = "cancelled"
    db.commit()
    return {"job_id": job_id, "status": "cancelled"}


@router.delete("/{job_id}")
async def delete_job(job_id: str, db: Session = Depends(get_db)):
    job = db.query(ExtractionJob).filter(ExtractionJob.id == job_id).first()
    if not job:
        raise HTTPException(404, "Job not found")
    db.delete(job)
    db.commit()
    return {"deleted": job_id}
