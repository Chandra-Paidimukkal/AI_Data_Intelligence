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
    query = db.query(ExtractionJob).order_by(ExtractionJob.created_at.desc()).limit(200)
    query = filter_by_user(query, current_user, ExtractionJob)
    jobs = query.all()
    return {"jobs": [
        {
            "job_id": j.id,
            "document_id": j.document_id,
            "schema_id": j.schema_id,
            "schema_name": j.schema_name,
            "batch_id": j.batch_id,
            "status": j.status,
            "provider": j.provider,
            "model": j.model,
            "duration_seconds": j.duration_seconds,
            "result": j.result,
            "created_at": j.created_at.isoformat() if j.created_at else None,
            "updated_at": j.updated_at.isoformat() if j.updated_at else None,
        }
        for j in jobs
    ]}


@router.get("/{job_id}")
async def get_job(job_id: str, db: Session = Depends(get_db)):
    job = db.query(ExtractionJob).filter(ExtractionJob.id == job_id).first()
    if not job:
        raise HTTPException(404, "Job not found")
    return {
        "job_id": job.id,
        "document_id": job.document_id,
        "schema_id": job.schema_id,
        "schema_name": job.schema_name,
        "status": job.status,
        "provider": job.provider,
        "model": job.model,
        "result": job.result,
        "confidence": job.confidence,
        "sources": job.sources,
        "evidence": job.evidence,
        "schema_fields": job.schema_fields,
        "validation": job.validation_errors,
        "failure_log": job.failure_log,
        "duration_seconds": job.duration_seconds,
        "error_message": job.error_message,
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
