"""
intelligence.py — Smart platform features endpoints.

POST /api/v1/intelligence/auto-schema
  Feature 1: Auto-generate a schema from a parsed document using GPT-4o.

POST /api/v1/intelligence/quality-score
  Feature 2: Compute quality score for an extraction job.

POST /api/v1/intelligence/smart-retry
  Feature 3: Re-extract low-confidence fields with targeted prompts.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from typing import Optional

from app.core.database import get_db
from app.models.document import Document
from app.models.job import ExtractionJob
from app.models.schema import SchemaDefinition
from app.services.schema_utils import normalize_schema
from loguru import logger

router = APIRouter(prefix="/intelligence", tags=["Intelligence"])


# ── Feature 1: Auto Schema Generator ─────────────────────────────────────────

class AutoSchemaRequest(BaseModel):
    document_id: str
    api_key: str = Field(..., description="OpenAI API key")
    model: str = Field(default="gpt-4o", description="GPT model to use")
    domain_hint: str = Field(default="", description="Optional domain hint (e.g. 'HVAC', 'foodservice')")
    max_fields: int = Field(default=30, description="Maximum number of fields to generate")
    save: bool = Field(default=True, description="Save the generated schema to the database")


@router.post("/auto-schema")
async def auto_generate_schema(req: AutoSchemaRequest, db: Session = Depends(get_db)):
    """
    Analyze a parsed document and automatically generate an extraction schema.
    Uses GPT-4o to understand the document and suggest relevant fields.
    """
    doc = db.query(Document).filter(Document.id == req.document_id).first()
    if not doc:
        raise HTTPException(404, f"Document '{req.document_id}' not found.")
    if doc.status != "parsed" or not doc.parsed_data:
        raise HTTPException(400, f"Document not parsed. Status: {doc.status}")

    from app.services.schema_generator import auto_generate_schema as _generate

    try:
        schema = await _generate(
            parsed_doc=doc.parsed_data,
            api_key=req.api_key,
            model=req.model,
            domain_hint=req.domain_hint,
            max_fields=req.max_fields,
        )
    except Exception as e:
        raise HTTPException(500, f"Schema generation failed: {e}")

    # Optionally save to database
    schema_id = None
    if req.save:
        import uuid
        from app.services.schema_utils import normalize_schema
        normalized = normalize_schema(schema)
        db_schema = SchemaDefinition(
            id=str(uuid.uuid4()),
            name=normalized["name"],
            description=normalized.get("description", ""),
            version=normalized.get("version", "1.0"),
            domain=normalized.get("domain", ""),
            fields=normalized["fields"],
            raw_definition=schema,
        )
        db.add(db_schema)
        db.commit()
        schema_id = db_schema.id

    return {
        "schema": schema,
        "schema_id": schema_id,
        "field_count": len(schema.get("fields", [])),
        "saved": req.save,
        "document_id": req.document_id,
        "document_name": doc.file_name,
    }


# ── Feature 2: Quality Scorer ─────────────────────────────────────────────────

class QualityScoreRequest(BaseModel):
    job_id: str = Field(..., description="Extraction job ID to score")


@router.post("/quality-score")
async def compute_quality_score(req: QualityScoreRequest, db: Session = Depends(get_db)):
    """
    Compute a quality score (0-100) and grade (A-F) for an extraction job.
    Returns detailed breakdown, missing fields, and improvement suggestions.
    """
    job = db.query(ExtractionJob).filter(ExtractionJob.id == req.job_id).first()
    if not job:
        raise HTTPException(404, f"Job '{req.job_id}' not found.")
    if job.status != "completed" or not job.result:
        raise HTTPException(400, f"Job not completed. Status: {job.status}")

    from app.services.quality_scorer import compute_quality_score, compute_quality_for_records

    result_data = job.result
    schema_fields = result_data.get("schema_fields", [])

    # Multi-record result
    if "records" in result_data:
        quality = compute_quality_for_records(
            records=result_data["records"],
            schema_fields=schema_fields,
        )
    else:
        # Single record result
        quality = compute_quality_score(
            result=result_data.get("result", {}),
            confidence=result_data.get("confidence", {}),
            sources=result_data.get("sources", {}),
            schema_fields=schema_fields,
            validation_errors=result_data.get("validation", {}),
            failure_log=result_data.get("failure_log", []),
        )

    return {
        "job_id": req.job_id,
        "schema_name": job.schema_name,
        "provider": job.provider,
        **quality,
    }


@router.get("/quality-score/{job_id}")
async def get_quality_score(job_id: str, db: Session = Depends(get_db)):
    """GET version of quality score — same as POST but via URL param."""
    job = db.query(ExtractionJob).filter(ExtractionJob.id == job_id).first()
    if not job:
        raise HTTPException(404, f"Job '{job_id}' not found.")
    if job.status != "completed" or not job.result:
        raise HTTPException(400, f"Job not completed. Status: {job.status}")

    from app.services.quality_scorer import compute_quality_score, compute_quality_for_records

    result_data = job.result
    schema_fields = result_data.get("schema_fields", [])

    if "records" in result_data:
        quality = compute_quality_for_records(
            records=result_data["records"],
            schema_fields=schema_fields,
        )
    else:
        quality = compute_quality_score(
            result=result_data.get("result", {}),
            confidence=result_data.get("confidence", {}),
            sources=result_data.get("sources", {}),
            schema_fields=schema_fields,
            validation_errors=result_data.get("validation", {}),
            failure_log=result_data.get("failure_log", []),
        )

    return {"job_id": job_id, "schema_name": job.schema_name, "provider": job.provider, **quality}


# ── Feature 3: Smart Retry ────────────────────────────────────────────────────

class SmartRetryRequest(BaseModel):
    job_id: str = Field(..., description="Extraction job ID to retry")
    provider: str = Field(default="openai", description="LLM provider for retry")
    api_key: str = Field(..., description="API key for the LLM provider")
    model: str = Field(default="gpt-4o-mini", description="Model to use for retry")
    base_url: str = Field(default="", description="Base URL (for Ollama/custom)")
    threshold: float = Field(default=0.5, description="Confidence threshold — retry fields below this")
    max_retries: int = Field(default=5, description="Maximum number of fields to retry (1-10)")


@router.post("/smart-retry")
async def smart_retry(req: SmartRetryRequest, db: Session = Depends(get_db)):
    """
    Re-extract low-confidence fields using targeted single-field prompts.
    Significantly improves accuracy for fields that were missed or uncertain.
    """
    job = db.query(ExtractionJob).filter(ExtractionJob.id == req.job_id).first()
    if not job:
        raise HTTPException(404, f"Job '{req.job_id}' not found.")
    if job.status != "completed" or not job.result:
        raise HTTPException(400, f"Job not completed. Status: {job.status}")

    # Get the document
    doc = db.query(Document).filter(Document.id == job.document_id).first()
    if not doc or not doc.parsed_data:
        raise HTTPException(400, "Document not found or not parsed.")

    # Get the schema field definitions
    schema_def = None
    if job.schema_id:
        schema_def = db.query(SchemaDefinition).filter(SchemaDefinition.id == job.schema_id).first()

    if not schema_def:
        raise HTTPException(400, "Schema not found. Smart retry requires a saved schema.")

    from app.services.smart_retry import smart_retry_low_confidence
    from app.services.llm_router import LLMRouter
    from app.services.quality_scorer import compute_quality_score

    # Cap max_retries
    max_retries = max(1, min(req.max_retries, 10))

    llm = LLMRouter(
        provider=req.provider,
        api_key=req.api_key,
        model=req.model,
        base_url=req.base_url,
    )

    if not llm.is_available():
        raise HTTPException(400, f"Provider '{req.provider}' is not available. Check API key.")

    result_data = job.result

    # Only works on single-record results for now
    if "records" in result_data:
        raise HTTPException(400, "Smart retry currently supports single-record extractions only.")

    normalized_schema = normalize_schema(
        schema_def.raw_definition or {"name": schema_def.name, "fields": schema_def.fields}
    )
    schema_fields_def = normalized_schema["fields"]

    try:
        retry_result = await smart_retry_low_confidence(
            result=result_data.get("result", {}),
            confidence=result_data.get("confidence", {}),
            sources=result_data.get("sources", {}),
            evidence=result_data.get("evidence", {}),
            schema_fields_def=schema_fields_def,
            parsed_doc=doc.parsed_data,
            llm=llm,
            threshold=req.threshold,
            max_retries=max_retries,
        )
    except Exception as e:
        raise HTTPException(500, f"Smart retry failed: {e}")

    # Count improvements
    improved = [r for r in retry_result["retry_log"] if r.get("status") == "improved"]
    no_improvement = [r for r in retry_result["retry_log"] if r.get("status") == "no_improvement"]

    # Update job result with improved values
    updated_result_data = {
        **result_data,
        "result": retry_result["result"],
        "confidence": retry_result["confidence"],
        "sources": retry_result["sources"],
        "evidence": retry_result["evidence"],
        "smart_retry_log": retry_result["retry_log"],
    }
    job.result = updated_result_data
    db.commit()

    # Compute new quality score
    schema_fields = result_data.get("schema_fields", [])
    new_quality = compute_quality_score(
        result=retry_result["result"],
        confidence=retry_result["confidence"],
        sources=retry_result["sources"],
        schema_fields=schema_fields,
        validation_errors=result_data.get("validation", {}),
        failure_log=result_data.get("failure_log", []),
    )

    return {
        "job_id": req.job_id,
        "status": "completed",
        "fields_retried": len(retry_result["retry_log"]),
        "fields_improved": len(improved),
        "fields_no_improvement": len(no_improvement),
        "improved_fields": [r["field"] for r in improved],
        "retry_log": retry_result["retry_log"],
        "new_quality_score": new_quality["score"],
        "new_grade": new_quality["grade"],
        "result": retry_result["result"],
        "confidence": retry_result["confidence"],
        "sources": retry_result["sources"],
    }
