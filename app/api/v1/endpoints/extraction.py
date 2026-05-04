"""
extraction.py — Schema-driven extraction API endpoint.

POST /api/v1/extraction/run
  Run extraction against an already-parsed document + schema definition.

POST /api/v1/extraction/run-inline
  Upload a document + schema in one request (parse + extract in one shot).

GET  /api/v1/extraction/{job_id}
  Retrieve a previous extraction result.

DELETE /api/v1/extraction/{job_id}
  Remove an extraction record.

The schema passed here is fully generic — see schema_utils.py for the
supported field types (string, number, integer, boolean, date, currency,
email, phone, url, list, object, list[object]).
"""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from typing import Any, Optional
import json, shutil, os

from app.core.config import settings
from app.core.database import get_db
from app.core.auth import get_current_user_optional
from loguru import logger
from app.models.document import Document
from app.models.job import ExtractionJob
from app.models.schema import SchemaDefinition
from app.services.parser import parse_document
from app.services.pipeline import run_extraction

router = APIRouter(prefix="/extraction", tags=["Extraction"])


# ── Request/Response models ───────────────────────────────────────────────────

class ProviderConfig(BaseModel):
    provider: str = Field(default="none",
        description=(
            "Extraction engine: "
            "openai | chatgpt | anthropic | gemini | groq | grok | emergence | "
            "ollama | landingai | python | hybrid | none. "
            "'python' = heuristic only. "
            "'hybrid' = heuristic + AI fallback. "
            "'none' = heuristic only (no AI)."
        ))
    api_key: str = Field(default="", description="API key (BYOK — not stored after request)")
    model: str = Field(default="", description="Model name (leave empty for provider default)")
    base_url: str = Field(default="",
        description="Base URL: Ollama endpoint, xAI base URL, Emergence base URL, or Landing AI environment")


class ExtractionRequest(BaseModel):
    document_id: str = Field(...,
        description="ID of an already-parsed document")
    schema_id: Optional[str] = Field(default=None,
        description="ID of a saved schema (from POST /api/v1/schemas). Use this OR 'schema'.")
    schema: Optional[dict] = Field(default=None,
        description="Inline extraction schema. Use this OR 'schema_id'.")
    provider_config: ProviderConfig = Field(default_factory=ProviderConfig)
    options: dict = Field(default_factory=dict,
        description="Optional overrides: {confidence_threshold, max_retries, ...}")


class InlineExtractionRequest(BaseModel):
    schema: dict = Field(...,
        description="Extraction schema")
    provider_config: ProviderConfig = Field(default_factory=ProviderConfig)
    options: dict = Field(default_factory=dict)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/run")
async def run_extraction_endpoint(
    req: ExtractionRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user_optional),
):
    """
    Run extraction on an already-parsed document.
    Provide either 'schema_id' (reference a saved schema) or 'schema' (inline definition).
    Returns the full extraction result immediately (synchronous).
    """
    # Resolve schema — prefer schema_id over inline schema
    if req.schema_id:
        saved = db.query(SchemaDefinition).filter(SchemaDefinition.id == req.schema_id).first()
        if not saved:
            raise HTTPException(404, f"Schema '{req.schema_id}' not found.")
        schema_dict = saved.raw_definition or {"name": saved.name, "fields": saved.fields}
        schema_name = saved.name
    elif req.schema:
        schema_dict = req.schema
        schema_name = req.schema.get("name", "inline")
    else:
        raise HTTPException(400, "Provide either 'schema_id' or 'schema'.")
    doc = db.query(Document).filter(Document.id == req.document_id).first()
    if not doc:
        raise HTTPException(404, f"Document '{req.document_id}' not found.")
    if doc.status != "parsed" or not doc.parsed_data:
        raise HTTPException(400, f"Document not parsed. Status: {doc.status}")

    job_id = str(uuid.uuid4())
    job = ExtractionJob(
        id=job_id,
        user_id=current_user.id if current_user else None,
        document_id=req.document_id,
        schema_name=schema_name,
        schema_id=req.schema_id,
        status="running",
        provider=req.provider_config.provider,
        model=req.provider_config.model,
    )
    db.add(job)
    db.commit()

    try:
        provider = req.provider_config.provider.lower()

        if provider == "landingai":
            from app.services.landingai_service import (
                extract_with_landingai, extract_multi_with_landingai,
                parse_with_landingai
            )
            if not req.provider_config.api_key:
                raise HTTPException(400, "Landing AI requires an api_key in provider_config.")

            environment = req.provider_config.base_url or "production"
            multi_record = req.options.get("multi_record", False)
            vision_parse = req.options.get("vision_parse", False)
            markdown = doc.parsed_data.get("markdown", "")

            if vision_parse:
                # User explicitly enabled vision mode — always use LandingAI vision parser
                try:
                    logger.info(f"Vision mode enabled — parsing with LandingAI vision: {doc.file_path}")
                    from app.services.vision_parser import vision_parse_document
                    vision_markdown = await vision_parse_document(
                        file_path=doc.file_path,
                        provider="landingai",
                        api_key=req.provider_config.api_key,
                        base_url=environment,
                    )
                    if vision_markdown:
                        markdown = vision_markdown
                        logger.info(f"Vision markdown: {len(markdown)} chars")
                except Exception as e:
                    logger.warning(f"Vision parse failed, using stored markdown: {e}")
            else:
                # Auto-detect: only re-parse if diagram values are image-only
                import re as _re
                has_dimensions_label = bool(_re.search(r'DIMENSIONS?:', markdown, _re.IGNORECASE))
                has_dimension_numbers = bool(_re.search(
                    r'\d+[-\s]\d+/\d+["\']?|\b\d{1,3}\.\d{1,4}["\']?\s*(?:in|inch|")',
                    markdown
                ))
                file_path = doc.file_path
                import os
                if has_dimensions_label and not has_dimension_numbers and file_path and os.path.exists(file_path):
                    try:
                        logger.info(f"Diagram-only PDF detected, re-parsing with LandingAI vision: {file_path}")
                        landingai_parsed = await parse_with_landingai(
                            file_path=file_path,
                            api_key=req.provider_config.api_key,
                            environment=environment,
                        )
                        markdown = landingai_parsed.get("markdown", "")
                    except Exception as e:
                        logger.warning(f"LandingAI vision parse failed, using stored markdown: {e}")

            if multi_record:
                extraction = await extract_multi_with_landingai(
                    markdown=markdown,
                    schema=schema_dict,
                    api_key=req.provider_config.api_key,
                    environment=environment,
                )
            else:
                extraction = await extract_with_landingai(
                    markdown=markdown,
                    schema=schema_dict,
                    api_key=req.provider_config.api_key,
                    environment=environment,
                )
        else:
            vision_parse = req.options.get("vision_parse", False)
            multi_record = req.options.get("multi_record", False)

            if vision_parse and provider in ("openai", "chatgpt", "anthropic", "gemini"):
                # Vision mode for non-LandingAI providers
                try:
                    import os
                    if doc.file_path and os.path.exists(doc.file_path):
                        logger.info(f"Vision mode enabled for {provider} — parsing PDF as images")
                        from app.services.vision_parser import vision_parse_document
                        vision_markdown = await vision_parse_document(
                            file_path=doc.file_path,
                            provider=provider,
                            api_key=req.provider_config.api_key,
                            model=req.provider_config.model,
                        )
                        if vision_markdown:
                            # Merge vision markdown with existing parsed data
                            existing_markdown = doc.parsed_data.get("markdown", "")
                            merged_parsed = dict(doc.parsed_data)
                            merged_parsed["markdown"] = existing_markdown + "\n\n<!-- VISION PARSE -->\n\n" + vision_markdown
                            extraction = await run_extraction(
                                parsed_doc=merged_parsed,
                                schema=schema_dict,
                                provider_config=req.provider_config.dict(),
                            )
                        else:
                            extraction = await run_extraction(
                                parsed_doc=doc.parsed_data,
                                schema=schema_dict,
                                provider_config=req.provider_config.dict(),
                            )
                    else:
                        extraction = await run_extraction(
                            parsed_doc=doc.parsed_data,
                            schema=schema_dict,
                            provider_config=req.provider_config.dict(),
                        )
                except Exception as e:
                    logger.warning(f"Vision parse failed for {provider}: {e}, falling back to text")
                    extraction = await run_extraction(
                        parsed_doc=doc.parsed_data,
                        schema=schema_dict,
                        provider_config=req.provider_config.dict(),
                    )
            else:
                extraction = await run_extraction(
                    parsed_doc=doc.parsed_data,
                    schema=schema_dict,
                    provider_config=req.provider_config.dict(),
                )
        job.status = "completed"
        job.result = extraction
        db.commit()
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        logger.error(f"Extraction failed:\n{error_detail}")
        job.status = "failed"
        job.error = str(e) or type(e).__name__
        db.commit()
        raise HTTPException(500, f"Extraction failed: {type(e).__name__}: {e}")

    return {
        "job_id": job_id,
        "document_id": req.document_id,
        "schema_name": schema_name,
        "status": "completed",
        **extraction,
    }


@router.post("/run-inline")
async def run_inline_extraction(
    schema: str = Form(..., description="JSON string of extraction schema"),
    file: UploadFile = File(...),
    provider: str = Form(default="none"),
    api_key: str = Form(default=""),
    model: str = Form(default=""),
    base_url: str = Form(default=""),
    db: Session = Depends(get_db),
):
    """
    Upload a document + schema and get extraction results in one shot.
    Parses the document then immediately runs extraction.
    """
    # Parse schema
    try:
        schema_dict = json.loads(schema)
    except Exception:
        raise HTTPException(400, "Invalid JSON in 'schema' field.")

    # Save uploaded file
    doc_id = str(uuid.uuid4())
    ext = Path(file.filename or "file.bin").suffix
    fpath = os.path.join(settings.UPLOAD_DIR, f"{doc_id}{ext}")
    with open(fpath, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Parse document
    try:
        parsed = parse_document(fpath, file.content_type or "")
    except Exception as e:
        raise HTTPException(500, f"Document parsing failed: {e}")

    # Save document record
    doc = Document(
        id=doc_id,
        file_name=file.filename,
        file_path=fpath,
        file_size=os.path.getsize(fpath),
        mime_type=file.content_type or "application/octet-stream",
        status="parsed",
        parsed_data=parsed,
        page_count=parsed["metadata"].get("page_count", 1),
    )
    db.add(doc)
    db.commit()

    # Run extraction
    provider_config = {
        "provider": provider, "api_key": api_key,
        "model": model, "base_url": base_url,
    }
    job_id = str(uuid.uuid4())
    job = ExtractionJob(
        id=job_id,
        document_id=doc_id,
        schema_name=schema_dict.get("name", "inline"),
        status="running",
    )
    db.add(job)
    db.commit()

    try:
        extraction = await run_extraction(
            parsed_doc=parsed,
            schema=schema_dict,
            provider_config=provider_config,
        )
        job.status = "completed"
        job.result = extraction
        db.commit()
    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        db.commit()
        raise HTTPException(500, f"Extraction failed: {e}")

    return {
        "job_id": job_id,
        "document_id": doc_id,
        "schema_name": schema_dict.get("name", "inline"),
        "status": "completed",
        "parse_metadata": parsed["metadata"],
        **extraction,
    }


@router.get("/run/{job_id}")
async def get_extraction_result(job_id: str, db: Session = Depends(get_db)):
    """Retrieve a previous extraction result by job ID."""
    job = db.query(ExtractionJob).filter(ExtractionJob.id == job_id).first()
    if not job:
        raise HTTPException(404, f"Extraction job '{job_id}' not found.")
    return {
        "job_id": job.id,
        "document_id": job.document_id,
        "schema_name": job.schema_name,
        "status": job.status,
        "error": getattr(job, "error", None),
        **(job.result or {}),
    }


@router.delete("/run/{job_id}")
async def delete_extraction(job_id: str, db: Session = Depends(get_db)):
    job = db.query(ExtractionJob).filter(ExtractionJob.id == job_id).first()
    if not job:
        raise HTTPException(404, f"Extraction job '{job_id}' not found.")
    db.delete(job)
    db.commit()
    return {"deleted": job_id}


@router.get("/document/{doc_id}")
async def list_extractions_for_document(doc_id: str, db: Session = Depends(get_db)):
    """List all extraction jobs for a given document."""
    jobs = db.query(ExtractionJob).filter(ExtractionJob.document_id == doc_id).all()
    return {
        "document_id": doc_id,
        "extractions": [
            {
                "job_id": j.id,
                "schema_name": j.schema_name,
                "status": j.status,
                "created_at": j.created_at.isoformat() if j.created_at else None,
            }
            for j in jobs
        ],
    }


# ── Schema field type reference ───────────────────────────────────────────────

@router.get("/field-types")
async def get_field_types():
    """Return supported field types and their schema options."""
    return {
        "scalar_types": [
            "string", "number", "integer", "boolean",
            "date", "currency", "email", "phone", "url",
        ],
        "complex_types": {
            "list": {
                "description": "A list of scalar values (strings by default).",
                "example": {
                    "name": "features",
                    "type": "list",
                    "source_labels": ["features", "standard features"],
                },
            },
            "list[object]": {
                "description": (
                    "A list of structured objects extracted from table rows "
                    "or repeated document sections. Define sub-fields in 'fields'."
                ),
                "example": {
                    "name": "line_items",
                    "type": "list",
                    "fields": [
                        {"name": "model", "type": "string",
                         "table_labels": ["model", "part number"]},
                        {"name": "quantity", "type": "integer",
                         "table_labels": ["qty", "quantity"]},
                        {"name": "price", "type": "currency",
                         "table_labels": ["price", "unit price"]},
                    ],
                },
            },
            "object": {
                "description": (
                    "A single structured object with named sub-fields. "
                    "Each sub-field is extracted independently."
                ),
                "example": {
                    "name": "contact_information",
                    "type": "object",
                    "fields": [
                        {"name": "address", "type": "string",
                         "source_labels": ["address"]},
                        {"name": "phone", "type": "phone",
                         "source_labels": ["phone", "tel"]},
                        {"name": "fax", "type": "phone",
                         "source_labels": ["fax"]},
                    ],
                },
            },
        },
        "field_options": {
            "source_labels": "Text labels to search for in document body",
            "table_labels": "Column headers to match in tables",
            "document_labels": "Section headings or document-level labels",
            "preferred_sources": "Search order: table | kv | text",
            "required": "Raise failure_log entry if not found",
            "fallback": "Default value when field cannot be extracted",
            "normalization_rules": "Post-processing: strip | uppercase | lowercase | title_case | normalize_date | remove_currency | digits_only",
            "validation_rules": "Validation: not_empty | numeric | positive | min_length:N | max_length:N | email",
            "confidence_threshold": "Below this confidence (0.0–1.0), AI fallback is triggered",
            "null_values": "Values to treat as null (e.g. ['N/A', '-', ''])",
        },
    }
