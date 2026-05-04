"""
landingai_service.py — Landing AI ADE integration.

Provides:
  1. parse_with_landingai()         — Parse with Landing AI DPT-2 model.
  2. extract_with_landingai()       — Single-record extraction (auto-chunks large docs).
  3. extract_multi_with_landingai() — Multi-record extraction (one result per model/variant).

For large PDFs (100+ pages), documents are automatically split into chunks,
each chunk is extracted independently, and results are merged.

API key passed per-request (BYOK) via provider_config.api_key.
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from loguru import logger

# Max chars per chunk sent to Landing AI (~8-10 pages worth)
_CHUNK_SIZE = 80_000
# Max concurrent Landing AI requests
_MAX_CONCURRENT = 3


# ── Parse ─────────────────────────────────────────────────────────────────────

async def parse_with_landingai(file_path: str, api_key: str, environment: str = "production") -> dict:
    """Parse a document using Landing AI's DPT-2 model."""
    try:
        from landingai_ade import AsyncLandingAIADE
    except ImportError:
        raise RuntimeError("landingai-ade is not installed. Run: pip install landingai-ade==1.12.0")

    env = _resolve_environment(environment)
    logger.info(f"Parsing with Landing AI ADE (env={env}): {file_path}")

    async with AsyncLandingAIADE(apikey=api_key, environment=env) as client:
        response = await client.parse(document=Path(file_path), model="dpt-2-latest")

    return _ade_response_to_parsed_doc(response, file_path)


# ── Single-record Extract ─────────────────────────────────────────────────────

async def extract_with_landingai(
    markdown: str,
    schema: dict,
    api_key: str,
    environment: str = "production",
) -> dict:
    """
    Single-record extraction via Landing AI REST API.
    For large documents, splits into chunks, extracts each, and merges
    by keeping the best (most complete) value per field.
    """
    chunks = _split_markdown(markdown, _CHUNK_SIZE)
    logger.info(f"Extracting (single) [{len(markdown)} chars, {len(chunks)} chunk(s)], schema: {schema.get('name', 'inline')}")

    if len(chunks) == 1:
        return await _extract_chunk(chunks[0], schema, api_key, environment)

    # Extract all chunks concurrently (with limit)
    results = await _extract_chunks_parallel(chunks, schema, api_key, environment, multi=False)

    # Merge: pick best value per field across all chunks
    return _merge_single_results(results, schema)


# ── Multi-record Extract ──────────────────────────────────────────────────────

async def extract_multi_with_landingai(
    markdown: str,
    schema: dict,
    api_key: str,
    environment: str = "production",
) -> dict:
    """
    Multi-record extraction — extracts ALL model variants.
    For large documents, splits into chunks and collects all records,
    then deduplicates by model_number.
    """
    chunks = _split_markdown(markdown, _CHUNK_SIZE)
    logger.info(f"Extracting (multi) [{len(markdown)} chars, {len(chunks)} chunk(s)], schema: {schema.get('name', 'inline')}")

    if len(chunks) == 1:
        return await _extract_chunk_multi(chunks[0], schema, api_key, environment)

    # Extract all chunks concurrently
    results = await _extract_chunks_parallel(chunks, schema, api_key, environment, multi=True)

    # Merge all records from all chunks, deduplicate
    return _merge_multi_results(results, schema)


# ── Chunk splitter ────────────────────────────────────────────────────────────

def _split_markdown(markdown: str, chunk_size: int) -> list[str]:
    """
    Split markdown into chunks of ~chunk_size chars.
    Splits at page break boundaries to keep pages intact.
    """
    if len(markdown) <= chunk_size:
        return [markdown]

    # Split at page breaks
    pages = re.split(r"<!-- PAGE BREAK -->", markdown)
    chunks = []
    current = ""

    for page in pages:
        if len(current) + len(page) > chunk_size and current:
            chunks.append(current.strip())
            current = page
        else:
            current += ("\n\n<!-- PAGE BREAK -->\n\n" if current else "") + page

    if current.strip():
        chunks.append(current.strip())

    logger.info(f"Split {len(markdown)} chars into {len(chunks)} chunks")
    return chunks


# ── Parallel extraction ───────────────────────────────────────────────────────

async def _extract_chunks_parallel(
    chunks: list[str],
    schema: dict,
    api_key: str,
    environment: str,
    multi: bool,
) -> list[dict]:
    """Extract multiple chunks with concurrency limit."""
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT)

    async def extract_one(chunk: str, idx: int) -> dict:
        async with semaphore:
            logger.info(f"Extracting chunk {idx+1}/{len(chunks)} [{len(chunk)} chars]")
            try:
                if multi:
                    return await _extract_chunk_multi(chunk, schema, api_key, environment)
                else:
                    return await _extract_chunk(chunk, schema, api_key, environment)
            except Exception as e:
                logger.warning(f"Chunk {idx+1} failed: {e}")
                return {}

    tasks = [extract_one(chunk, i) for i, chunk in enumerate(chunks)]
    return await asyncio.gather(*tasks)


# ── Single chunk extraction ───────────────────────────────────────────────────

def _convert_fractions_in_markdown(markdown: str) -> str:
    """
    Convert fractional inch measurements like 13-3/4" to decimals (13.75")
    so LandingAI can extract them as numbers.
    Also converts standalone fractions like 3/4 → 0.75
    """
    import re

    def frac_to_decimal(m):
        whole = int(m.group(1)) if m.group(1) else 0
        num = int(m.group(2))
        den = int(m.group(3))
        val = whole + num / den
        return f"{val:.4f}".rstrip('0').rstrip('.')

    # Pattern: 13-3/4 or 13 3/4
    markdown = re.sub(r'(\d+)[-\s](\d+)/(\d+)', frac_to_decimal, markdown)
    # Pattern: standalone 3/4
    markdown = re.sub(r'\b(\d+)/(\d+)\b', lambda m: f"{int(m.group(1))/int(m.group(2)):.4f}".rstrip('0').rstrip('.'), markdown)
    return markdown


async def _extract_chunk(
    markdown: str,
    schema: dict,
    api_key: str,
    environment: str,
) -> dict:
    """Extract a single chunk — single record mode. Retries once on timeout."""
    import httpx
    # Convert fractional measurements to decimals for better extraction
    markdown = _convert_fractions_in_markdown(markdown)
    ade_schema = _schema_to_ade_format(schema)
    url, headers = _build_request(api_key, environment)
    files = {"markdown": ("upload.md", markdown.encode("utf-8"), "text/markdown")}
    data = {"schema": json.dumps(ade_schema), "strict": "false"}

    for attempt in range(2):  # try twice
        try:
            async with httpx.AsyncClient(timeout=300) as client:  # 5 min timeout
                r = await client.post(url, files=files, data=data, headers=headers)
            if r.status_code not in (200, 206):
                raise RuntimeError(f"Landing AI error {r.status_code}: {r.text[:300]}")
            return _ade_extract_response_to_pipeline_result(r.json(), schema)
        except (httpx.ReadTimeout, httpx.ConnectTimeout) as e:
            if attempt == 0:
                logger.warning(f"Timeout on attempt 1, retrying...")
                await asyncio.sleep(5)
            else:
                raise RuntimeError(f"Landing AI timed out after 2 attempts: {e}")


async def _extract_chunk_multi(
    markdown: str,
    schema: dict,
    api_key: str,
    environment: str,
) -> dict:
    """Extract a single chunk — multi-record mode. Retries once on timeout."""
    import httpx
    # Convert fractional measurements to decimals for better extraction
    markdown = _convert_fractions_in_markdown(markdown)
    field_schema = _schema_to_ade_format(schema)
    array_schema = {
        "type": "object",
        "properties": {
            "records": {
                "type": "array",
                "description": (
                    "Extract ALL equipment models/variants found in the document. "
                    "Each element represents one distinct model with its own specifications."
                ),
                "items": field_schema,
            }
        },
    }
    url, headers = _build_request(api_key, environment)
    files = {"markdown": ("upload.md", markdown.encode("utf-8"), "text/markdown")}
    data = {"schema": json.dumps(array_schema), "strict": "false"}

    for attempt in range(2):  # try twice
        try:
            async with httpx.AsyncClient(timeout=300) as client:  # 5 min timeout
                r = await client.post(url, files=files, data=data, headers=headers)
            if r.status_code not in (200, 206):
                raise RuntimeError(f"Landing AI error {r.status_code}: {r.text[:300]}")
            return _ade_multi_response_to_pipeline_result(r.json(), schema)
        except (httpx.ReadTimeout, httpx.ConnectTimeout) as e:
            if attempt == 0:
                logger.warning(f"Timeout on attempt 1, retrying...")
                await asyncio.sleep(5)
            else:
                raise RuntimeError(f"Landing AI timed out after 2 attempts: {e}")


# ── Result mergers ────────────────────────────────────────────────────────────

def _merge_single_results(results: list[dict], schema: dict) -> dict:
    """
    Merge single-record results from multiple chunks.
    For each field, keep the value with the highest confidence.
    """
    fields = schema.get("fields", [])
    field_names = [f["name"] for f in fields]

    merged_result: dict = {}
    merged_conf: dict = {}
    merged_sources: dict = {}
    merged_evidence: dict = {}
    failure_log: list = []
    total_duration = 0.0

    for res in results:
        if not res:
            continue
        result = res.get("result", {})
        conf = res.get("confidence", {})
        sources = res.get("sources", {})
        evidence = res.get("evidence", {})
        total_duration += res.get("duration_seconds", 0)
        failure_log.extend(res.get("failure_log", []))

        for fname in field_names:
            val = result.get(fname)
            c = conf.get(fname, 0.0)
            # Keep value with highest confidence, prefer non-null
            if val is not None and (merged_result.get(fname) is None or c > merged_conf.get(fname, 0)):
                merged_result[fname] = val
                merged_conf[fname] = c
                merged_sources[fname] = sources.get(fname, "landingai_ade")
                merged_evidence[fname] = evidence.get(fname, "")

    # Fill remaining nulls
    for fname in field_names:
        if fname not in merged_result:
            merged_result[fname] = None
            merged_conf[fname] = 0.0
            merged_sources[fname] = "landingai_ade"
            merged_evidence[fname] = ""

    return {
        "result": merged_result,
        "confidence": merged_conf,
        "sources": merged_sources,
        "evidence": merged_evidence,
        "validation": {},
        "schema_fields": field_names,
        "failure_log": failure_log,
        "duration_seconds": round(total_duration, 2),
    }


def _merge_multi_results(results: list[dict], schema: dict) -> dict:
    """
    Merge multi-record results from multiple chunks.
    Collects all records from all chunks, then deduplicates by model_number.
    """
    field_names = schema.get("fields", [])
    field_names = [f["name"] for f in field_names]

    all_records: list[dict] = []
    failure_log: list = []
    total_duration = 0.0

    for res in results:
        if not res:
            continue
        all_records.extend(res.get("records", []))
        failure_log.extend(res.get("failure_log", []))
        total_duration += res.get("duration_seconds", 0)

    # Deduplicate
    all_records = _deduplicate_records(all_records, field_names)
    all_records = _fix_zero_nulls(all_records, field_names)

    # Compute quality score
    from app.services.quality_scorer import compute_quality_for_records
    quality = compute_quality_for_records(all_records, field_names)

    return {
        "records": all_records,
        "total_records": len(all_records),
        "schema_fields": field_names,
        "failure_log": failure_log,
        "duration_seconds": round(total_duration, 2),
        "quality": quality,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_request(api_key: str, environment: str) -> tuple:
    env = _resolve_environment(environment)
    base_url = "https://api.va.eu-west-1.landing.ai" if env == "eu" else "https://api.va.landing.ai"
    url = f"{base_url}/v1/ade/extract"
    headers = {"Authorization": f"Basic {api_key}"}
    return url, headers


# ── Environment resolver ──────────────────────────────────────────────────────

_ENV_MAP = {
    "eu-west-1": "eu", "eu_west_1": "eu", "eu-west": "eu",
    "eu": "eu", "europe": "eu",
    "production": "production", "prod": "production", "us": "production", "": "production",
}


def _resolve_environment(value: str) -> str:
    if not value:
        return "production"
    v = value.lower().strip().rstrip("/")
    if v in _ENV_MAP:
        return _ENV_MAP[v]
    for pattern, env in _ENV_MAP.items():
        if pattern and pattern in v:
            return env
    logger.warning(f"Unrecognised Landing AI environment '{value}', defaulting to 'production'")
    return "production"


# ── Response converters ───────────────────────────────────────────────────────

def _ade_extract_response_to_pipeline_result(response_json: dict, schema: dict) -> dict:
    """Single-record: convert raw Landing AI JSON → pipeline result."""
    fields = schema.get("fields", [])
    field_names = [f["name"] for f in fields]

    extraction = response_json.get("extraction") or {}
    ext_meta = response_json.get("extraction_metadata") or {}
    ade_meta = response_json.get("metadata") or {}

    result: dict = {}
    confidence: dict = {}
    sources: dict = {}
    evidence: dict = {}

    for fname in field_names:
        value = extraction.get(fname)
        meta = ext_meta.get(fname, {}) if isinstance(ext_meta, dict) else {}
        refs = meta.get("references", []) if isinstance(meta, dict) else []
        result[fname] = value
        confidence[fname] = 0.9 if value is not None else 0.0
        sources[fname] = "landingai_ade"
        evidence[fname] = f"LandingAI ADE references: {refs}" if refs else ""

    duration = (ade_meta.get("duration_ms", 0) or 0) / 1000
    failure_log = [
        {"type": "schema_warning", "reason": w.get("msg", "")}
        for w in (ade_meta.get("warnings") or [])
    ]

    return {
        "result": result,
        "confidence": confidence,
        "sources": sources,
        "evidence": evidence,
        "validation": {},
        "schema_fields": field_names,
        "failure_log": failure_log,
        "duration_seconds": round(duration, 2),
    }


def _ade_multi_response_to_pipeline_result(response_json: dict, schema: dict) -> dict:
    """
    Multi-record: Landing AI returns { extraction: { records: [...] } }.
    Convert each record into a pipeline result, then deduplicate by model_number.
    """
    fields = schema.get("fields", [])
    field_names = [f["name"] for f in fields]

    extraction = response_json.get("extraction") or {}
    ext_meta = response_json.get("extraction_metadata") or {}
    ade_meta = response_json.get("metadata") or {}

    raw_records = extraction.get("records") or []

    meta_records_wrap = ext_meta.get("records", {}) if isinstance(ext_meta, dict) else {}
    meta_records = meta_records_wrap.get("value", []) if isinstance(meta_records_wrap, dict) else []

    duration = (ade_meta.get("duration_ms", 0) or 0) / 1000
    failure_log = [
        {"type": "schema_warning", "reason": w.get("msg", "")}
        for w in (ade_meta.get("warnings") or [])
    ]

    records = []
    for i, rec in enumerate(raw_records):
        if not isinstance(rec, dict):
            continue

        result: dict = {}
        confidence: dict = {}
        sources: dict = {}
        evidence: dict = {}

        rec_meta = meta_records[i] if isinstance(meta_records, list) and i < len(meta_records) else {}

        for fname in field_names:
            value = rec.get(fname)
            meta = rec_meta.get(fname, {}) if isinstance(rec_meta, dict) else {}
            refs = meta.get("references", []) if isinstance(meta, dict) else []
            result[fname] = value
            confidence[fname] = 0.9 if value is not None else 0.0
            sources[fname] = "landingai_ade"
            evidence[fname] = f"LandingAI ADE references: {refs}" if refs else ""

        records.append({
            "result": result,
            "confidence": confidence,
            "sources": sources,
            "evidence": evidence,
            "validation": {},
            "schema_fields": field_names,
        })

    # Deduplicate: merge records with the same model_number,
    # keeping the most complete (most non-null fields) version
    records = _deduplicate_records(records, field_names)

    # Post-process: fix zero values that should be null for certain fields
    records = _fix_zero_nulls(records, field_names)

    # Compute quality score across all records
    from app.services.quality_scorer import compute_quality_for_records
    quality = compute_quality_for_records(records, field_names)

    return {
        "records": records,
        "total_records": len(records),
        "schema_fields": field_names,
        "failure_log": failure_log,
        "duration_seconds": round(duration, 2),
        "quality": quality,
    }


def _deduplicate_records(records: list, field_names: list) -> list:
    """
    Merge duplicate records that have the same model_number.
    When two records share a model_number:
    - Keep non-null values from both (prefer the record with more non-null fields)
    - For conflicting non-null values, prefer the one with higher completeness score
    """
    seen: dict = {}  # model_number -> index in deduped list
    deduped: list = []

    for rec in records:
        result = rec.get("result", {})
        model = result.get("model_number")

        if not model:
            deduped.append(rec)
            continue

        if model not in seen:
            seen[model] = len(deduped)
            deduped.append(rec)
        else:
            # Merge into existing record
            existing_idx = seen[model]
            existing = deduped[existing_idx]
            existing_result = existing.get("result", {})
            existing_conf = existing.get("confidence", {})

            merged_result = dict(existing_result)
            merged_conf = dict(existing_conf)

            for fname in field_names:
                existing_val = existing_result.get(fname)
                new_val = result.get(fname)

                # Fill in nulls from the other record
                if existing_val is None and new_val is not None:
                    merged_result[fname] = new_val
                    merged_conf[fname] = rec.get("confidence", {}).get(fname, 0.9)
                # If both non-null and different, keep existing (first occurrence)
                # but fix zero values that should be null
                elif existing_val == 0 and new_val is not None and new_val != 0:
                    merged_result[fname] = new_val
                    merged_conf[fname] = rec.get("confidence", {}).get(fname, 0.9)

            deduped[existing_idx] = {
                **existing,
                "result": merged_result,
                "confidence": merged_conf,
            }

    return deduped


def _fix_zero_nulls(records: list, field_names: list) -> list:
    """
    Fields that should never be 0 (they're either a real measurement or null).
    Replace 0 with null for these fields.
    """
    zero_should_be_null = {
        "cord_length_total_ft", "shipping_weight_lbs", "installed_weight_lbs",
        "length_in", "width_in", "height_in", "min_circuit_amps_mca",
        "max_over_current_protection_mop", "nominal_output_heating_elec_kw",
        "nominal_output_heating_gas_mbh",
    }
    for rec in records:
        result = rec.get("result", {})
        conf = rec.get("confidence", {})
        for fname in field_names:
            if fname in zero_should_be_null and result.get(fname) == 0:
                result[fname] = None
                conf[fname] = 0.0
    return records


def _ade_response_to_parsed_doc(response: Any, file_path: str) -> dict:
    """Convert a Landing AI parse response to our internal parsed_doc format."""
    path = Path(file_path)
    markdown = getattr(response, "markdown", "") or ""
    chunks_raw = getattr(response, "chunks", []) or []
    splits_raw = getattr(response, "splits", []) or []
    grounding_raw = getattr(response, "grounding", {}) or {}
    metadata_raw = getattr(response, "metadata", {}) or {}

    chunks = [c.model_dump() if hasattr(c, "model_dump") else dict(c) for c in chunks_raw]
    splits = [s.model_dump() if hasattr(s, "model_dump") else dict(s) for s in splits_raw]
    grounding = grounding_raw.model_dump() if hasattr(grounding_raw, "model_dump") else (grounding_raw if isinstance(grounding_raw, dict) else {})
    metadata = metadata_raw.model_dump() if hasattr(metadata_raw, "model_dump") else (metadata_raw if isinstance(metadata_raw, dict) else {})

    document_text = _markdown_to_plain(markdown)

    from app.services.parser import _chunks_to_tables, _extract_kv_from_text, _detect_sections
    tables = _chunks_to_tables(chunks)
    kv_pairs = _extract_kv_from_text(document_text)
    sections = _detect_sections(document_text)

    return {
        "markdown": markdown,
        "chunks": chunks,
        "splits": splits,
        "grounding": grounding,
        "document_text": document_text,
        "tables": tables,
        "kv_pairs": kv_pairs,
        "sections": sections,
        "layout_blocks": [],
        "pages": [{"page": i + 1, "text": s.get("markdown", ""), "table_count": 0} for i, s in enumerate(splits)],
        "metadata": {
            "file_name": path.name,
            "page_count": metadata.get("page_count", len(splits) or 1),
            "table_count": len(tables),
            "ocr_used": False,
            "version": metadata.get("version", "ade-landingai"),
            "failed_pages": [],
            "credit_usage": metadata.get("credit_usage", 0),
            "duration_ms": metadata.get("duration_ms", 0),
            "filename": path.name,
            "provider": "landingai",
        },
    }


def _schema_to_ade_format(schema: dict) -> dict:
    """Convert our internal schema to Landing AI JSON Schema format."""
    type_map = {
        "string": "string", "number": "number", "integer": "integer",
        "boolean": "boolean", "date": "string", "currency": "number",
        "email": "string", "phone": "string", "url": "string",
    }

    # Enhanced descriptions for diagram fields — tell LandingAI exactly where to look
    diagram_field_hints = {
        "Length_Diagram": (
            "Extract equipment length/depth in inches ONLY from the [DIAGRAM DIMENSIONS] section "
            "or dimensional drawing annotations. Look for the horizontal measurement labeled as "
            "depth or length (front-to-back dimension). Convert fractions like 13-3/4 to decimal (13.75). "
            "Convert mm to inches if needed (divide by 25.4). Return NULL if not in a diagram."
        ),
        "Width_Diagram": (
            "Extract equipment width in inches ONLY from the [DIAGRAM DIMENSIONS] section "
            "or dimensional drawing annotations. Look for the horizontal measurement labeled as "
            "width (side-to-side dimension). Convert fractions like 36-5/16 to decimal (36.3125). "
            "Convert mm to inches if needed. Return NULL if not in a diagram."
        ),
        "Height_Diagram": (
            "Extract equipment height in inches ONLY from the [DIAGRAM DIMENSIONS] section "
            "or dimensional drawing annotations. Look for the vertical measurement. "
            "Convert fractions like 35-1/32 to decimal (35.03125). "
            "Convert mm to inches if needed. Return NULL if not in a diagram."
        ),
        "Diameter_Diagram": (
            "Extract equipment diameter in inches ONLY from the [DIAGRAM DIMENSIONS] section "
            "or dimensional drawing annotations. Look for circular/round duct measurements. "
            "Convert fractions to decimal. Convert mm to inches if needed. Return NULL if not found."
        ),
    }

    properties: dict = {}
    for field in schema.get("fields", []):
        fname = field["name"]
        ftype = field.get("type", "string")
        # Use enhanced diagram description if available, otherwise use field description
        desc = diagram_field_hints.get(fname) or field.get("description") or field.get("instruction") or fname.replace("_", " ")

        if ftype == "list":
            sub_fields = field.get("fields", [])
            if sub_fields:
                sub_props = {}
                for sf in sub_fields:
                    sfname = sf["name"]
                    sf_desc = diagram_field_hints.get(sfname) or sf.get("description", sfname)
                    sub_props[sfname] = {
                        "type": type_map.get(sf.get("type", "string"), "string"),
                        "description": sf_desc
                    }
                properties[fname] = {"type": "array", "items": {"type": "object", "properties": sub_props}, "description": desc}
            else:
                properties[fname] = {"type": "array", "items": {"type": "string"}, "description": desc}
        elif ftype == "object":
            sub_fields = field.get("fields", [])
            sub_props = {}
            for sf in sub_fields:
                sfname = sf["name"]
                sf_desc = diagram_field_hints.get(sfname) or sf.get("description", sfname)
                sub_props[sfname] = {
                    "type": type_map.get(sf.get("type", "string"), "string"),
                    "description": sf_desc
                }
            properties[fname] = {"type": "object", "properties": sub_props, "description": desc}
        else:
            # Allow null for all scalar fields so Landing AI doesn't raise schema violations
            properties[fname] = {"type": ["null", type_map.get(ftype, "string")], "description": desc}

    return {"type": "object", "properties": properties}


def _markdown_to_plain(md: str) -> str:
    text = re.sub(r"<a id='[^']*'></a>", "", md)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"<!-- PAGE BREAK -->", "\n\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
