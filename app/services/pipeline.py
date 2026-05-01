"""
pipeline.py — Extraction orchestration engine.

Orchestrates:
  1. Heuristic extraction (all field types incl. list, object, list[object])
  2. AI fallback for low-confidence / missing fields
  3. Post-processing: normalisation, validation, null-checks
  4. Record-mode assembly for table-anchored schemas

Fully generic — zero hardcoded domains or field names.
Schema drives all behavior.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any
from loguru import logger

from app.services.schema_utils import (
    normalize_schema, check_schema_compatibility,
    apply_normalization, apply_validation,
    get_all_labels, field_to_json_schema,
)
from app.services.python_extractor import python_extract
from app.services.field_retrieval import build_field_context_for_ai
from app.services.llm_router import LLMRouter


# ── Entry point ───────────────────────────────────────────────────────────────

async def run_extraction(
    parsed_doc: dict,
    schema: dict,
    provider_config: dict,
) -> dict:
    """
    Full extraction pipeline.
    Returns {result, confidence, sources, evidence, validation,
             schema_fields, failure_log, duration_seconds}.
    """
    start = time.time()
    schema = normalize_schema(schema)
    fields = schema["fields"]

    # Domain compatibility check
    compat = check_schema_compatibility(schema, parsed_doc)
    if not compat["compatible"]:
        return _empty_result(fields, start, {"domain_mismatch": compat["reason"]})

    llm = LLMRouter(
        provider=provider_config.get("provider", "none"),
        api_key=provider_config.get("api_key", ""),
        model=provider_config.get("model", ""),
        base_url=provider_config.get("base_url", ""),
    )

    result: dict[str, Any] = {}
    confidence: dict[str, float] = {}
    sources: dict[str, str] = {}
    evidence: dict[str, str] = {}
    validation_errors: dict[str, list] = {}
    failure_log: list[dict] = []
    needs_ai: list[dict] = []

    # ── Step 1: Heuristic extraction ─────────────────────────────────────────
    for field in fields:
        fname = field["name"]
        extr = python_extract(field, parsed_doc)

        value = extr["value"]
        conf = extr["confidence"]

        # Normalisation
        if value is not None and field.get("normalization_rules"):
            value = _apply_norm_deep(value, field)

        # Null-value filtering — also treat common placeholder sentinels as null
        if _is_null_value(value, field) or _is_placeholder(value):
            value = field.get("fallback")
            conf = 0.0

        # Post-process: if a string value looks like a spec blob (very long),
        # try to extract a cleaner value using type-specific patterns
        if isinstance(value, str) and len(value) > 200:
            cleaned = _extract_from_blob(value, field)
            if cleaned is not None:
                value = cleaned
                conf = min(conf, 0.5)  # lower confidence since we had to rescue it
            else:
                value = field.get("fallback")
                conf = 0.0

        result[fname] = value
        confidence[fname] = round(conf, 3)
        sources[fname] = extr["source"]
        evidence[fname] = extr["evidence"]

        if conf < field.get("confidence_threshold", 0.5) or _is_empty(value):
            needs_ai.append(field)

    # ── Step 2: AI extraction for low-confidence fields ──────────────────────
    if needs_ai and llm.is_available():
        ai_results = await _ai_extract(needs_ai, parsed_doc, llm, schema)
        for field in needs_ai:
            fname = field["name"]
            ai_val = ai_results.get(fname)
            if ai_val is not None and not _is_empty(ai_val):
                if field.get("normalization_rules"):
                    ai_val = _apply_norm_deep(ai_val, field)
                result[fname] = ai_val
                confidence[fname] = round(
                    min(confidence.get(fname, 0.0) + 0.3, 0.88), 3
                )
                sources[fname] = f"ai:{llm.provider}"
                evidence[fname] = f"AI ({llm.provider}/{llm.model}) extraction"

    # ── Step 3: Required-field fallback ──────────────────────────────────────
    for field in fields:
        fname = field["name"]
        if field.get("required") and _is_empty(result.get(fname)):
            fb = field.get("fallback")
            if fb is not None:
                result[fname] = fb
                sources[fname] = "fallback"
                confidence[fname] = 0.1
            else:
                failure_log.append({
                    "field": fname,
                    "type": "required_missing",
                    "reason": "Required field could not be extracted",
                })

    # ── Step 4: Validation ────────────────────────────────────────────────────
    for field in fields:
        fname = field["name"]
        val = result.get(fname)
        # Validate scalars only (lists/objects validated by AI schema)
        if not isinstance(val, (list, dict)):
            errs = apply_validation(
                val,
                field.get("validation_rules", []),
                field.get("allowed_values", []),
                field.get("type", "string"),
            )
            if errs:
                validation_errors[fname] = errs

    # ── Step 5: Quality scoring ───────────────────────────────────────────────
    from app.services.quality_scorer import compute_quality_score
    quality = compute_quality_score(
        result=result,
        confidence=confidence,
        sources=sources,
        schema_fields=[f["name"] for f in fields],
        validation_errors=validation_errors,
        failure_log=failure_log,
    )

    return {
        "result": result,
        "confidence": confidence,
        "sources": sources,
        "evidence": evidence,
        "validation": validation_errors,
        "schema_fields": [f["name"] for f in fields],
        "failure_log": failure_log,
        "duration_seconds": round(time.time() - start, 2),
        "quality": quality,
    }


# ── AI extraction ─────────────────────────────────────────────────────────────

async def _ai_extract(
    fields: list, parsed_doc: dict, llm: LLMRouter, schema: dict
) -> dict:
    """
    Build a precise AI prompt using JSON Schema for the target fields,
    then parse the structured response.
    """
    # Build output JSON Schema
    output_schema: dict[str, Any] = {
        "type": "object",
        "properties": {},
    }
    for field in fields:
        output_schema["properties"][field["name"]] = field_to_json_schema(field)

    # Build per-field context (deduplicated)
    context_parts: list[str] = []
    seen_ctx: set[str] = set()
    for field in fields:
        ctx = build_field_context_for_ai(field, parsed_doc)
        if ctx not in seen_ctx:
            context_parts.append(ctx)
            seen_ctx.add(ctx)

    combined_context = "\n---\n".join(context_parts)

    # Build field instructions
    field_instructions = []
    for f in fields:
        desc = f.get("description") or f.get("instruction") or ""
        ftype = f.get("type", "string")
        sub_hint = ""
        if ftype in ("list", "object") and f.get("fields"):
            sub_names = [sf["name"] for sf in f["fields"]]
            sub_hint = f" (sub-fields: {', '.join(sub_names)})"
        field_instructions.append(
            f'  "{f["name"]}" ({ftype}{sub_hint}): {desc}'
        )

    system_prompt = (
        "You are a precise document data extraction assistant for commercial foodservice equipment spec sheets. "
        "You MUST respond with valid JSON ONLY — no explanation, no markdown fences, no extra text. "
        "Extract exactly what is written in the document. "
        "Use null for any value that is not present or cannot be determined. "
        "For list fields, return a JSON array. "
        "For object fields, return a JSON object. "
        "Do NOT invent or hallucinate values. "
        "IMPORTANT: 'X-Z' is a placeholder meaning the value is unknown — treat it as null. "
        "For model_number: extract only the model code (e.g. 'ECG-48R'), NOT the full spec text. "
        "For equipment_type_l2: classify based on what the equipment does (e.g. 'Grills' for griddles). "
        "For heat_type: look for 'electric', 'gas', 'propane', etc. in the document. "
        "For manufacturer: look for the brand/company name, not a placeholder."
    )

    user_prompt = f"""Extract the following fields from the document context below.

DOCUMENT CONTEXT:
{combined_context[:4000]}

FIELDS TO EXTRACT:
{chr(10).join(field_instructions)}

RULES:
- Return null for any field you cannot find with confidence.
- 'X-Z' means unknown/not-applicable — always return null instead.
- model_number should be a short alphanumeric code like 'ECG-48R', not a paragraph of text.
- For classification fields (equipment_type_l2, heat_type), infer from the document content.

REQUIRED OUTPUT FORMAT (JSON Schema):
{json.dumps(output_schema, indent=2)}

Return ONLY a JSON object matching the schema above. Use null for missing values."""

    try:
        raw = await llm.complete(user_prompt, system_prompt)
        return _parse_json_response(raw)
    except Exception as e:
        logger.error(f"AI extraction error: {e}")
        return {}


def _parse_json_response(text: str) -> dict:
    """Robustly parse AI JSON response, stripping markdown fences if present."""
    if not text:
        return {}
    # Strip code fences
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        # Try to find a JSON object anywhere in the response
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _empty_result(fields: list, start: float, validation: dict) -> dict:
    return {
        "result": {},
        "confidence": {},
        "sources": {},
        "evidence": {},
        "validation": validation,
        "schema_fields": [f["name"] for f in fields],
        "failure_log": [{"type": "domain_mismatch", **validation}],
        "duration_seconds": round(time.time() - start, 2),
    }


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (list, dict)) and not value:
        return True
    return False


def _is_null_value(value: Any, field: dict) -> bool:
    if value is None:
        return False
    null_values = field.get("null_values", [])
    return str(value).strip() in [str(n) for n in null_values]


# Common placeholder / sentinel values that indicate "not found" in spec sheets
# Note: X-Z, X-Y, Y-Z are also valid electrical phase labels in tables,
# so we only treat them as placeholders when they appear as standalone field values
_PLACEHOLDER_SENTINELS = {"n/a", "na", "tbd", "tba", "-", "--", "---", "?", "unknown"}
# These are placeholders ONLY when they are the sole value (not part of a table with real data)
_CONDITIONAL_PLACEHOLDERS = {"x-z", "x-y", "y-z"}


def _is_placeholder(value: Any) -> bool:
    """Return True if the value is a known placeholder sentinel."""
    if value is None:
        return False
    normalized = str(value).strip().lower()
    return normalized in _PLACEHOLDER_SENTINELS or normalized in _CONDITIONAL_PLACEHOLDERS


# Patterns used to rescue a real value from a spec-blob cell
_MODEL_NUMBER_RE = re.compile(
    r"\b([A-Z]{1,6}[\-_]?\d{2,6}[A-Z0-9\-/]*)\b"
)
_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _extract_from_blob(blob: str, field: dict) -> Any:
    """
    When a heuristic returns a very long spec-blob string instead of a clean
    value, try to rescue the correct value using field-type patterns and the
    field name as a hint.

    Returns the extracted value, or None if nothing useful found.
    """
    fname = field.get("name", "").lower()
    ftype = field.get("type", "string")

    # Model number: look for alphanumeric model codes in the blob
    if "model" in fname:
        # Prefer matches from the column header (before the first newline)
        header_line = blob.split("\n")[0]
        m = _MODEL_NUMBER_RE.search(header_line)
        if m:
            return m.group(1)
        # Fall back to scanning the whole blob
        matches = _MODEL_NUMBER_RE.findall(blob)
        if matches:
            return matches[0]

    # Numeric fields: grab the first number
    if ftype in ("number", "integer", "currency") or any(
        kw in fname for kw in ("weight", "amps", "kw", "mbh", "length", "width",
                                "height", "capacity", "cord", "volt")
    ):
        m = _NUMBER_RE.search(blob)
        if m:
            raw = m.group(0).replace(",", "")
            try:
                return int(raw) if ftype == "integer" else float(raw)
            except ValueError:
                return raw

    # For short expected values (< 50 chars), try the first line
    first_line = blob.split("\n")[0].strip()
    if first_line and len(first_line) < 80:
        return first_line

    return None


def _apply_norm_deep(value: Any, field: dict) -> Any:
    """Apply normalization_rules recursively for objects/lists."""
    rules = field.get("normalization_rules", [])
    if not rules:
        return value
    if isinstance(value, list):
        return [_apply_norm_deep(item, field) for item in value]
    if isinstance(value, dict):
        return {k: apply_normalization(v, rules) if isinstance(v, str) else v
                for k, v in value.items()}
    return apply_normalization(value, rules)
