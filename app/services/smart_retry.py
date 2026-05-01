"""
smart_retry.py — Feature 3: Smart Retry for Low-Confidence Fields

When a field has low confidence or is null after initial extraction,
retry with a highly targeted prompt focused only on that specific field.

Strategy:
  1. Identify fields below confidence threshold
  2. For each field, build a focused context (most relevant document sections)
  3. Send a targeted single-field prompt to the LLM
  4. Merge improved results back into the original extraction
"""
from __future__ import annotations

import json
import re
from typing import Any

from loguru import logger

from app.services.llm_router import LLMRouter
from app.services.field_retrieval import build_field_context_for_ai


async def smart_retry_low_confidence(
    result: dict,
    confidence: dict,
    sources: dict,
    evidence: dict,
    schema_fields_def: list[dict],
    parsed_doc: dict,
    llm: LLMRouter,
    threshold: float = 0.5,
    max_retries: int = 5,
) -> dict:
    """
    Retry extraction for fields below the confidence threshold.

    Returns updated {result, confidence, sources, evidence, retry_log}.
    """
    if not llm.is_available():
        return {
            "result": result,
            "confidence": confidence,
            "sources": sources,
            "evidence": evidence,
            "retry_log": [],
        }

    # Find fields that need retry
    fields_to_retry = []
    for field_def in schema_fields_def:
        fname = field_def["name"]
        ftype = field_def.get("type", "string")

        # Skip list/object fields — too complex for single-field retry
        if ftype in ("list", "object"):
            continue

        current_conf = confidence.get(fname, 0)
        current_val = result.get(fname)

        if current_conf < threshold or current_val is None:
            fields_to_retry.append(field_def)

    if not fields_to_retry:
        return {
            "result": result,
            "confidence": confidence,
            "sources": sources,
            "evidence": evidence,
            "retry_log": [],
        }

    # Cap retries
    fields_to_retry = fields_to_retry[:max_retries]
    logger.info(f"Smart retry: {len(fields_to_retry)} fields below threshold {threshold}")

    retry_log = []
    updated_result = dict(result)
    updated_confidence = dict(confidence)
    updated_sources = dict(sources)
    updated_evidence = dict(evidence)

    for field_def in fields_to_retry:
        fname = field_def["name"]
        old_val = result.get(fname)
        old_conf = confidence.get(fname, 0)

        try:
            new_val, new_conf, new_evidence = await _retry_single_field(
                field_def, parsed_doc, llm
            )

            if new_val is not None and new_conf > old_conf:
                updated_result[fname] = new_val
                updated_confidence[fname] = new_conf
                updated_sources[fname] = f"smart_retry:{llm.provider}"
                updated_evidence[fname] = new_evidence

                retry_log.append({
                    "field": fname,
                    "status": "improved",
                    "old_value": old_val,
                    "new_value": new_val,
                    "old_confidence": round(old_conf, 3),
                    "new_confidence": round(new_conf, 3),
                })
                logger.info(f"Smart retry improved '{fname}': {old_val!r} → {new_val!r} ({old_conf:.2f} → {new_conf:.2f})")
            else:
                retry_log.append({
                    "field": fname,
                    "status": "no_improvement",
                    "old_value": old_val,
                    "new_value": new_val,
                    "old_confidence": round(old_conf, 3),
                    "new_confidence": round(new_conf, 3),
                })

        except Exception as e:
            logger.warning(f"Smart retry failed for '{fname}': {e}")
            retry_log.append({
                "field": fname,
                "status": "error",
                "error": str(e),
            })

    return {
        "result": updated_result,
        "confidence": updated_confidence,
        "sources": updated_sources,
        "evidence": updated_evidence,
        "retry_log": retry_log,
    }


async def _retry_single_field(
    field_def: dict,
    parsed_doc: dict,
    llm: LLMRouter,
) -> tuple[Any, float, str]:
    """
    Retry extraction for a single field with a highly targeted prompt.
    Returns (value, confidence, evidence).
    """
    fname = field_def["name"]
    ftype = field_def.get("type", "string")
    description = field_def.get("description") or field_def.get("instruction") or fname
    allowed_values = field_def.get("allowed_values", [])

    # Build focused context for this specific field
    context = build_field_context_for_ai(field_def, parsed_doc)

    # Also include document text window around likely locations
    doc_text = parsed_doc.get("document_text", "")
    doc_excerpt = _find_relevant_excerpt(fname, description, doc_text, window=800)

    full_context = f"{context}\n\n--- Document Text ---\n{doc_excerpt}" if doc_excerpt else context

    # Build type-specific instruction
    type_hint = _get_type_hint(ftype, fname)
    allowed_hint = f"\nAllowed values: {allowed_values}" if allowed_values else ""

    system_prompt = (
        "You are a precise data extraction specialist. "
        "Your task is to find ONE specific field from a document. "
        "Respond with ONLY a JSON object: {\"value\": <extracted_value>, \"confidence\": <0.0-1.0>}. "
        "Use null for value if not found. "
        "Confidence: 0.9=very sure, 0.7=fairly sure, 0.5=uncertain, 0.0=not found."
    )

    user_prompt = f"""Find the value of this field from the document:

FIELD: {fname}
DESCRIPTION: {description}
TYPE: {ftype}{allowed_hint}
{type_hint}

DOCUMENT CONTEXT:
{full_context[:3000]}

Respond ONLY with JSON: {{"value": <value_or_null>, "confidence": <0.0-1.0>}}"""

    raw = await llm.complete(user_prompt, system_prompt)

    if not raw:
        return None, 0.0, ""

    # Parse response
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        parsed = json.loads(raw)
        value = parsed.get("value")
        conf = float(parsed.get("confidence", 0.5))

        # Coerce type
        if value is not None:
            value = _coerce_value(value, ftype)

        evidence = f"Smart retry via {llm.provider}: confidence={conf:.2f}"
        return value, conf, evidence

    except Exception:
        # Try to extract value from raw text
        m = re.search(r'"value"\s*:\s*([^,}]+)', raw)
        if m:
            val_str = m.group(1).strip().strip('"')
            if val_str.lower() in ("null", "none", ""):
                return None, 0.0, ""
            return val_str, 0.5, f"Smart retry (parsed): {val_str}"
        return None, 0.0, ""


def _find_relevant_excerpt(fname: str, description: str, doc_text: str, window: int = 800) -> str:
    """Find the most relevant section of document text for a field."""
    if not doc_text:
        return ""

    text_lower = doc_text.lower()
    fname_lower = fname.lower().replace("_", " ")

    # Search terms: field name words + description keywords
    search_terms = fname_lower.split() + [
        w for w in description.lower().split()
        if len(w) > 4 and w not in ("from", "the", "this", "that", "with", "value", "field")
    ]

    best_idx = -1
    best_score = 0

    for term in search_terms[:5]:
        idx = text_lower.find(term)
        if idx != -1:
            # Score by how many other terms are nearby
            nearby = doc_text[max(0, idx-200):idx+200].lower()
            score = sum(1 for t in search_terms if t in nearby)
            if score > best_score:
                best_score = score
                best_idx = idx

    if best_idx == -1:
        return doc_text[:window]

    start = max(0, best_idx - window // 2)
    end = min(len(doc_text), best_idx + window // 2)
    return doc_text[start:end]


def _get_type_hint(ftype: str, fname: str) -> str:
    """Return type-specific extraction hints."""
    hints = {
        "number": "Extract numeric value only. No units. Convert fractions to decimals (e.g. 43-5/16 → 43.3125).",
        "integer": "Extract whole number only. No units or text.",
        "date": "Extract date in YYYY format if year only, or YYYY-MM-DD if full date.",
        "boolean": "Return true or false only.",
        "currency": "Extract numeric value only, no currency symbols.",
        "email": "Extract email address only.",
        "phone": "Extract phone number only.",
        "url": "Extract URL only.",
    }
    base = hints.get(ftype, "")

    # Field-specific hints
    fname_lower = fname.lower()
    if "weight" in fname_lower:
        base += " Weight is in pounds (lbs). Convert kg if needed (1 kg = 2.205 lbs)."
    elif "height" in fname_lower or "width" in fname_lower or "length" in fname_lower or "depth" in fname_lower:
        base += " Dimension is in inches. Convert mm if needed (1 inch = 25.4 mm). Convert fractions like 43-5/16 to decimal."
    elif "kw" in fname_lower or "kilowatt" in fname_lower:
        base += " Power in kilowatts (kW). Extract numeric value only."
    elif "btu" in fname_lower:
        base += " Capacity in BTU/hr. Convert MBH to BTU/hr (1 MBH = 1000 BTU/hr) if needed."
    elif "amp" in fname_lower or "mca" in fname_lower or "mop" in fname_lower:
        base += " Electrical amperage value. Extract numeric value only."
    elif "year" in fname_lower:
        base += " Extract 4-digit year only (e.g. 2024)."

    return base


def _coerce_value(value: Any, ftype: str) -> Any:
    """Coerce extracted value to the correct Python type."""
    if value is None:
        return None
    if ftype in ("number", "currency"):
        try:
            return float(str(value).replace(",", ""))
        except (ValueError, TypeError):
            return value
    if ftype == "integer":
        try:
            return int(str(value).replace(",", "").split(".")[0])
        except (ValueError, TypeError):
            return value
    if ftype == "boolean":
        if isinstance(value, bool):
            return value
        return str(value).lower() in ("true", "yes", "1")
    return value
