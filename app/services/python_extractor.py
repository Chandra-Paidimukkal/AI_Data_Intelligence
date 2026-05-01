"""
python_extractor.py — Heuristic extraction engine.

Handles all field types generically:
  - string / number / integer / boolean / date / currency / email / phone / url
  - list      → returns a Python list (from bullets, numbered items, or comma-sep)
  - object    → returns a Python dict (from KV clusters or sub-table rows)
  - list[object] → returns list[dict] (from table rows or repeated KV sections)

Zero hardcoded domain knowledge. Everything driven by schema metadata.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from app.services.field_retrieval import retrieve_field_context, build_field_context_for_ai
from app.services.schema_utils import get_all_labels


# ── Type-specific regex patterns ──────────────────────────────────────────────

TYPE_PATTERNS: dict[str, str] = {
    "email":    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    "phone":    r"[\+]?[\d\s\-().]{7,20}",
    "url":      r"https?://[^\s]{3,}",
    "currency": r"[$€£¥₹]?\s*\d[\d,]*\.?\d*",
    "date":     r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{2}[/-]\d{2}|[A-Za-z]+ \d{1,2},?\s+\d{4}",
    "number":   r"-?\d[\d,]*\.?\d*",
    "integer":  r"-?\d+",
}

# Bullet/numbered list item detector
_LIST_ITEM_RE = re.compile(
    r"^[\s]*(?:[\*\-•◦▪►]|\d+[\.\)])\s+(.+)$",
    re.MULTILINE
)


# ── Entry point ───────────────────────────────────────────────────────────────

def python_extract(field: dict, parsed_doc: dict) -> dict:
    """
    Heuristic extraction for a single field.
    Returns {value, source, evidence, confidence, method}.
    """
    ftype = field.get("type", "string")

    # Route to type-specific extractor
    if ftype == "list":
        return _extract_list(field, parsed_doc)
    if ftype == "object":
        return _extract_object(field, parsed_doc)

    # Scalar types: try structured retrieval first
    ctx = retrieve_field_context(field, parsed_doc)
    if ctx["value"] is not None and ctx["value"] != "":
        cleaned = _coerce(ctx["value"], ftype)
        if cleaned is not None:
            # If the cleaned value is suspiciously long (spec blob), try to
            # extract a model code from the column header evidence
            if isinstance(cleaned, str) and len(cleaned) > 150:
                rescued = _rescue_from_evidence(ctx.get("evidence", ""), field)
                if rescued:
                    return {
                        "value": rescued,
                        "source": ctx["source"],
                        "evidence": ctx["evidence"],
                        "confidence": 0.6,
                        "method": "heuristic_rescued",
                    }
            else:
                return {
                    "value": cleaned,
                    "source": ctx["source"],
                    "evidence": ctx["evidence"],
                    "confidence": ctx["confidence"],
                    "method": "heuristic_structured",
                }

    # Pattern match fallback
    text = parsed_doc.get("document_text", "")
    pval = _pattern_in_context(field, text)
    if pval is not None:
        return {
            "value": pval,
            "source": "text_pattern",
            "evidence": "Pattern matched in document text",
            "confidence": 0.4,
            "method": "heuristic_pattern",
        }

    return {
        "value": field.get("fallback"),
        "source": "fallback",
        "evidence": "",
        "confidence": 0.0,
        "method": "fallback",
    }


# ── List extractor ────────────────────────────────────────────────────────────

def _extract_list(field: dict, parsed_doc: dict) -> dict:
    """
    Extract a list field. Handles three cases:
      A. list[object] — sub-fields defined → extract from table rows
      B. list[string] — no sub-fields, extract from bullet items or table column
      C. list[string] — comma/newline separated scalar values
    """
    sub_fields = field.get("fields", [])

    # Case A: list of objects — delegate to table-row extraction
    if sub_fields:
        return _extract_list_of_objects(field, sub_fields, parsed_doc)

    # Case B/C: flat list
    labels = get_all_labels(field)
    text = parsed_doc.get("document_text", "")

    # Try table column first
    for table in parsed_doc.get("tables", []):
        headers_lower = [h.lower() for h in table.get("headers", [])]
        for label in labels:
            for i, h in enumerate(headers_lower):
                if label in h or h in label:
                    vals = [
                        str(row[i]).strip()
                        for row in table.get("rows", [])
                        if i < len(row) and str(row[i]).strip()
                    ]
                    if vals:
                        return {
                            "value": vals,
                            "source": "table",
                            "evidence": f"Column '{table['headers'][i]}'",
                            "confidence": 0.9,
                            "method": "heuristic_table_column",
                        }

    # Try bullet/numbered list near label
    text_lower = text.lower()
    for label in labels:
        idx = text_lower.find(label)
        if idx != -1:
            region = text[idx: min(len(text), idx + 2000)]
            items = _extract_bullet_items(region)
            if items:
                return {
                    "value": items,
                    "source": "text",
                    "evidence": region[:300],
                    "confidence": 0.75,
                    "method": "heuristic_bullets",
                }

    return {"value": field.get("fallback"), "source": "fallback",
            "evidence": "", "confidence": 0.0, "method": "fallback"}


def _extract_list_of_objects(field: dict, sub_fields: list, parsed_doc: dict) -> dict:
    """Extract a list of structured objects from table rows."""
    tables = parsed_doc.get("tables", [])
    labels = get_all_labels(field)

    best_table = None
    best_score = 0

    # Pick the table whose headers best match the sub-field names/labels
    for table in tables:
        headers_lower = [h.lower().strip() for h in table.get("headers", [])]
        score = _table_match_score(sub_fields, headers_lower)
        if score > best_score:
            best_score = score
            best_table = table

    if best_table and best_score > 0:
        records = _table_to_records(sub_fields, best_table)
        if records:
            return {
                "value": records,
                "source": "table",
                "evidence": f"Table with {len(records)} rows",
                "confidence": 0.88,
                "method": "heuristic_table_records",
            }

    # Fallback: try KV-cluster extraction (repeated sections)
    records = _kv_cluster_to_records(field, sub_fields, parsed_doc)
    if records:
        return {
            "value": records,
            "source": "kv",
            "evidence": f"KV cluster — {len(records)} records",
            "confidence": 0.7,
            "method": "heuristic_kv_cluster",
        }

    return {"value": field.get("fallback"), "source": "fallback",
            "evidence": "", "confidence": 0.0, "method": "fallback"}


# ── Object extractor ──────────────────────────────────────────────────────────

def _extract_object(field: dict, parsed_doc: dict) -> dict:
    """
    Extract a single object with named sub-fields.
    Each sub-field is extracted independently and assembled into a dict.
    """
    sub_fields = field.get("fields", [])
    if not sub_fields:
        # No sub-schema — fall back to text extraction
        ctx = retrieve_field_context(field, parsed_doc)
        return {
            "value": ctx["value"],
            "source": ctx["source"],
            "evidence": ctx["evidence"],
            "confidence": ctx["confidence"],
            "method": "heuristic_plain",
        }

    result: dict[str, Any] = {}
    total_conf = 0.0
    hit = 0

    for sf in sub_fields:
        sub_result = python_extract(sf, parsed_doc)
        result[sf["name"]] = sub_result["value"]
        total_conf += sub_result["confidence"]
        if sub_result["value"] is not None:
            hit += 1

    avg_conf = (total_conf / len(sub_fields)) if sub_fields else 0.0

    return {
        "value": result,
        "source": "heuristic_subfields",
        "evidence": f"{hit}/{len(sub_fields)} sub-fields extracted",
        "confidence": round(avg_conf, 3),
        "method": "heuristic_object",
    }


# ── Table-row helpers ─────────────────────────────────────────────────────────

def _table_match_score(sub_fields: list, headers_lower: list) -> float:
    """How many sub-field labels appear in this table's headers?"""
    if not headers_lower:
        return 0.0
    matched = 0
    for sf in sub_fields:
        sf_labels = get_all_labels(sf)
        if any(
            any(sl in h or h in sl for h in headers_lower)
            for sl in sf_labels
        ):
            matched += 1
    return matched / len(sub_fields)


def _table_to_records(sub_fields: list, table: dict) -> list[dict]:
    """Convert table rows to list[dict] using sub-field label mapping."""
    headers = table.get("headers", [])
    headers_lower = [h.lower().strip() for h in headers]
    rows = table.get("rows", [])

    # Build mapping: sub_field_name → column index
    col_map: dict[str, int] = {}
    for sf in sub_fields:
        sf_labels = get_all_labels(sf)
        for label in sf_labels:
            for i, h in enumerate(headers_lower):
                if label in h or h in label:
                    col_map[sf["name"]] = i
                    break
            if sf["name"] in col_map:
                break

    records = []
    for row in rows:
        record: dict[str, Any] = {}
        for sf in sub_fields:
            col_idx = col_map.get(sf["name"])
            if col_idx is not None and col_idx < len(row):
                raw = str(row[col_idx]).strip()
                record[sf["name"]] = _coerce(raw, sf.get("type", "string")) if raw else sf.get("fallback")
            else:
                record[sf["name"]] = sf.get("fallback")

        # Accept row if at least 1/3 of fields are populated
        filled = sum(1 for v in record.values() if v is not None and str(v).strip())
        if filled >= max(1, len(sub_fields) // 3):
            records.append(record)

    return records


def _kv_cluster_to_records(field: dict, sub_fields: list, parsed_doc: dict) -> list[dict]:
    """
    For fields like contact_information that appear as named KV sections,
    try to extract a single object (not necessarily repeated rows).
    """
    result: dict[str, Any] = {}
    for sf in sub_fields:
        ctx = retrieve_field_context(sf, parsed_doc)
        if ctx["value"] is not None:
            result[sf["name"]] = ctx["value"]

    if any(v is not None for v in result.values()):
        return [result]
    return []


# ── Bullet/list parsing ───────────────────────────────────────────────────────

def _extract_bullet_items(text: str) -> list[str]:
    """Extract bullet or numbered list items from a text block."""
    items = [m.group(1).strip() for m in _LIST_ITEM_RE.finditer(text)]
    # Fallback: newline-separated non-empty lines
    if not items:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if len(lines) > 1:
            items = lines[1:]  # skip the header line
    return [i for i in items if i]


# ── Type coercion ─────────────────────────────────────────────────────────────

def _coerce(value: Any, ftype: str) -> Optional[Any]:
    """Coerce a raw string value to the target field type."""
    if value is None:
        return None
    val = str(value).strip()
    if not val:
        return None

    pattern = TYPE_PATTERNS.get(ftype)
    if pattern:
        m = re.search(pattern, val)
        if m:
            return m.group(0).strip()
        if ftype in ("email", "url", "phone"):
            return None  # strict — must match pattern

    if ftype == "boolean":
        if val.lower() in ("yes", "true", "1", "y"):
            return True
        if val.lower() in ("no", "false", "0", "n"):
            return False
        return None

    if ftype in ("number", "currency"):
        cleaned = re.sub(r"[^\d.\-]", "", val)
        try:
            return float(cleaned)
        except ValueError:
            return None

    if ftype == "integer":
        cleaned = re.sub(r"[^\d\-]", "", val)
        try:
            return int(cleaned)
        except ValueError:
            return None

    return val  # string / date / other


# ── Pattern match in label context ───────────────────────────────────────────

def _pattern_in_context(field: dict, text: str) -> Optional[str]:
    ftype = field.get("type", "string")
    pattern = TYPE_PATTERNS.get(ftype)
    if not pattern:
        return None
    labels = get_all_labels(field)
    text_lower = text.lower()
    for label in labels:
        idx = text_lower.find(label)
        if idx != -1:
            window = text[max(0, idx - 20): idx + 300]
            m = re.search(pattern, window)
            if m:
                return m.group(0).strip()
    return None


# ── Rescue value from evidence string ────────────────────────────────────────

_MODEL_CODE_RE = re.compile(r"\b([A-Z]{1,6}[\-_]?\d{2,6}[A-Z0-9\-/]*)\b")
_MODELS_LINE_RE = re.compile(r"Models?[:\s]+([A-Z0-9\-/,\s]+)", re.IGNORECASE)


def _rescue_from_evidence(evidence: str, field: dict) -> Optional[str]:
    """
    When a table cell returns a spec blob, try to extract the real value
    from the column header (which is embedded in the evidence string).

    Evidence format: "Column '<header>': <cell_value>"
    """
    fname = field.get("name", "").lower()

    # Extract the column header portion from evidence
    header_match = re.match(r"Column '([^']*)'", evidence)
    header = header_match.group(1) if header_match else evidence

    if "model" in fname:
        # Try "Models: ECG-24/36/48/60/72R" pattern in header
        m = _MODELS_LINE_RE.search(header)
        if m:
            # Return the first model code from the list
            codes = _MODEL_CODE_RE.findall(m.group(1))
            if codes:
                return codes[0]
        # Fallback: any model code in the header
        codes = _MODEL_CODE_RE.findall(header)
        if codes:
            return codes[0]

    return None
