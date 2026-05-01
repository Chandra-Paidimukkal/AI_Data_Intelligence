"""
schema_utils.py — Schema normalisation, validation, compatibility.
Fully generic — zero hardcoded domains or field names.
Supports nested field types: string, number, integer, boolean, date,
currency, email, phone, url, list, list[object], object.
"""
from __future__ import annotations
from typing import Any
import re

VALID_FIELD_TYPES = {
    "string", "number", "integer", "boolean", "date",
    "list", "object", "currency", "email", "phone", "url",
}


# ── Schema normalisation ──────────────────────────────────────────────────────

def normalize_schema(raw: dict) -> dict:
    return {
        "name": raw.get("name", "Unnamed Schema"),
        "description": raw.get("description", ""),
        "version": raw.get("version", "1.0"),
        "domain": raw.get("domain", ""),
        "fields": [normalize_field(f) for f in raw.get("fields", [])],
        "record_mode": raw.get("record_mode", False),
        "record_anchor": raw.get("record_anchor", None),
        "domain_keywords": raw.get("domain_keywords", []),
        "reject_domain_mismatch": raw.get("reject_domain_mismatch", False),
    }


def normalize_field(f: dict) -> dict:
    """
    Normalise one field. Recursively normalises sub-fields for object/list types.
    """
    ftype = f.get("type", "string")

    # Recursively normalise nested sub-fields
    sub_fields = []
    if ftype in ("object", "list") and f.get("fields"):
        sub_fields = [normalize_field(sf) for sf in f["fields"]]

    return {
        "name": f.get("name", ""),
        "type": ftype,
        # Human-readable hints (used by AI prompt builder)
        "description": f.get("description", ""),
        "instruction": f.get("instruction", ""),
        # Label sets used by heuristic search
        "source_labels": f.get("source_labels", []),
        "table_labels": f.get("table_labels", []),
        "document_labels": f.get("document_labels", []),
        # Extraction controls
        "source_hint": f.get("source_hint", ""),
        "preferred_sources": f.get("preferred_sources", ["table", "kv", "text"]),
        "required": f.get("required", False),
        "fallback": f.get("fallback", None),
        "null_values": f.get("null_values", []),
        # Nested schema (for object / list[object])
        "fields": sub_fields,
        # Post-processing
        "validation_rules": f.get("validation_rules", []),
        "normalization_rules": f.get("normalization_rules", []),
        "allowed_values": f.get("allowed_values", []),
        # Record/table controls
        "record_anchor": f.get("record_anchor", False),
        "primary_identifier": f.get("primary_identifier", False),
        "confidence_threshold": f.get("confidence_threshold", 0.5),
        "domain_keywords": f.get("domain_keywords", []),
        "retry_config": f.get("retry_config", {"max_retries": 2, "strategy": "context_expand"}),
        "engine_config": f.get("engine_config", {"prefer": "auto"}),
    }


# ── Schema validation ─────────────────────────────────────────────────────────

def validate_schema_definition(schema: dict) -> dict:
    errors: list[str] = []
    if not schema.get("name"):
        errors.append("Schema must have a name.")
    fields = schema.get("fields", [])
    if not fields:
        errors.append("Schema must have at least one field.")
    names: set[str] = set()
    for i, f in enumerate(fields):
        if not f.get("name"):
            errors.append(f"Field #{i+1} missing 'name'.")
        elif f["name"] in names:
            errors.append(f"Duplicate field name: '{f['name']}'.")
        else:
            names.add(f["name"])
        ftype = f.get("type", "string")
        if ftype not in VALID_FIELD_TYPES:
            errors.append(
                f"Field '{f.get('name')}' has invalid type '{ftype}'. "
                f"Allowed: {', '.join(sorted(VALID_FIELD_TYPES))}"
            )
    return {"valid": len(errors) == 0, "errors": errors}


# ── Domain compatibility ──────────────────────────────────────────────────────

def check_schema_compatibility(schema: dict, parsed_doc: dict) -> dict:
    if not schema.get("reject_domain_mismatch"):
        return {"compatible": True, "score": 1.0, "reason": "Domain check disabled"}
    keywords = schema.get("domain_keywords", [])
    if not keywords:
        return {"compatible": True, "score": 1.0, "reason": "No domain keywords defined"}
    doc_text = (parsed_doc.get("document_text", "") or "").lower()
    matched = [kw for kw in keywords if kw.lower() in doc_text]
    score = len(matched) / len(keywords)
    if score < 0.2:
        return {
            "compatible": False, "score": score,
            "reason": f"Document matches {score:.0%} of domain keywords."
        }
    return {"compatible": True, "score": score, "reason": f"Domain match: {score:.0%}"}


# ── Label helpers ─────────────────────────────────────────────────────────────

def get_all_labels(field: dict) -> list[str]:
    """All searchable labels for a field — name + every label set."""
    labels: list[str] = [field["name"]]
    labels.extend(field.get("source_labels", []))
    labels.extend(field.get("table_labels", []))
    labels.extend(field.get("document_labels", []))
    return [l.lower().strip() for l in labels if l]


def get_table_labels(field: dict) -> list[str]:
    tl = field.get("table_labels", [])
    return [l.lower().strip() for l in (tl or get_all_labels(field))]


# ── Normalisation & validation ────────────────────────────────────────────────

def apply_normalization(value: Any, rules: list) -> Any:
    if value is None:
        return value
    val = str(value)
    for rule in rules:
        if rule == "uppercase":
            val = val.upper()
        elif rule == "lowercase":
            val = val.lower()
        elif rule == "strip":
            val = val.strip()
        elif rule == "remove_currency":
            val = re.sub(r"[$€£¥₹,]", "", val).strip()
        elif rule == "digits_only":
            val = re.sub(r"\D", "", val)
        elif rule == "normalize_date":
            val = _normalize_date(val)
        elif rule == "title_case":
            val = val.title()
    return val


def _normalize_date(val: str) -> str:
    val = val.strip()
    patterns = [
        (r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", r"\3-\2-\1"),
        (r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", r"\1-\2-\3"),
    ]
    for pat, repl in patterns:
        if re.match(pat, val):
            return re.sub(pat, repl, val)
    return val


def apply_validation(value: Any, rules: list, allowed_values: list, field_type: str) -> list[str]:
    errors: list[str] = []
    if value is None or value == "":
        return errors
    val = str(value)
    for rule in rules:
        if rule == "not_empty" and not val.strip():
            errors.append("Value must not be empty")
        elif rule == "numeric" and not re.match(r"^-?\d+(\.\d+)?$", val.replace(",", "")):
            errors.append(f"Value '{val}' is not numeric")
        elif rule == "positive":
            try:
                if float(re.sub(r"[^\d.\-]", "", val) or 0) < 0:
                    errors.append("Value must be positive")
            except ValueError:
                pass
        elif rule.startswith("min_length:"):
            if len(val) < int(rule.split(":")[1]):
                errors.append(f"Value too short (min {rule.split(':')[1]})")
        elif rule.startswith("max_length:"):
            if len(val) > int(rule.split(":")[1]):
                errors.append(f"Value too long (max {rule.split(':')[1]})")
        elif rule == "email" and not re.match(r"[^@]+@[^@]+\.[^@]+", val):
            errors.append("Invalid email format")
    if allowed_values and val not in allowed_values:
        errors.append(f"Value '{val}' not in allowed values: {allowed_values}")
    return errors


# ── JSON schema renderer (used in AI prompts) ─────────────────────────────────

def field_to_json_schema(field: dict) -> dict:
    """
    Convert a normalised field to a JSON Schema fragment.
    Used to build precise AI prompts.
    """
    ftype = field["type"]

    if ftype == "object":
        props = {sf["name"]: field_to_json_schema(sf) for sf in field.get("fields", [])}
        return {"type": "object", "properties": props,
                "description": field.get("description", "")}

    if ftype == "list":
        sub_fields = field.get("fields", [])
        if sub_fields:
            # list of objects
            props = {sf["name"]: field_to_json_schema(sf) for sf in sub_fields}
            return {
                "type": "array",
                "items": {"type": "object", "properties": props},
                "description": field.get("description", "")
            }
        else:
            return {"type": "array", "items": {"type": "string"},
                    "description": field.get("description", "")}

    type_map = {
        "string": "string", "number": "number", "integer": "integer",
        "boolean": "boolean", "date": "string", "currency": "number",
        "email": "string", "phone": "string", "url": "string",
    }
    return {
        "type": type_map.get(ftype, "string"),
        "description": field.get("description", ""),
    }
