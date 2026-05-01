"""
schema_generator.py — Feature 1: Auto Schema Generator

Analyzes a parsed document and automatically generates a schema
with relevant fields, types, and descriptions.

Uses GPT-4o to understand the document content and suggest
the most useful fields to extract.
"""
from __future__ import annotations

import json
import re
from typing import Any

from loguru import logger


VALID_TYPES = ["string", "number", "integer", "boolean", "date", "currency",
               "email", "phone", "url", "list", "object"]


async def auto_generate_schema(
    parsed_doc: dict,
    api_key: str,
    model: str = "gpt-4o",
    domain_hint: str = "",
    max_fields: int = 30,
) -> dict:
    """
    Analyze a parsed document and generate a schema automatically.

    Returns a schema dict ready to be saved via POST /api/v1/schemas.
    """
    import httpx

    # Build document summary for the prompt
    doc_summary = _build_doc_summary(parsed_doc)

    system_prompt = (
        "You are an expert data extraction schema designer. "
        "Analyze the document content and generate a comprehensive extraction schema. "
        "You MUST respond with valid JSON ONLY — no explanation, no markdown fences. "
        "The schema must follow the exact format specified."
    )

    domain_context = f"\nDomain hint: {domain_hint}" if domain_hint else ""

    user_prompt = f"""Analyze this document and generate an extraction schema.{domain_context}

DOCUMENT CONTENT:
{doc_summary}

Generate a schema that extracts ALL useful structured data from this type of document.
Focus on: model numbers, dimensions, weights, electrical specs, capacities, dates, manufacturer info.

Return ONLY this JSON structure:
{{
  "name": "descriptive_schema_name_snake_case",
  "description": "One sentence describing what this schema extracts",
  "domain": "domain_keyword",
  "fields": [
    {{
      "name": "field_name_snake_case",
      "type": "string|number|integer|boolean|date|currency|list",
      "description": "Clear description of what to extract and where to find it",
      "required": true|false,
      "table_labels": ["Column Header 1", "Column Header 2"],
      "source_labels": ["Label in document", "Alternative label"]
    }}
  ]
}}

Rules:
- field names must be snake_case
- type must be one of: string, number, integer, boolean, date, currency, list
- include table_labels if the field appears in a table column
- include source_labels with the exact text labels used in the document
- required: true only for the most critical identifier fields (model number, etc.)
- generate between 10 and {max_fields} fields
- for multi-model documents, include a "models" field of type "list" with sub-fields
- descriptions should tell the AI exactly where to find the value"""

    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 3000,
            "response_format": {"type": "json_object"},
        }

        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                json=payload, headers=headers,
            )
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"]

        schema = _parse_and_validate_schema(raw, parsed_doc)
        return schema

    except Exception as e:
        logger.error(f"Auto schema generation failed: {e}")
        # Return a basic fallback schema based on document analysis
        return _fallback_schema(parsed_doc, domain_hint)


def _build_doc_summary(parsed_doc: dict) -> str:
    """Build a concise document summary for the prompt."""
    parts = []

    # Document text excerpt
    doc_text = parsed_doc.get("document_text", "")
    if doc_text:
        parts.append(f"=== Document Text (first 1500 chars) ===\n{doc_text[:1500]}")

    # Tables
    tables = parsed_doc.get("tables", [])
    if tables:
        parts.append(f"\n=== Tables Found: {len(tables)} ===")
        for i, table in enumerate(tables[:3]):
            headers = table.get("headers", [])
            rows = table.get("rows", [])
            if headers:
                parts.append(f"Table {i+1} headers: {' | '.join(str(h) for h in headers if h)}")
            if rows:
                # Show first 2 data rows
                for row in rows[:2]:
                    parts.append(f"  Row: {' | '.join(str(c) for c in row[:8])}")

    # KV pairs
    kv_pairs = parsed_doc.get("kv_pairs", [])
    if kv_pairs:
        parts.append(f"\n=== Key-Value Pairs ===")
        for kv in kv_pairs[:20]:
            parts.append(f"  {kv['key']}: {kv['value']}")

    # Sections
    sections = parsed_doc.get("sections", [])
    if sections:
        section_titles = [s["title"] for s in sections[:10]]
        parts.append(f"\n=== Sections: {', '.join(section_titles)}")

    return "\n".join(parts)


def _parse_and_validate_schema(raw: str, parsed_doc: dict) -> dict:
    """Parse and validate the generated schema, fixing common issues."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        schema = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            schema = json.loads(m.group(0))
        else:
            raise ValueError("Could not parse schema JSON")

    # Validate and fix fields
    fields = schema.get("fields", [])
    valid_fields = []

    for field in fields:
        if not isinstance(field, dict):
            continue
        name = field.get("name", "").strip()
        if not name:
            continue

        # Fix name: ensure snake_case
        name = re.sub(r"[^a-zA-Z0-9_]", "_", name).lower().strip("_")
        if not name:
            continue

        # Fix type
        ftype = field.get("type", "string").lower()
        if ftype not in VALID_TYPES:
            ftype = "string"

        clean_field = {
            "name": name,
            "type": ftype,
            "description": field.get("description", ""),
            "required": bool(field.get("required", False)),
        }

        # Add labels if present
        if field.get("table_labels"):
            clean_field["table_labels"] = [str(l) for l in field["table_labels"] if l]
        if field.get("source_labels"):
            clean_field["source_labels"] = [str(l) for l in field["source_labels"] if l]

        # Handle nested fields for list type
        if ftype == "list" and field.get("fields"):
            sub_fields = []
            for sf in field["fields"]:
                if isinstance(sf, dict) and sf.get("name"):
                    sf_name = re.sub(r"[^a-zA-Z0-9_]", "_", sf["name"]).lower()
                    sub_fields.append({
                        "name": sf_name,
                        "type": sf.get("type", "string") if sf.get("type") in VALID_TYPES else "string",
                        "description": sf.get("description", ""),
                    })
            if sub_fields:
                clean_field["fields"] = sub_fields

        valid_fields.append(clean_field)

    if not valid_fields:
        raise ValueError("No valid fields generated")

    schema["fields"] = valid_fields
    schema["name"] = re.sub(r"[^a-zA-Z0-9_]", "_", schema.get("name", "auto_schema")).lower()
    schema["version"] = "1.0"

    return schema


def _fallback_schema(parsed_doc: dict, domain_hint: str) -> dict:
    """
    Generate a basic schema from document analysis without AI.
    Used as fallback when AI generation fails.
    """
    fields = []
    seen = set()

    # Extract field names from KV pairs
    for kv in parsed_doc.get("kv_pairs", [])[:20]:
        key = kv["key"].strip()
        name = re.sub(r"[^a-zA-Z0-9_]", "_", key).lower().strip("_")
        if name and name not in seen and len(name) > 1:
            # Guess type from value
            val = str(kv["value"])
            ftype = "number" if re.match(r"^[\d.,]+$", val.replace(" ", "")) else "string"
            fields.append({
                "name": name,
                "type": ftype,
                "description": f"Value of '{key}' from document",
                "source_labels": [key],
            })
            seen.add(name)

    # Extract field names from table headers
    for table in parsed_doc.get("tables", [])[:3]:
        for header in table.get("headers", []):
            if not header:
                continue
            name = re.sub(r"[^a-zA-Z0-9_]", "_", str(header)).lower().strip("_")
            if name and name not in seen and len(name) > 1:
                fields.append({
                    "name": name,
                    "type": "string",
                    "description": f"Value from '{header}' column",
                    "table_labels": [str(header)],
                })
                seen.add(name)

    # Always include basic fields
    basic = [
        {"name": "manufacturer", "type": "string", "description": "Manufacturer or brand name", "required": True},
        {"name": "model_number", "type": "string", "description": "Model number or code", "required": True},
        {"name": "product_description", "type": "string", "description": "Product description"},
    ]
    for b in basic:
        if b["name"] not in seen:
            fields.insert(0, b)
            seen.add(b["name"])

    doc_name = parsed_doc.get("metadata", {}).get("file_name", "document")
    schema_name = re.sub(r"[^a-zA-Z0-9_]", "_", doc_name.replace(".pdf", "")).lower()

    return {
        "name": f"{schema_name}_schema" if schema_name else "auto_generated_schema",
        "description": f"Auto-generated schema for {doc_name}",
        "version": "1.0",
        "domain": domain_hint or "general",
        "fields": fields[:30],
    }
