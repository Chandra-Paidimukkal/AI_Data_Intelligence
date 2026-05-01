"""
field_retrieval.py — Heuristic context retrieval from ADE-parsed documents.

Search priority (configurable per-field via preferred_sources):
  1. table   — structured table rows/columns
  2. kv      — key-value pairs
  3. chunk   — grounded ADE text chunks
  4. text    — raw document text window

For list/object fields, returns structured context strings
that the AI extractor can parse.
"""
from __future__ import annotations

import re
from typing import Optional

from app.services.schema_utils import get_all_labels, get_table_labels


# ── Public interface ──────────────────────────────────────────────────────────

def retrieve_field_context(field: dict, parsed_doc: dict, window: int = 400) -> dict:
    """
    Returns {value, source, evidence, confidence}.
    value may be a string, list, or dict for complex types.
    """
    preferred = field.get("preferred_sources", ["table", "kv", "text"])
    labels = get_all_labels(field)
    t_labels = get_table_labels(field)
    ftype = field.get("type", "string")

    result = None

    if "table" in preferred:
        result = _from_tables(field, parsed_doc.get("tables", []), t_labels, ftype)

    if result is None and "kv" in preferred:
        result = _from_kv(labels, parsed_doc.get("kv_pairs", []))

    if result is None and "text" in preferred:
        result = _from_chunks(labels, parsed_doc.get("chunks", []), ftype)

    if result is None and "text" in preferred:
        result = _from_text(labels, parsed_doc.get("document_text", ""), window, ftype)

    if result is None:
        return {"value": field.get("fallback"), "source": "fallback",
                "evidence": "", "confidence": 0.0}
    return result


def build_field_context_for_ai(field: dict, parsed_doc: dict) -> str:
    """
    Build a rich context string for one field, used in AI extraction prompts.
    Pulls the most relevant table rows, KV pairs, and text chunks.
    """
    labels = get_all_labels(field)
    t_labels = get_table_labels(field)
    ftype = field.get("type", "string")
    snippets: list[str] = []

    # 1. Matching table(s)
    for table in parsed_doc.get("tables", []):
        headers = table.get("headers", [])
        headers_lower = [h.lower() for h in headers]
        col_indices = _matching_col_indices(t_labels, headers_lower)

        if col_indices:
            # For list/object: include all rows
            header_str = " | ".join(headers)
            snippets.append(f"[TABLE] {header_str}")
            for row in table.get("rows", [])[:30]:
                snippets.append(" | ".join(
                    str(row[i]) if i < len(row) else "" for i in range(len(headers))
                ))
        elif ftype in ("list", "object"):
            # For list/object fields, include ALL tables as context — AI decides
            header_str = " | ".join(h for h in headers if h)  # skip empty headers
            if header_str:
                snippets.append(f"[TABLE] {header_str}")
                for row in table.get("rows", [])[:20]:
                    snippets.append(" | ".join(str(c) for c in row))

    # 2. Matching KV pairs
    for pair in parsed_doc.get("kv_pairs", []):
        key_lower = pair["key"].lower()
        if any(l in key_lower or key_lower in l for l in labels):
            snippets.append(f"{pair['key']}: {pair['value']}")

    # 3. Matching text/title chunks (ADE grounded)
    for chunk in parsed_doc.get("chunks", []):
        if chunk.get("type") not in ("text", "title"):
            continue
        plain = _strip_markup(chunk.get("markdown", ""))
        plain_lower = plain.lower()
        for label in labels[:3]:
            if label in plain_lower:
                idx = plain_lower.find(label)
                snippets.append(plain[max(0, idx - 30): idx + 500])
                break

    # 4. Fallback: first 600 chars of document
    if not snippets:
        snippets.append(parsed_doc.get("document_text", "")[:600])

    return "\n".join(snippets[:15])  # cap to avoid prompt explosion


# ── Table search ──────────────────────────────────────────────────────────────

def _from_tables(field: dict, tables: list, t_labels: list, ftype: str) -> Optional[dict]:
    for table in tables:
        headers = table.get("headers", [])
        headers_lower = [h.lower().strip() for h in headers]
        col_indices = _matching_col_indices(t_labels, headers_lower)

        if not col_indices:
            continue

        rows = table.get("rows", [])
        primary_col = col_indices[0]

        if ftype == "list" or field.get("record_anchor"):
            # Return every row as a list item
            values = []
            for row in rows:
                if primary_col < len(row) and str(row[primary_col]).strip():
                    values.append(str(row[primary_col]).strip())
            if values:
                return {
                    "value": values,
                    "source": "table",
                    "evidence": f"Column '{headers[primary_col]}' — {len(values)} rows",
                    "confidence": 0.9
                }
        else:
            # Return first non-empty value
            for row in rows:
                if primary_col < len(row) and str(row[primary_col]).strip():
                    cell_val = str(row[primary_col]).strip()
                    col_header = headers[primary_col] if primary_col < len(headers) else ""
                    return {
                        "value": cell_val,
                        "source": "table",
                        "evidence": f"Column '{col_header}': {cell_val}",
                        "confidence": 0.9
                    }
    return None


def _matching_col_indices(labels: list, headers_lower: list) -> list[int]:
    """Return column indices whose headers match any label.
    
    Empty-string headers are only matched if the label is also empty,
    preventing accidental matches against placeholder columns.
    """
    matched: list[int] = []
    for label in labels:
        if not label:
            continue  # skip empty labels
        for i, h in enumerate(headers_lower):
            if not h:
                continue  # skip empty-header columns — they're usually placeholders
            if label in h or h in label:
                if i not in matched:
                    matched.append(i)
    return matched


# ── KV search ─────────────────────────────────────────────────────────────────

def _from_kv(labels: list, kv_pairs: list) -> Optional[dict]:
    for pair in kv_pairs:
        key = pair["key"].lower().strip()
        for label in labels:
            if label in key or key in label:
                val = pair["value"].strip()
                if val:
                    return {
                        "value": val,
                        "source": "kv",
                        "evidence": f"{pair['key']}: {pair['value']}",
                        "confidence": 0.75
                    }
    return None


# ── Chunk search (ADE grounded) ───────────────────────────────────────────────

def _from_chunks(labels: list, chunks: list, ftype: str) -> Optional[dict]:
    for chunk in chunks:
        if chunk.get("type") not in ("text", "title"):
            continue
        plain = _strip_markup(chunk.get("markdown", ""))
        plain_lower = plain.lower()

        for label in labels:
            idx = plain_lower.find(label)
            if idx != -1:
                after = plain[idx + len(label):]
                m = re.match(r"\s*[:=]?\s*([^\n]{1,200})", after)
                if m:
                    val = m.group(1).strip().rstrip(",;")
                    if val:
                        return {
                            "value": val,
                            "source": "chunk",
                            "evidence": plain[max(0, idx - 30): idx + 250].strip(),
                            "confidence": 0.7
                        }
    return None


# ── Raw text search ───────────────────────────────────────────────────────────

def _from_text(labels: list, text: str, window: int, ftype: str) -> Optional[dict]:
    text_lower = text.lower()
    for label in labels:
        idx = text_lower.find(label)
        if idx != -1:
            after = text[idx + len(label):]
            m = re.match(r"\s*[:=]?\s*([^\n]{1,200})", after)
            if m:
                return {
                    "value": m.group(1).strip(),
                    "source": "text",
                    "evidence": text[max(0, idx - 50): idx + window].strip(),
                    "confidence": 0.5
                }
    return None


# ── Markup stripper ───────────────────────────────────────────────────────────

def _strip_markup(md: str) -> str:
    text = re.sub(r"<a id='[^']*'></a>", "", md)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"<::.*?::>", "", text, flags=re.DOTALL)
    return re.sub(r"\s{2,}", " ", text).strip()
