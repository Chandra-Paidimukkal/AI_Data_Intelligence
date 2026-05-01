"""
quality_scorer.py — Feature 2: Extraction Quality Scoring

Computes a quality score (0-100) and grade (A-F) for any extraction result.
Analyzes:
  - Field coverage (% of fields extracted)
  - Average confidence
  - Critical field presence
  - Validation errors
  - Source quality (table > kv > text > fallback)
"""
from __future__ import annotations
from typing import Any


# Fields considered "critical" — if missing, score drops significantly
_CRITICAL_FIELD_HINTS = {
    "model_number", "model", "modelnumber", "manufacturer", "brand",
    "model_no", "part_number", "item_number",
}

# Source quality weights
_SOURCE_WEIGHTS = {
    "table": 1.0,
    "kv": 0.85,
    "chunk": 0.75,
    "text": 0.65,
    "text_pattern": 0.55,
    "ai:openai": 0.80,
    "ai:anthropic": 0.80,
    "ai:gemini": 0.80,
    "ai:groq": 0.75,
    "landingai_ade": 0.90,
    "fallback": 0.10,
    "heuristic_rescued": 0.50,
}


def compute_quality_score(
    result: dict,
    confidence: dict,
    sources: dict,
    schema_fields: list[str],
    validation_errors: dict,
    failure_log: list,
) -> dict:
    """
    Compute a quality score for an extraction result.

    Returns:
    {
        score: 0-100,
        grade: "A" | "B" | "C" | "D" | "F",
        breakdown: { coverage, avg_confidence, source_quality, penalty },
        missing_fields: [...],
        missing_critical: [...],
        low_confidence_fields: [...],
        suggestions: [...],
    }
    """
    if not schema_fields:
        return _empty_score()

    total = len(schema_fields)
    extracted = [f for f in schema_fields if result.get(f) is not None]
    missing = [f for f in schema_fields if result.get(f) is None]

    # ── Coverage score (0-40 points) ──────────────────────────────────────────
    coverage_pct = len(extracted) / total if total > 0 else 0
    coverage_score = coverage_pct * 40

    # ── Average confidence score (0-35 points) ────────────────────────────────
    conf_values = [confidence.get(f, 0) for f in extracted]
    avg_conf = sum(conf_values) / len(conf_values) if conf_values else 0
    confidence_score = avg_conf * 35

    # ── Source quality score (0-15 points) ────────────────────────────────────
    source_scores = []
    for f in extracted:
        src = sources.get(f, "fallback")
        # Handle "ai:provider" format
        weight = _SOURCE_WEIGHTS.get(src, 0.5)
        if src.startswith("ai:"):
            weight = _SOURCE_WEIGHTS.get(src, 0.75)
        source_scores.append(weight)
    avg_source = sum(source_scores) / len(source_scores) if source_scores else 0
    source_score = avg_source * 15

    # ── Penalty (0-10 points deducted) ────────────────────────────────────────
    penalty = 0

    # Missing critical fields
    missing_critical = [
        f for f in missing
        if any(hint in f.lower() for hint in _CRITICAL_FIELD_HINTS)
    ]
    penalty += len(missing_critical) * 3

    # Validation errors
    penalty += len(validation_errors) * 2

    # Failure log entries
    penalty += len([e for e in failure_log if e.get("type") == "required_missing"]) * 2

    penalty = min(penalty, 10)

    # ── Total score ───────────────────────────────────────────────────────────
    raw_score = coverage_score + confidence_score + source_score - penalty
    score = max(0, min(100, round(raw_score)))

    # ── Grade ─────────────────────────────────────────────────────────────────
    if score >= 90:
        grade = "A"
    elif score >= 75:
        grade = "B"
    elif score >= 60:
        grade = "C"
    elif score >= 45:
        grade = "D"
    else:
        grade = "F"

    # ── Low confidence fields ─────────────────────────────────────────────────
    low_conf_fields = [
        f for f in extracted
        if confidence.get(f, 0) < 0.5
    ]

    # ── Suggestions ───────────────────────────────────────────────────────────
    suggestions = _generate_suggestions(
        missing, missing_critical, low_conf_fields,
        sources, validation_errors, coverage_pct, avg_conf
    )

    return {
        "score": score,
        "grade": grade,
        "breakdown": {
            "coverage": round(coverage_score, 1),
            "avg_confidence": round(confidence_score, 1),
            "source_quality": round(source_score, 1),
            "penalty": round(penalty, 1),
        },
        "stats": {
            "total_fields": total,
            "extracted_fields": len(extracted),
            "missing_fields": len(missing),
            "coverage_pct": round(coverage_pct * 100, 1),
            "avg_confidence_pct": round(avg_conf * 100, 1),
        },
        "missing_fields": missing,
        "missing_critical": missing_critical,
        "low_confidence_fields": low_conf_fields,
        "validation_errors": list(validation_errors.keys()),
        "suggestions": suggestions,
    }


def compute_quality_for_records(
    records: list[dict],
    schema_fields: list[str],
) -> dict:
    """Compute quality score for multi-record extraction results."""
    if not records:
        return _empty_score()

    scores = []
    for rec in records:
        s = compute_quality_score(
            result=rec.get("result", {}),
            confidence=rec.get("confidence", {}),
            sources=rec.get("sources", {}),
            schema_fields=schema_fields,
            validation_errors=rec.get("validation", {}),
            failure_log=[],
        )
        scores.append(s["score"])

    avg_score = round(sum(scores) / len(scores))
    min_score = min(scores)
    max_score = max(scores)

    if avg_score >= 90: grade = "A"
    elif avg_score >= 75: grade = "B"
    elif avg_score >= 60: grade = "C"
    elif avg_score >= 45: grade = "D"
    else: grade = "F"

    return {
        "score": avg_score,
        "grade": grade,
        "record_count": len(records),
        "per_record_scores": scores,
        "min_score": min_score,
        "max_score": max_score,
        "stats": {
            "total_fields": len(schema_fields),
            "records": len(records),
        },
        "suggestions": [],
    }


def _generate_suggestions(
    missing: list,
    missing_critical: list,
    low_conf: list,
    sources: dict,
    validation_errors: dict,
    coverage_pct: float,
    avg_conf: float,
) -> list[str]:
    suggestions = []

    if missing_critical:
        suggestions.append(
            f"Critical fields missing: {', '.join(missing_critical[:3])}. "
            "Add more specific source_labels or table_labels to help the extractor find them."
        )

    if coverage_pct < 0.5:
        suggestions.append(
            "Less than 50% of fields were extracted. Consider using Landing AI or an LLM provider "
            "for better coverage on complex documents."
        )

    if avg_conf < 0.6 and avg_conf > 0:
        suggestions.append(
            "Average confidence is low. Try adding more descriptive field descriptions "
            "to guide the AI extractor."
        )

    if low_conf:
        suggestions.append(
            f"Low confidence on: {', '.join(low_conf[:4])}. "
            "Use Smart Retry to re-extract these fields with a targeted prompt."
        )

    fallback_fields = [f for f, s in sources.items() if s == "fallback"]
    if len(fallback_fields) > 3:
        suggestions.append(
            f"{len(fallback_fields)} fields fell back to default values. "
            "The document may not contain this data, or the schema labels need updating."
        )

    if validation_errors:
        suggestions.append(
            f"Validation errors on: {', '.join(list(validation_errors.keys())[:3])}. "
            "Check allowed_values in your schema definition."
        )

    return suggestions[:5]  # cap at 5 suggestions


def _empty_score() -> dict:
    return {
        "score": 0, "grade": "F",
        "breakdown": {"coverage": 0, "avg_confidence": 0, "source_quality": 0, "penalty": 0},
        "stats": {"total_fields": 0, "extracted_fields": 0, "missing_fields": 0,
                  "coverage_pct": 0, "avg_confidence_pct": 0},
        "missing_fields": [], "missing_critical": [],
        "low_confidence_fields": [], "validation_errors": [],
        "suggestions": [],
    }
