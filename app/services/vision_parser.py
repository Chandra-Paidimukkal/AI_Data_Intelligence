"""
vision_parser.py — AI Vision-based PDF parsing.

Converts PDF pages to images and sends them to vision-capable AI models
to extract text, tables, and dimension annotations that text parsers miss.

Supported providers:
  - openai / chatgpt  → GPT-4o vision
  - anthropic         → Claude 3.5 Sonnet vision
  - gemini            → Gemini 1.5 Flash vision
  - landingai         → LandingAI ADE (native vision parser)
  - groq              → Not supported (no vision)
  - python            → Not supported (no vision)
"""
from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path
from typing import Any

from loguru import logger


# ── Entry point ───────────────────────────────────────────────────────────────

async def vision_parse_document(
    file_path: str,
    provider: str,
    api_key: str,
    model: str = "",
    base_url: str = "",
    max_pages: int = 10,
) -> str:
    """
    Parse a PDF using AI vision — returns enriched markdown string.
    Converts each page to an image and sends to the vision API.
    The returned markdown is appended to the existing parsed markdown
    to supplement text extraction with vision-extracted content.
    """
    provider = provider.lower()

    if provider == "landingai":
        return await _vision_parse_landingai(file_path, api_key, base_url)

    if provider in ("openai", "chatgpt"):
        return await _vision_parse_openai(file_path, api_key, model or "gpt-4o", max_pages)

    if provider == "anthropic":
        return await _vision_parse_anthropic(file_path, api_key, model or "claude-3-5-sonnet-20241022", max_pages)

    if provider == "gemini":
        return await _vision_parse_gemini(file_path, api_key, model or "gemini-1.5-flash", max_pages)

    raise ValueError(f"Vision parsing not supported for provider: {provider}. Use landingai, openai, anthropic, or gemini.")


# ── PDF → images ──────────────────────────────────────────────────────────────

def _pdf_to_images(file_path: str, max_pages: int = 10, dpi: int = 150) -> list[bytes]:
    """Convert PDF pages to PNG images. Returns list of PNG bytes."""
    try:
        import fitz
    except ImportError:
        raise RuntimeError("PyMuPDF not installed. Run: pip install pymupdf")

    doc = fitz.open(file_path)
    images = []
    for page_num in range(min(doc.page_count, max_pages)):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=dpi)
        images.append(pix.tobytes("png"))
    doc.close()
    return images


def _image_to_base64(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode("utf-8")


# ── Vision prompt ─────────────────────────────────────────────────────────────

_VISION_PROMPT = """You are analyzing a page from an industrial/HVAC equipment specification sheet.

Extract ALL text visible in this image including:
1. All dimension values from diagrams (measurements with arrows, dimension lines)
   - Convert fractions to decimals: 13-3/4 = 13.75, 36-5/16 = 36.3125
   - Note which dimension is Height (H), Width (W), Depth/Length (D or L)
2. All values from specification tables
3. Model numbers, weights, electrical data
4. Any text annotations on diagrams

Format your response as structured text:
- For dimensions: "DIMENSION: [label] = [value] inches"
- For table data: reproduce the table in plain text
- For other data: reproduce as-is

Be precise and complete. Do not skip any numbers or measurements."""


# ── LandingAI vision ──────────────────────────────────────────────────────────

async def _vision_parse_landingai(file_path: str, api_key: str, environment: str) -> str:
    """Use LandingAI's native vision parser — best quality for spec sheets."""
    from app.services.landingai_service import parse_with_landingai
    result = await parse_with_landingai(file_path, api_key, environment or "production")
    return result.get("markdown", "")


# ── OpenAI GPT-4o vision ──────────────────────────────────────────────────────

async def _vision_parse_openai(file_path: str, api_key: str, model: str, max_pages: int) -> str:
    """Use GPT-4o vision to extract text from PDF pages."""
    try:
        import httpx
    except ImportError:
        raise RuntimeError("httpx not installed")

    images = _pdf_to_images(file_path, max_pages)
    all_text = []

    for i, img_bytes in enumerate(images):
        b64 = _image_to_base64(img_bytes)
        payload = {
            "model": model,
            "max_tokens": 2000,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Page {i+1}:\n{_VISION_PROMPT}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}},
                    ],
                }
            ],
        }
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                )
            if r.status_code == 200:
                text = r.json()["choices"][0]["message"]["content"]
                all_text.append(f"[VISION PAGE {i+1}]\n{text}")
                logger.info(f"GPT-4o vision page {i+1}: {len(text)} chars")
            else:
                logger.warning(f"GPT-4o vision page {i+1} failed: {r.status_code}")
        except Exception as e:
            logger.warning(f"GPT-4o vision page {i+1} error: {e}")

    return "\n\n".join(all_text)


# ── Anthropic Claude vision ───────────────────────────────────────────────────

async def _vision_parse_anthropic(file_path: str, api_key: str, model: str, max_pages: int) -> str:
    """Use Claude vision to extract text from PDF pages."""
    try:
        import httpx
    except ImportError:
        raise RuntimeError("httpx not installed")

    images = _pdf_to_images(file_path, max_pages)
    all_text = []

    for i, img_bytes in enumerate(images):
        b64 = _image_to_base64(img_bytes)
        payload = {
            "model": model,
            "max_tokens": 2000,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                        {"type": "text", "text": f"Page {i+1}:\n{_VISION_PROMPT}"},
                    ],
                }
            ],
        }
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    json=payload,
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    },
                )
            if r.status_code == 200:
                text = r.json()["content"][0]["text"]
                all_text.append(f"[VISION PAGE {i+1}]\n{text}")
                logger.info(f"Claude vision page {i+1}: {len(text)} chars")
            else:
                logger.warning(f"Claude vision page {i+1} failed: {r.status_code}")
        except Exception as e:
            logger.warning(f"Claude vision page {i+1} error: {e}")

    return "\n\n".join(all_text)


# ── Gemini vision ─────────────────────────────────────────────────────────────

async def _vision_parse_gemini(file_path: str, api_key: str, model: str, max_pages: int) -> str:
    """Use Gemini vision to extract text from PDF pages."""
    try:
        import httpx
    except ImportError:
        raise RuntimeError("httpx not installed")

    images = _pdf_to_images(file_path, max_pages)
    all_text = []

    for i, img_bytes in enumerate(images):
        b64 = _image_to_base64(img_bytes)
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": f"Page {i+1}:\n{_VISION_PROMPT}"},
                        {"inline_data": {"mime_type": "image/png", "data": b64}},
                    ]
                }
            ]
        }
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
            if r.status_code == 200:
                text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                all_text.append(f"[VISION PAGE {i+1}]\n{text}")
                logger.info(f"Gemini vision page {i+1}: {len(text)} chars")
            else:
                logger.warning(f"Gemini vision page {i+1} failed: {r.status_code}")
        except Exception as e:
            logger.warning(f"Gemini vision page {i+1} error: {e}")

    return "\n\n".join(all_text)
