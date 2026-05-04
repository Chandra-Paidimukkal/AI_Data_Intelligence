"""
chat.py â€” Platform-aware AI chatbot with vision/document upload support.

POST /api/v1/chat          â€” Full response
POST /api/v1/chat/stream   â€” SSE streaming
POST /api/v1/chat/upload   â€” Upload image/doc for vision analysis
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional, AsyncGenerator

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.config import settings

from app.core.database import get_db
from app.models.document import Document
from app.models.job import ExtractionJob
from app.models.schema import SchemaDefinition
from loguru import logger

router = APIRouter(prefix="/chat", tags=["Chat"])

# â”€â”€ Request models â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatAttachment(BaseModel):
    type: str        # "image" | "document"
    name: str        # original filename
    mime_type: str   # e.g. "image/png", "application/pdf"
    data: str        # base64-encoded content (for images) or extracted text (for docs)

class ChatRequest(BaseModel):
    message: str
    api_key: str
    model: str = "gpt-4o"
    provider: str = "openai"
    history: list[ChatMessage] = []
    attachments: list[ChatAttachment] = []  # uploaded files
    job_id: Optional[str] = None
    document_id: Optional[str] = None
    schema_id: Optional[str] = None


# â”€â”€ Upload endpoint â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@router.post("/upload")
async def upload_chat_file(file: UploadFile = File(...)):
    """
    Upload an image or document to use in the chatbot.
    Returns base64-encoded content + metadata for use in ChatRequest.attachments.
    Supports: images (PNG, JPG, WebP, GIF), PDFs, DOCX, TXT, CSV.
    """
    filename = file.filename or "upload"
    ext = Path(filename).suffix.lower()
    mime = file.content_type or "application/octet-stream"
    content = await file.read()

    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}
    is_image = ext in IMAGE_EXTS or mime.startswith("image/")

    if is_image:
        # Return base64 for vision models
        b64 = base64.b64encode(content).decode("utf-8")
        return {
            "type": "image",
            "name": filename,
            "mime_type": mime,
            "data": b64,
            "size": len(content),
        }
    else:
        # Extract text from document
        text = _extract_text_from_bytes(content, ext, filename)
        return {
            "type": "document",
            "name": filename,
            "mime_type": mime,
            "data": text[:8000],  # cap at 8000 chars
            "size": len(content),
        }


def _extract_text_from_bytes(content: bytes, ext: str, filename: str) -> str:
    """Extract text from document bytes."""
    import tempfile
    # Write to temp file and parse
    tmp_path = os.path.join(settings.UPLOAD_DIR, f"chat_tmp_{uuid.uuid4()}{ext}")
    try:
        with open(tmp_path, "wb") as f:
            f.write(content)
        from app.services.parser import parse_document
        result = parse_document(tmp_path)
        return result.get("document_text", "")[:8000]
    except Exception as e:
        logger.warning(f"Could not extract text from {filename}: {e}")
        # Fallback: try to decode as text
        try:
            return content.decode("utf-8", errors="ignore")[:8000]
        except Exception:
            return f"[Could not extract text from {filename}]"
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def _build_user_content(message: str, attachments: list, provider: str) -> object:
    """
    Build the user message content, including any attachments.
    For vision-capable providers (GPT-4o, Gemini), images are embedded as base64.
    For other providers, document text is appended to the message.
    """
    if not attachments:
        return message

    vision_providers = {"openai", "chatgpt", "gemini", "anthropic"}
    prov = provider.lower()

    # Build content parts for vision providers
    if prov in vision_providers:
        parts = []

        # Add text message first
        if message:
            parts.append({"type": "text", "text": message})

        for att in attachments:
            if att.type == "image" and prov in ("openai", "chatgpt"):
                # OpenAI vision format
                parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{att.mime_type};base64,{att.data}",
                        "detail": "high",
                    }
                })
            elif att.type == "image" and prov == "anthropic":
                # Anthropic vision format
                parts.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": att.mime_type,
                        "data": att.data,
                    }
                })
            else:
                # Document or non-vision provider: append text
                parts.append({
                    "type": "text",
                    "text": f"\n\n--- Attached: {att.name} ---\n{att.data}\n---"
                })

        return parts if len(parts) > 1 else message

    else:
        # Non-vision providers: append document text to message
        extra = ""
        for att in attachments:
            if att.type == "document":
                extra += f"\n\n--- Attached document: {att.name} ---\n{att.data}\n---"
            else:
                extra += f"\n\n[Image attached: {att.name} â€” vision not supported for this provider]"
        return message + extra


# â”€â”€ System prompt builder â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _build_system_prompt(
    job: Optional[ExtractionJob],
    doc: Optional[Document],
    schema: Optional[SchemaDefinition],
) -> str:
    parts = [
        "You are a highly capable AI assistant â€” like ChatGPT, Gemini, or Claude. "
        "You can answer ANY question on ANY topic: science, technology, history, coding, math, "
        "business, health, AI, machine learning, general knowledge, creative writing, and more.\n\n"
        "You are also embedded in the AQT Data Intelligence Platform â€” an AI-powered document "
        "extraction system for industrial equipment spec sheets. When users ask about the platform, "
        "you can help with:\n"
        "- Understanding extraction results and confidence scores\n"
        "- Diagnosing why fields are null or have low confidence\n"
        "- Building and improving extraction schemas\n"
        "- Understanding document content\n"
        "- Comparing AI engine results\n\n"
        "IMPORTANT: Never say you can only answer platform questions. Answer ALL questions fully "
        "and helpfully, just like ChatGPT or Gemini would. Be concise, accurate, and use markdown "
        "formatting for clarity."
    ]

    # Add document context
    if doc and doc.parsed_data:
        doc_text = doc.parsed_data.get("document_text", "")[:3000]
        kv_pairs = doc.parsed_data.get("kv_pairs", [])[:20]
        tables = doc.parsed_data.get("tables", [])

        parts.append(f"\n\n---\n## Current Document: {doc.file_name}")
        parts.append(f"Pages: {doc.page_count} | Status: {doc.status}")

        if doc_text:
            parts.append(f"\n### Document Text (excerpt):\n{doc_text}")

        if kv_pairs:
            kv_str = "\n".join(f"- {p['key']}: {p['value']}" for p in kv_pairs[:15])
            parts.append(f"\n### Key-Value Pairs:\n{kv_str}")

        if tables:
            parts.append(f"\n### Tables found: {len(tables)}")
            for i, t in enumerate(tables[:2]):
                headers = " | ".join(str(h) for h in t.get("headers", []))
                parts.append(f"Table {i+1} headers: {headers}")

    # Add schema context
    if schema:
        fields_summary = []
        for f in (schema.fields or [])[:30]:
            desc = f.get("description", "")
            fields_summary.append(f"  - {f['name']} ({f.get('type','string')}): {desc[:80]}")

        parts.append(f"\n\n---\n## Current Schema: {schema.name}")
        if schema.description:
            parts.append(f"Description: {schema.description}")
        parts.append(f"Fields ({len(schema.fields or [])}):\n" + "\n".join(fields_summary))

    # Add extraction result context
    if job and job.result:
        result_data = job.result
        parts.append(f"\n\n---\n## Current Extraction Result")
        parts.append(f"Job ID: {job.id}")
        parts.append(f"Schema: {job.schema_name}")
        parts.append(f"Provider: {job.provider or 'python'}")
        parts.append(f"Status: {job.status}")

        # Single record
        if "result" in result_data and isinstance(result_data["result"], dict):
            result = result_data["result"]
            confidence = result_data.get("confidence", {})
            sources = result_data.get("sources", {})

            parts.append("\n### Extracted Fields:")
            for fname, val in result.items():
                conf = confidence.get(fname, 0)
                src = sources.get(fname, "")
                conf_pct = f"{round(conf * 100)}%"
                parts.append(f"  - {fname}: {val} (confidence: {conf_pct}, source: {src})")

            # Highlight low confidence fields
            low_conf = [f for f, c in confidence.items() if c < 0.5]
            if low_conf:
                parts.append(f"\n### Low Confidence Fields: {', '.join(low_conf)}")

            # Failure log
            failure_log = result_data.get("failure_log", [])
            if failure_log:
                parts.append("\n### Warnings:")
                for f in failure_log[:5]:
                    parts.append(f"  - {f.get('reason', f.get('type', ''))}")

        # Multi-record
        elif "records" in result_data:
            records = result_data["records"]
            parts.append(f"\n### Multi-Record Result: {len(records)} records")
            for i, rec in enumerate(records[:5]):
                model_num = rec.get("result", {}).get("ModelNumber") or \
                           rec.get("result", {}).get("model_number") or f"Record {i+1}"
                non_null = sum(1 for v in rec.get("result", {}).values() if v is not None)
                total = len(rec.get("result", {}))
                parts.append(f"  - {model_num}: {non_null}/{total} fields extracted")

    return "\n".join(parts)


# â”€â”€ Non-streaming endpoint â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@router.post("")
async def chat(req: ChatRequest, db: Session = Depends(get_db)):
    """Send a message and get a full response."""
    if not req.api_key:
        raise HTTPException(400, "OpenAI API key required.")

    # Load context
    job = db.query(ExtractionJob).filter(ExtractionJob.id == req.job_id).first() if req.job_id else None
    doc = db.query(Document).filter(Document.id == req.document_id).first() if req.document_id else None
    schema = db.query(SchemaDefinition).filter(SchemaDefinition.id == req.schema_id).first() if req.schema_id else None

    system_prompt = _build_system_prompt(job, doc, schema)

    # Build messages â€” support vision (image attachments) for GPT-4o
    messages = [{"role": "system", "content": system_prompt}]
    for h in req.history[-10:]:
        messages.append({"role": h.role, "content": h.content})

    # Build user message with optional attachments
    user_content = _build_user_content(req.message, req.attachments, req.provider)
    messages.append({"role": "user", "content": user_content})

    provider = req.provider.lower().strip()

    try:
        import httpx

        # â”€â”€ OpenAI / ChatGPT â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if provider in ("openai", "chatgpt"):
            model = req.model or "gpt-4o"
            headers = {
                "Authorization": f"Bearer {req.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1500,
            }
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    json=payload, headers=headers,
                )
                r.raise_for_status()
                data = r.json()
                reply = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})

        # â”€â”€ Google Gemini â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        elif provider == "gemini":
            model = req.model or "gemini-2.0-flash"
            # Gemini uses a different message format; combine history into user message
            user_message = req.message
            if req.history:
                history_text = "\n".join(
                    f"{'User' if h.role == 'user' else 'Assistant'}: {h.content}"
                    for h in req.history[-10:]
                )
                user_message = f"{history_text}\nUser: {req.message}"
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent?key={req.api_key}"
            )
            payload = {
                "system_instruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"parts": [{"text": user_message}]}],
                "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1500},
            }
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(url, json=payload)
                r.raise_for_status()
                reply = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                usage = {}

        # â”€â”€ Anthropic Claude â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        elif provider == "anthropic":
            model = req.model or "claude-3-5-haiku-20241022"
            headers = {
                "x-api-key": req.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }
            # Anthropic uses system separately; messages must be user/assistant only
            anthropic_messages = [
                {"role": h.role if h.role in ("user", "assistant") else "user", "content": h.content}
                for h in req.history[-10:]
            ]
            anthropic_messages.append({"role": "user", "content": req.message})
            payload = {
                "model": model,
                "max_tokens": 1500,
                "system": system_prompt,
                "messages": anthropic_messages,
            }
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    json=payload, headers=headers,
                )
                r.raise_for_status()
                reply = r.json()["content"][0]["text"]
                usage = {}

        # â”€â”€ Groq â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        elif provider == "groq":
            model = req.model or "llama-3.1-8b-instant"
            headers = {
                "Authorization": f"Bearer {req.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1500,
            }
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    json=payload, headers=headers,
                )
                r.raise_for_status()
                data = r.json()
                reply = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})

        # â”€â”€ Perplexity â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        elif provider == "perplexity":
            model = req.model or "llama-3.1-sonar-large-128k-online"
            headers = {
                "Authorization": f"Bearer {req.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1500,
            }
            async with httpx.AsyncClient(timeout=90) as client:
                r = await client.post(
                    "https://api.perplexity.ai/chat/completions",
                    json=payload, headers=headers,
                )
                r.raise_for_status()
                data = r.json()
                reply = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})

        # â”€â”€ Emergence AI â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        elif provider == "emergence":
            model = req.model or "em-llm-001"
            headers = {
                "Authorization": f"Bearer {req.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1500,
            }
            async with httpx.AsyncClient(timeout=90) as client:
                r = await client.post(
                    "https://api.emergence.ai/v0/chat/completions",
                    json=payload, headers=headers,
                )
                r.raise_for_status()
                data = r.json()
                if "choices" in data:
                    reply = data["choices"][0]["message"]["content"]
                elif "response" in data:
                    reply = data["response"]
                else:
                    reply = str(data)
                usage = data.get("usage", {})

        else:
            raise HTTPException(400, f"Unsupported provider: {provider}")

        return {
            "reply": reply,
            "model": req.model,
            "provider": provider,
            "usage": usage,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error [{provider}]: {e}")
        raise HTTPException(500, f"Chat failed: {e}")


# â”€â”€ Streaming endpoint â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@router.post("/stream")
async def chat_stream(req: ChatRequest, db: Session = Depends(get_db)):
    """Send a message and stream the response token by token (SSE)."""
    if not req.api_key:
        raise HTTPException(400, "OpenAI API key required.")

    job = db.query(ExtractionJob).filter(ExtractionJob.id == req.job_id).first() if req.job_id else None
    doc = db.query(Document).filter(Document.id == req.document_id).first() if req.document_id else None
    schema = db.query(SchemaDefinition).filter(SchemaDefinition.id == req.schema_id).first() if req.schema_id else None

    system_prompt = _build_system_prompt(job, doc, schema)

    messages = [{"role": "system", "content": system_prompt}]
    for h in req.history[-10:]:
        messages.append({"role": h.role, "content": h.content})
    user_content = _build_user_content(req.message, req.attachments, req.provider)
    messages.append({"role": "user", "content": user_content})

    provider = req.provider.lower().strip()

    async def generate() -> AsyncGenerator[str, None]:
        import httpx

        # â”€â”€ Non-OpenAI providers: do a regular request, emit as single chunk â”€â”€
        if provider not in ("openai", "chatgpt"):
            try:
                if provider == "gemini":
                    model = req.model or "gemini-2.0-flash"
                    user_message = req.message
                    if req.history:
                        history_text = "\n".join(
                            f"{'User' if h.role == 'user' else 'Assistant'}: {h.content}"
                            for h in req.history[-10:]
                        )
                        user_message = f"{history_text}\nUser: {req.message}"
                    url = (
                        f"https://generativelanguage.googleapis.com/v1beta/models/"
                        f"{model}:generateContent?key={req.api_key}"
                    )
                    payload = {
                        "system_instruction": {"parts": [{"text": system_prompt}]},
                        "contents": [{"parts": [{"text": user_message}]}],
                        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1500},
                    }
                    async with httpx.AsyncClient(timeout=60) as client:
                        r = await client.post(url, json=payload)
                        r.raise_for_status()
                        full_response = r.json()["candidates"][0]["content"]["parts"][0]["text"]

                elif provider == "anthropic":
                    model = req.model or "claude-3-5-haiku-20241022"
                    headers = {
                        "x-api-key": req.api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    }
                    anthropic_messages = [
                        {"role": h.role if h.role in ("user", "assistant") else "user", "content": h.content}
                        for h in req.history[-10:]
                    ]
                    anthropic_messages.append({"role": "user", "content": req.message})
                    payload = {
                        "model": model,
                        "max_tokens": 1500,
                        "system": system_prompt,
                        "messages": anthropic_messages,
                    }
                    async with httpx.AsyncClient(timeout=60) as client:
                        r = await client.post(
                            "https://api.anthropic.com/v1/messages",
                            json=payload, headers=headers,
                        )
                        r.raise_for_status()
                        full_response = r.json()["content"][0]["text"]

                elif provider == "groq":
                    model = req.model or "llama-3.1-8b-instant"
                    headers = {
                        "Authorization": f"Bearer {req.api_key}",
                        "Content-Type": "application/json",
                    }
                    payload = {
                        "model": model,
                        "messages": messages,
                        "temperature": 0.7,
                        "max_tokens": 1500,
                    }
                    async with httpx.AsyncClient(timeout=60) as client:
                        r = await client.post(
                            "https://api.groq.com/openai/v1/chat/completions",
                            json=payload, headers=headers,
                        )
                        r.raise_for_status()
                        full_response = r.json()["choices"][0]["message"]["content"]

                elif provider == "perplexity":
                    model = req.model or "llama-3.1-sonar-large-128k-online"
                    headers = {
                        "Authorization": f"Bearer {req.api_key}",
                        "Content-Type": "application/json",
                    }
                    payload = {
                        "model": model,
                        "messages": messages,
                        "temperature": 0.7,
                        "max_tokens": 1500,
                    }
                    async with httpx.AsyncClient(timeout=90) as client:
                        r = await client.post(
                            "https://api.perplexity.ai/chat/completions",
                            json=payload, headers=headers,
                        )
                        r.raise_for_status()
                        full_response = r.json()["choices"][0]["message"]["content"]

                elif provider == "emergence":
                    model = req.model or "em-llm-001"
                    headers = {
                        "Authorization": f"Bearer {req.api_key}",
                        "Content-Type": "application/json",
                    }
                    payload = {
                        "model": model,
                        "messages": messages,
                        "temperature": 0.7,
                        "max_tokens": 1500,
                    }
                    async with httpx.AsyncClient(timeout=90) as client:
                        r = await client.post(
                            "https://api.emergence.ai/v0/chat/completions",
                            json=payload, headers=headers,
                        )
                        r.raise_for_status()
                        data = r.json()
                        if "choices" in data:
                            full_response = data["choices"][0]["message"]["content"]
                        elif "response" in data:
                            full_response = data["response"]
                        else:
                            full_response = str(data)

                else:
                    full_response = f"Unsupported provider: {provider}"

                yield f"data: {json.dumps({'content': full_response})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return

        # â”€â”€ OpenAI true SSE streaming â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        headers = {
            "Authorization": f"Bearer {req.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": req.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1500,
            "stream": True,
        }
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream(
                    "POST",
                    "https://api.openai.com/v1/chat/completions",
                    json=payload,
                    headers=headers,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                yield "data: [DONE]\n\n"
                                break
                            try:
                                data = json.loads(data_str)
                                delta = data["choices"][0]["delta"].get("content", "")
                                if delta:
                                    yield f"data: {json.dumps({'content': delta})}\n\n"
                            except Exception:
                                pass
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
