"""
llm_router.py — Multi-provider LLM router.

Providers:
  openai      — OpenAI GPT models (gpt-4o, gpt-4o-mini, etc.)
  chatgpt     — Alias for openai
  anthropic   — Anthropic Claude models
  gemini      — Google Gemini models
  groq        — Groq (ultra-fast inference: llama3, mixtral, etc.)
  grok        — xAI Grok models (grok-beta, grok-2)
  perplexity  — Perplexity AI (llama-3.1-sonar-large-128k-online)
  emergence   — Emergence AI orchestration API
  ollama      — Local models via Ollama
  python      — Heuristic-only, no LLM
  hybrid      — Heuristic first, LLM fallback (default pipeline behavior)
  none        — Disable AI, heuristic only

BYOK: keys passed per request, never stored permanently.
"""
from __future__ import annotations

import json
import re
from loguru import logger


class LLMRouter:
    def __init__(
        self,
        provider: str,
        api_key: str = "",
        model: str = "",
        base_url: str = "",
    ):
        # Normalize aliases
        p = provider.lower().strip()
        if p == "chatgpt":
            p = "openai"
        if p in ("python", "heuristic"):
            p = "none"
        # perplexity is handled as-is (no alias needed)
        self.provider = p
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    def is_available(self) -> bool:
        if self.provider in ("none", "hybrid", ""):
            return False
        if self.provider in ("openai", "anthropic", "gemini", "groq", "grok", "perplexity", "emergence") and not self.api_key:
            return False
        return True

    # ── Public methods ────────────────────────────────────────────────────────

    async def complete(self, prompt: str, system: str = "") -> str:
        """Send a completion request. Returns raw text response."""
        try:
            if self.provider == "openai":
                return await self._openai(prompt, system)
            elif self.provider == "anthropic":
                return await self._anthropic(prompt, system)
            elif self.provider == "gemini":
                return await self._gemini(prompt, system)
            elif self.provider == "groq":
                return await self._groq(prompt, system)
            elif self.provider == "grok":
                return await self._grok(prompt, system)
            elif self.provider == "perplexity":
                return await self._perplexity(prompt, system)
            elif self.provider == "emergence":
                return await self._emergence(prompt, system)
            elif self.provider == "ollama":
                return await self._ollama(prompt, system)
            else:
                logger.warning(f"Unknown LLM provider: '{self.provider}'")
                return ""
        except Exception as e:
            logger.error(f"LLM [{self.provider}] error: {e}")
            return ""

    async def extract_structured(self, prompt: str, schema_hint: str = "") -> dict:
        """
        Extract a structured JSON dict from the AI.
        Uses JSON response mode where available.
        """
        system = (
            "You are a precise document data extraction assistant. "
            "You MUST respond with valid JSON ONLY. "
            "No explanation, no markdown code fences, no extra text. "
            "If a value cannot be found in the document, use null. "
            "Do NOT invent or hallucinate values. "
            f"{schema_hint}"
        )

        if self.provider == "openai":
            raw = await self._openai(prompt, system, json_mode=True)
        elif self.provider == "groq":
            raw = await self._groq(prompt, system)
        elif self.provider == "grok":
            raw = await self._grok(prompt, system, json_mode=True)
        else:
            raw = await self.complete(prompt, system)

        return _parse_json_safe(raw)

    # ── Provider implementations ──────────────────────────────────────────────

    async def _openai(self, prompt: str, system: str, json_mode: bool = False) -> str:
        import httpx
        model = self.model or "gpt-4o-mini"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 4096,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        async with httpx.AsyncClient(timeout=90) as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                json=payload, headers=headers,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]

    async def _anthropic(self, prompt: str, system: str) -> str:
        import httpx
        model = self.model or "claude-3-5-haiku-20241022"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "max_tokens": 4096,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }
        async with httpx.AsyncClient(timeout=90) as client:
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                json=payload, headers=headers,
            )
            r.raise_for_status()
            return r.json()["content"][0]["text"]

    async def _gemini(self, prompt: str, system: str) -> str:
        import httpx
        model = self.model or "gemini-1.5-flash"
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={self.api_key}"
        )
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 4096,
                "responseMimeType": "application/json",
            },
        }
        async with httpx.AsyncClient(timeout=90) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            return r.json()["candidates"][0]["content"]["parts"][0]["text"]

    async def _groq(self, prompt: str, system: str) -> str:
        import httpx
        model = self.model or "llama-3.1-8b-instant"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 4096,
            "response_format": {"type": "json_object"},
        }
        async with httpx.AsyncClient(timeout=90) as client:
            r = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload, headers=headers,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]

    async def _grok(self, prompt: str, system: str, json_mode: bool = False) -> str:
        """xAI Grok — uses OpenAI-compatible API."""
        import httpx
        model = self.model or "grok-beta"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 4096,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        base = self.base_url or "https://api.x.ai"
        async with httpx.AsyncClient(timeout=90) as client:
            r = await client.post(
                f"{base}/v1/chat/completions",
                json=payload, headers=headers,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]

    async def _perplexity(self, prompt: str, system: str) -> str:
        """Perplexity AI — OpenAI-compatible API at api.perplexity.ai."""
        import httpx
        model = self.model or "llama-3.1-sonar-large-128k-online"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 4096,
        }
        async with httpx.AsyncClient(timeout=90) as client:
            r = await client.post(
                "https://api.perplexity.ai/chat/completions",
                json=payload, headers=headers,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]

    async def _emergence(self, prompt: str, system: str) -> str:
        """
        Emergence AI — orchestration API.
        Uses their /run endpoint with an agent task.
        Docs: https://api.emergence.ai
        """
        import httpx
        model = self.model or "em-llm-001"
        base = self.base_url or "https://api.emergence.ai"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 4096,
        }
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                f"{base}/v0/chat/completions",
                json=payload, headers=headers,
            )
            r.raise_for_status()
            data = r.json()
            # Handle both OpenAI-style and Emergence-style responses
            if "choices" in data:
                return data["choices"][0]["message"]["content"]
            if "response" in data:
                return data["response"]
            if "output" in data:
                return data["output"]
            return str(data)

    async def _ollama(self, prompt: str, system: str) -> str:
        import httpx
        model = self.model or "llama3"
        base = self.base_url or "http://localhost:11434"
        payload = {
            "model": model,
            "prompt": f"{system}\n\n{prompt}",
            "stream": False,
            "format": "json",
        }
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(f"{base}/api/generate", json=payload)
            r.raise_for_status()
            return r.json().get("response", "")


# ── JSON parsing helper ───────────────────────────────────────────────────────

def _parse_json_safe(text: str) -> dict:
    if not text:
        return {}
    text = text.strip()
    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    text = text.strip()
    try:
        parsed = json.loads(text)
        # Ensure we always return a dict at top level
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"items": parsed}
        return {"value": parsed}
    except Exception:
        pass
    # Try to extract the outermost JSON object
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            pass
    return {}
