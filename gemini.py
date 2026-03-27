"""
Gemini client with round-robin API key rotation.
Set GEMINI_API_KEY as a comma-separated list of keys in .env:
  GEMINI_API_KEY=key1,key2,key3
"""
from __future__ import annotations

import asyncio
import itertools
import logging

from google import genai
from google.genai import types as gtypes

from partpilot.config import GEMINI_API_KEY, GEMINI_CHAT_MODEL, GEMINI_EMBED_MODEL

log = logging.getLogger(__name__)

# ── Key pool ──────────────────────────────────────────────────────────────────

_keys: list[str] = [k.strip() for k in GEMINI_API_KEY.split(",") if k.strip()]
_clients: list[genai.Client] = [genai.Client(api_key=k) for k in _keys]
_cycle = itertools.cycle(range(len(_clients)))
_lock = asyncio.Lock()

if len(_keys) > 1:
    log.info("Gemini key rotation enabled: %d keys loaded", len(_keys))


async def _next_client() -> genai.Client:
    async with _lock:
        return _clients[next(_cycle)]


# ── Rate-limit aware caller ───────────────────────────────────────────────────

_RETRYABLE = (429, 503)
_MAX_RETRIES = len(_clients) + 2  # try every key at least once before giving up


async def _call_with_rotation(fn, *args, **kwargs):
    """Call fn(client, *args, **kwargs), rotating keys on 429/503."""
    last_exc = None
    for attempt in range(_MAX_RETRIES):
        client = await _next_client()
        try:
            return await fn(client, *args, **kwargs)
        except Exception as e:
            last_exc = e
            msg = str(e)
            # Rotate on rate-limit or quota errors
            if any(str(code) in msg for code in _RETRYABLE) or "quota" in msg.lower():
                wait = 2 ** min(attempt, 4)
                log.warning("Key %d hit rate limit, rotating. Retry %d/%d in %ds",
                            attempt % len(_clients), attempt + 1, _MAX_RETRIES, wait)
                await asyncio.sleep(wait)
                continue
            raise  # non-retryable error, raise immediately
    raise last_exc


# ── Public API ────────────────────────────────────────────────────────────────

async def embed_text(text: str) -> list[float]:
    """Embed a query string (RETRIEVAL_QUERY task)."""
    async def _fn(client):
        resp = await client.aio.models.embed_content(
            model=GEMINI_EMBED_MODEL,
            contents=text,
            config=gtypes.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=768,
            ),
        )
        return resp.embeddings[0].values
    return await _call_with_rotation(_fn)


async def embed_document(text: str) -> list[float]:
    """Embed a single product document (RETRIEVAL_DOCUMENT task)."""
    async def _fn(client):
        resp = await client.aio.models.embed_content(
            model=GEMINI_EMBED_MODEL,
            contents=text,
            config=gtypes.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=768,
            ),
        )
        return resp.embeddings[0].values
    return await _call_with_rotation(_fn)


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """Batch embed multiple product documents."""
    from google.genai.types import Content, Part
    contents = [Content(parts=[Part(text=t)]) for t in texts]

    async def _fn(client):
        resp = await client.aio.models.embed_content(
            model=GEMINI_EMBED_MODEL,
            contents=contents,
            config=gtypes.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=768,
            ),
        )
        return [e.values for e in resp.embeddings]
    return await _call_with_rotation(_fn)


async def generate(prompt: str, system: str | None = None) -> str:
    """Single-turn generation with optional system instruction."""
    async def _fn(client):
        resp = await client.aio.models.generate_content(
            model=GEMINI_CHAT_MODEL,
            contents=prompt,
            config=gtypes.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=2048,
                system_instruction=system,
            ),
        )
        return resp.text
    return await _call_with_rotation(_fn)
