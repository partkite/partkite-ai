"""
Gemini client with:
- Round-robin API key rotation
- Global token bucket rate limiter (shared across all concurrent requests)

Set GEMINI_API_KEY as a comma-separated list of keys in .env:
  GEMINI_API_KEY=key1,key2,key3
"""
from __future__ import annotations

import asyncio
import itertools
import logging
import time

from google import genai
from google.genai import types as gtypes

from config import GEMINI_API_KEY, GEMINI_CHAT_MODEL, GEMINI_EMBED_MODEL

log = logging.getLogger(__name__)

# ── Key pool ──────────────────────────────────────────────────────────────────

_keys: list[str] = [k.strip() for k in GEMINI_API_KEY.split(",") if k.strip()]
_clients: list[genai.Client] = [genai.Client(api_key=k) for k in _keys]
_cycle = itertools.cycle(range(len(_clients)))
_key_lock = asyncio.Lock()

if len(_keys) > 1:
    log.info("Gemini key rotation enabled: %d keys loaded", len(_keys))


async def _next_client() -> genai.Client:
    async with _key_lock:
        return _clients[next(_cycle)]


# ── Global token bucket (shared across ALL concurrent requests) ───────────────
# Limits total Gemini API calls to RPM_SAFE regardless of how many users
# are hitting the server simultaneously. Uses asyncio.sleep so waiting
# requests yield the event loop — no user blocks another.

_RPM_SAFE   = 250          # conservative cap; tune to your tier
_RPS        = _RPM_SAFE / 60.0

class _TokenBucket:
    def __init__(self, rate: float, capacity: float):
        self._rate     = rate
        self._capacity = capacity
        self._tokens   = capacity
        self._last     = time.monotonic()
        self._lock     = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            self._tokens = min(
                self._capacity,
                self._tokens + (now - self._last) * self._rate,
            )
            self._last = now
            if self._tokens < 1:
                wait = (1 - self._tokens) / self._rate
                self._tokens = 0
            else:
                self._tokens -= 1
                wait = 0.0
        if wait > 0:
            await asyncio.sleep(wait)

_bucket = _TokenBucket(rate=_RPS, capacity=float(max(len(_keys), 1) * 5))


# ── Rate-limit aware caller ───────────────────────────────────────────────────

_RETRYABLE  = (429, 503)
_MAX_RETRIES = len(_clients) + 2


async def _call_with_rotation(fn):
    """Acquire rate-limit token, then call fn(client), rotating keys on 429/503."""
    await _bucket.acquire()
    last_exc = None
    for attempt in range(_MAX_RETRIES):
        client = await _next_client()
        try:
            return await fn(client)
        except Exception as e:
            last_exc = e
            msg = str(e)
            if any(str(code) in msg for code in _RETRYABLE) or "quota" in msg.lower():
                wait = 2 ** min(attempt, 4)
                log.warning(
                    "Key %d rate limited. Retry %d/%d in %ds",
                    attempt % len(_clients), attempt + 1, _MAX_RETRIES, wait,
                )
                await asyncio.sleep(wait)
                continue
            raise
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
    """Batch embed multiple documents in one API call."""
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


async def generate(prompt: str, system: str | None = None, max_tokens: int = 2048) -> str:
    """Single-turn generation with optional system instruction."""
    async def _fn(client):
        resp = await client.aio.models.generate_content(
            model=GEMINI_CHAT_MODEL,
            contents=prompt,
            config=gtypes.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=max_tokens,
                system_instruction=system,
            ),
        )
        return resp.text
    return await _call_with_rotation(_fn)
