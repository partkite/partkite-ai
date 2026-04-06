"""
Google Cloud Vision API client with:
- Token bucket rate limiter (mirrors gemini.py pattern)
- Async HTTP calls using httpx
- TEXT_DETECTION for images, DOCUMENT_TEXT_DETECTION for PDFs

Set GOOGLE_VISION_API in .env with your API key.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import time

import httpx

from config import GOOGLE_VISION_API_KEY

log = logging.getLogger(__name__)

_VISION_URL = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}"

# ── Token bucket rate limiter ─────────────────────────────────────────────────
# Vision API free tier: 1800 requests/minute. We cap conservatively at 60 RPM.
_RPM_SAFE = 60
_RPS = _RPM_SAFE / 60.0


class _TokenBucket:
    def __init__(self, rate: float, capacity: float):
        self._rate = rate
        self._capacity = capacity
        self._tokens = capacity
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

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
            log.debug("Vision rate limit: sleeping %.2fs", wait)
            await asyncio.sleep(wait)


_bucket = _TokenBucket(rate=_RPS, capacity=10.0)

_MAX_RETRIES = 4
_RETRYABLE_CODES = {429, 503}


async def detect_text(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    """
    Run OCR on raw image bytes. Returns extracted text string.
    Uses DOCUMENT_TEXT_DETECTION for better multi-column / table support.
    """
    await _bucket.acquire()

    b64 = base64.b64encode(image_bytes).decode()
    payload = {
        "requests": [
            {
                "image": {"content": b64},
                "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
            }
        ]
    }

    last_exc: Exception | None = None
    async with httpx.AsyncClient(timeout=30.0) as client:
        for attempt in range(_MAX_RETRIES):
            try:
                resp = await client.post(_VISION_URL, json=payload)
                if resp.status_code in _RETRYABLE_CODES:
                    wait = 2 ** min(attempt, 4)
                    log.warning(
                        "Vision API %d. Retry %d/%d in %ds",
                        resp.status_code, attempt + 1, _MAX_RETRIES, wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                annotation = (
                    data.get("responses", [{}])[0]
                    .get("fullTextAnnotation", {})
                    .get("text", "")
                )
                return annotation.strip()
            except httpx.HTTPStatusError as e:
                last_exc = e
                if e.response.status_code not in _RETRYABLE_CODES:
                    raise
            except Exception as e:
                last_exc = e
                raise

    raise last_exc or RuntimeError("Vision API failed after retries")
