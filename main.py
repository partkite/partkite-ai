"""
PartPilot — FastAPI entry point.
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

import asyncpg
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import LOG_LEVEL
from db import close_pool, get_pool, init_pool
from handlers.bom import handle_bom
from handlers.circuit import handle_circuit
from handlers.compare import handle_compare
from handlers.lookup import handle_lookup
from handlers.semantic import handle_semantic
from models import BOMRequest, QueryIntent, QueryRequest, QueryResponse
from router import classify

log = logging.getLogger(__name__)

# ── Timeouts ──────────────────────────────────────────────────────────────────
_QUERY_TIMEOUT = 30.0
_BOM_TIMEOUT   = 120.0

# ── BOM concurrency cap ───────────────────────────────────────────────────────
_BOM_CONCURRENCY = 5
_bom_semaphore: asyncio.Semaphore | None = None


# ── App setup ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _bom_semaphore
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    _bom_semaphore = asyncio.Semaphore(_BOM_CONCURRENCY)
    await init_pool()
    yield
    await close_pool()


app = FastAPI(
    title="PartPilot",
    description="Intelligent electronics parts search & assistant",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _handle_exc(e: Exception, context: str) -> None:
    if isinstance(e, asyncpg.TooManyConnectionsError):
        raise HTTPException(status_code=503, detail="Database busy — please retry shortly.")
    if isinstance(e, asyncio.TimeoutError):
        raise HTTPException(status_code=504, detail="Request timed out. Please try again.")
    log.error("%s error: %s", context, e, exc_info=True)
    raise HTTPException(status_code=500, detail="Internal server error.")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """
    Deep health check — verifies DB connectivity and Gemini reachability.
    Returns 200 only when both are healthy.
    """
    result: dict = {"status": "ok", "db": "ok", "gemini": "ok"}

    # DB check
    try:
        pool = get_pool()
        async with pool.acquire(timeout=3) as conn:
            await conn.fetchval("SELECT 1")
    except Exception as e:
        result["db"] = f"error: {e}"
        result["status"] = "degraded"

    # Gemini check — lightweight: just instantiate and check the client is alive
    # (avoid a real API call to save quota; connection errors surface on first real request)
    try:
        from gemini import _clients
        if not _clients:
            raise RuntimeError("No Gemini clients initialised")
    except Exception as e:
        result["gemini"] = f"error: {e}"
        result["status"] = "degraded"

    status_code = 200 if result["status"] == "ok" else 503
    from fastapi.responses import JSONResponse
    return JSONResponse(content=result, status_code=status_code)


@app.post("/api/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    async def _handle():
        intent, parts = await classify(req.query)
        match intent:
            case QueryIntent.PART_LOOKUP:
                return await handle_lookup(req.query, parts, req.limit, req.only_in_stock)
            case QueryIntent.COMPARE:
                return await handle_compare(req.query, parts, req.limit, req.only_in_stock)
            case QueryIntent.CIRCUIT_HELP:
                return await handle_circuit(req.query, req.limit, req.only_in_stock)
            case QueryIntent.SEMANTIC | QueryIntent.CATEGORY:
                return await handle_semantic(req.query, intent, req.limit, req.only_in_stock)
            case QueryIntent.BOM:
                return await handle_bom(BOMRequest(
                    text=req.query,
                    only_in_stock=req.only_in_stock,
                    limit_per_item=req.limit,
                ))
            case _:
                return await handle_semantic(req.query, QueryIntent.SEMANTIC, req.limit, req.only_in_stock)

    try:
        return await asyncio.wait_for(_handle(), timeout=_QUERY_TIMEOUT)
    except (HTTPException, asyncio.TimeoutError, asyncpg.TooManyConnectionsError):
        raise
    except Exception as e:
        _handle_exc(e, "query")


@app.post("/api/bom", response_model=QueryResponse)
async def bom(req: BOMRequest) -> QueryResponse:
    if _bom_semaphore._value == 0:
        raise HTTPException(
            status_code=429,
            detail=f"Server busy — max {_BOM_CONCURRENCY} concurrent BOM requests. Please retry shortly.",
        )

    async with _bom_semaphore:
        try:
            return await asyncio.wait_for(handle_bom(req), timeout=_BOM_TIMEOUT)
        except (HTTPException, asyncio.TimeoutError, asyncpg.TooManyConnectionsError):
            raise
        except Exception as e:
            _handle_exc(e, "bom")
