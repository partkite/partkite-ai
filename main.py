"""
PartPilot — FastAPI entry point.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from partpilot.db import close_pool, init_pool
from partpilot.handlers.bom import handle_bom
from partpilot.handlers.circuit import handle_circuit
from partpilot.handlers.compare import handle_compare
from partpilot.handlers.lookup import handle_lookup
from partpilot.handlers.semantic import handle_semantic
from partpilot.models import (
    BOMRequest,
    QueryIntent,
    QueryRequest,
    QueryResponse,
)
from partpilot.router import classify


@asynccontextmanager
async def lifespan(app: FastAPI):
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


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    """
    Natural language query endpoint.
    Handles: part lookup, semantic search, compare, category browse, circuit help.
    """
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
            bom_req = BOMRequest(
                text=req.query,
                only_in_stock=req.only_in_stock,
                limit_per_item=req.limit,
            )
            return await handle_bom(bom_req)

        case _:
            return await handle_semantic(req.query, QueryIntent.SEMANTIC, req.limit, req.only_in_stock)


@app.post("/api/bom", response_model=QueryResponse)
async def bom(req: BOMRequest) -> QueryResponse:
    """
    Dedicated BOM parsing endpoint.
    Accepts a messy multi-item parts list and returns matched products per item.
    """
    return await handle_bom(req)
