from __future__ import annotations
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


# ── Intent ────────────────────────────────────────────────────────────────────

class QueryIntent(str, Enum):
    PART_LOOKUP   = "PART_LOOKUP"    # "LM2596", "ESP32-WROOM-32D"
    SEMANTIC      = "SEMANTIC"       # "convert 12V to 5V at 1A"
    COMPARE       = "COMPARE"        # "compare LM7805 vs AMS1117"
    CATEGORY      = "CATEGORY"       # "types of motor drivers"
    CIRCUIT_HELP  = "CIRCUIT_HELP"   # "what resistor for 5mm LED at 5V"
    BOM           = "BOM"            # messy multi-item list


# ── Request / Response ────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    only_in_stock: bool = False
    limit: int = Field(default=10, ge=1, le=50)


class BOMRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)
    only_in_stock: bool = False
    limit_per_item: int = Field(default=5, ge=1, le=20)


class ProductResult(BaseModel):
    id: str
    product_name: str
    sku: str | None
    price: float | None
    source: str
    product_url: str | None
    categories: list[str] | None
    brand: str | None
    is_in_stock: bool
    raw_data: dict[str, Any]
    score: float


class BOMItem(BaseModel):
    part: str
    qty: int | None = None
    spec: str | None = None          # e.g. "0402", "3.3V", "5mm"
    results: list[ProductResult] = []


class QueryResponse(BaseModel):
    intent: QueryIntent
    query: str
    results: list[ProductResult] = []
    answer: str | None = None        # LLM-generated text for COMPARE / CIRCUIT_HELP / SEMANTIC
    bom_items: list[BOMItem] = []    # populated for BOM intent


class EmbedRequest(BaseModel):
    """Internal — used by the bulk embedding job."""
    batch_size: int = Field(default=50, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
