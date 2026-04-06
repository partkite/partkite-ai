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
    only_in_stock: bool = True       # BOM searches are purchase-intent — default to in-stock only
    limit_per_item: int = Field(default=5, ge=1, le=20)


class HealthRequest(BaseModel):
    names: list[str] = Field(..., min_length=1)


class BOMItemFlag(BaseModel):
    name: str
    nrnd: bool = False
    counterfeit_risk: str = "low"   # "low" | "medium" | "high"


class BOMHealth(BaseModel):
    score: int                       # 0–100 overall
    sourcability: int                # % of items found on 3+ distributors
    price_spread: int                # price consistency across distributors
    completeness: int                # avg confidence of parsed items
    recognition: int                 # % of items not skipped
    counterfeit: int                 # 100 = no risk, lower = more risk
    flags: list[BOMItemFlag] = []    # per-item nrnd / counterfeit flags


class ProductVariant(BaseModel):
    id: str
    title: str
    sku: str | None
    price: float | None
    is_available: bool
    attributes: dict[str, Any]


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
    description: str | None
    ai_description: str | None = None
    raw_data: dict[str, Any]
    score: float
    variants: list[ProductVariant] = []


class BOMItem(BaseModel):
    part: str
    name: str                        # normalized name from LLM
    qty: int | None = None
    confidence: float = 1.0
    results: list[ProductResult] = []
    skipped: bool = False            # True when confidence < 0.65


class QueryResponse(BaseModel):
    intent: QueryIntent
    query: str
    results: list[ProductResult] = []
    answer: str | None = None        # LLM-generated text for COMPARE / CIRCUIT_HELP / SEMANTIC
    bom_items: list[BOMItem] = []    # populated for BOM intent
    bom_health: BOMHealth | None = None  # populated for BOM intent


class EmbedRequest(BaseModel):
    """Internal — used by the bulk embedding job."""
    batch_size: int = Field(default=50, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
