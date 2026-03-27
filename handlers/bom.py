"""BOM handler — Gemini parses messy list, parallel search per item."""
from __future__ import annotations

import asyncio
import json
import re

from partpilot.gemini import embed_text, generate
from partpilot.models import BOMItem, BOMRequest, ProductResult, QueryIntent, QueryResponse
from partpilot.search.hybrid import hybrid_search
from partpilot.search.trigram import trigram_search

_BOM_SYSTEM = """You are an electronics BOM (Bill of Materials) parser.
Parse the messy parts list into structured JSON.

Return ONLY a JSON array, no markdown:
[
  {"part": "ESP32-WROOM-32D", "qty": 1, "spec": null},
  {"part": "10k resistor", "qty": 2, "spec": "0402"},
  {"part": "100nF capacitor", "qty": 50, "spec": null},
  ...
]
If quantity is unclear, set qty to null. Expand abbreviations into searchable names."""


async def handle_bom(req: BOMRequest) -> QueryResponse:
    # Step 1: Parse BOM with Gemini
    raw = await generate(req.text, system=_BOM_SYSTEM)
    raw = re.sub(r"```(?:json)?|```", "", raw).strip()

    try:
        parsed: list[dict] = json.loads(raw)
    except json.JSONDecodeError:
        parsed = []

    if not parsed:
        return QueryResponse(
            intent=QueryIntent.BOM,
            query=req.text,
            answer="Could not parse the parts list. Please check the format.",
        )

    # Step 2: Search for each item in parallel
    tasks = [
        _search_bom_item(item, req.limit_per_item, req.only_in_stock)
        for item in parsed
    ]
    bom_items = await asyncio.gather(*tasks)

    return QueryResponse(
        intent=QueryIntent.BOM,
        query=req.text,
        bom_items=list(bom_items),
    )


async def _search_bom_item(item: dict, limit: int, only_in_stock: bool) -> BOMItem:
    part = item.get("part", "")
    spec = item.get("spec")
    qty = item.get("qty")

    # Build search query: part + spec for better precision
    search_query = f"{part} {spec}".strip() if spec else part

    # Try trigram first (fast for part numbers like BC547, IRF540N)
    trgm_results = await trigram_search(search_query, limit=limit, only_in_stock=only_in_stock)

    if len(trgm_results) >= limit // 2:
        results = trgm_results
    else:
        # Supplement with vector search
        embedding = await embed_text(search_query)
        results = await hybrid_search(
            search_query, embedding,
            limit=limit,
            only_in_stock=only_in_stock,
        )

    return BOMItem(part=part, qty=qty, spec=spec, results=results)
