"""PART_LOOKUP handler — trigram-first, vector fallback."""
from __future__ import annotations

import asyncio

from partpilot.gemini import embed_text
from partpilot.models import ProductResult, QueryIntent, QueryResponse
from partpilot.search.hybrid import hybrid_search
from partpilot.search.trigram import trigram_search


async def handle_lookup(
    query: str,
    parts: list[str],
    limit: int,
    only_in_stock: bool,
) -> QueryResponse:
    # If multiple parts extracted, search each and merge
    search_terms = parts if parts else [query]

    tasks = [_search_one(term, limit, only_in_stock) for term in search_terms]
    results_per_term = await asyncio.gather(*tasks)

    # Deduplicate by product id, keep highest score
    seen: dict[str, ProductResult] = {}
    for results in results_per_term:
        for r in results:
            if r.id not in seen or r.score > seen[r.id].score:
                seen[r.id] = r

    merged = sorted(seen.values(), key=lambda x: x.score, reverse=True)[:limit]

    return QueryResponse(
        intent=QueryIntent.PART_LOOKUP,
        query=query,
        results=merged,
    )


async def _search_one(term: str, limit: int, only_in_stock: bool) -> list[ProductResult]:
    # Try trigram first (fast, great for part numbers)
    trgm = await trigram_search(term, limit=limit, only_in_stock=only_in_stock)
    if len(trgm) >= limit // 2:
        return trgm
    # Supplement with vector search if trigram results are thin
    embedding = await embed_text(term)
    return await hybrid_search(term, embedding, limit=limit, only_in_stock=only_in_stock)
