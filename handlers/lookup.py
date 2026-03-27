"""PART_LOOKUP handler — trigram-first with score validation, hybrid fallback, exact-match pinning."""
from __future__ import annotations

import asyncio
import re

from gemini import embed_text
from models import ProductResult, QueryIntent, QueryResponse
from search.hybrid import hybrid_search
from search.trigram import trigram_search

# Minimum trigram score to trust results without falling back to hybrid
_TRGM_CONFIDENCE  = 0.6
# Trigram score above which the top result is pinned to rank 1
_EXACT_PIN_SCORE  = 0.85
_PART_NUM_RE      = re.compile(r'^[A-Za-z0-9\-]+$')


def _is_part_number(term: str) -> bool:
    tokens = term.strip().split()
    return len(tokens) <= 2 and all(_PART_NUM_RE.match(t) for t in tokens)


async def handle_lookup(
    query: str,
    parts: list[str],
    limit: int,
    only_in_stock: bool,
) -> QueryResponse:
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
    trgm = await trigram_search(term, limit=limit, only_in_stock=only_in_stock)

    # Check if trigram returned confident results
    strong = trgm and trgm[0].score >= _TRGM_CONFIDENCE
    if strong:
        results = trgm
    else:
        # Weak or no trigram match — use hybrid for better recall
        embedding = await embed_text(term)
        results   = await hybrid_search(term, embedding, limit=limit, only_in_stock=only_in_stock)

    # For part-number queries, pin an exact trigram match to rank 1
    if _is_part_number(term) and trgm and trgm[0].score >= _EXACT_PIN_SCORE:
        pinned  = trgm[0]
        rest    = [r for r in results if r.id != pinned.id]
        results = [pinned] + rest

    return results[:limit]
