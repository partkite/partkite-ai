"""COMPARE handler — fetch best match per part, let Gemini synthesize a comparison."""
from __future__ import annotations

import asyncio
import re

from gemini import embed_text, generate
from models import ProductResult, QueryIntent, QueryResponse
from search.hybrid import hybrid_search
from search.trigram import trigram_search

_COMPARE_SYSTEM = """You are an expert electronics engineer.
Compare the given parts clearly and concisely:
- Key specs differences
- Use cases where each excels
- Availability / price in India (from the product data provided)
- Recommendation based on common use cases
Format as plain text, no markdown tables."""

# Minimum trigram score to trust the result without falling back to vector search
_TRGM_CONFIDENCE = 0.6


async def handle_compare(
    query: str,
    parts: list[str],
    limit: int,
    only_in_stock: bool,
) -> QueryResponse:
    search_terms = parts if len(parts) >= 2 else _extract_vs_parts(query)

    tasks = [_fetch_best(term, only_in_stock) for term in search_terms]
    best_per_part: list[list[ProductResult]] = await asyncio.gather(*tasks)

    all_results: list[ProductResult] = []
    sections: list[str] = []

    for term, results in zip(search_terms, best_per_part):
        all_results.extend(results)
        if results:
            top = results[0]
            sections.append(
                f"### {term}\n"
                f"Best match: {top.product_name}\n"
                f"SKU: {top.sku or 'N/A'} | Price: ₹{top.price} | "
                f"Source: {top.source} | In stock: {top.is_in_stock}"
            )
        else:
            sections.append(f"### {term}\nNo products found in database.")

    product_context = "\n\n".join(sections)
    prompt = f"User wants to compare: {query}\n\nProduct data from database:\n{product_context}"
    answer = await generate(prompt, system=_COMPARE_SYSTEM)

    seen: dict[str, ProductResult] = {}
    for r in all_results:
        if r.id not in seen:
            seen[r.id] = r

    return QueryResponse(
        intent=QueryIntent.COMPARE,
        query=query,
        results=list(seen.values())[:limit],
        answer=answer,
    )


async def _fetch_best(term: str, only_in_stock: bool) -> list[ProductResult]:
    """
    Try trigram first. If the top result score is below threshold
    (weak or no match), fall back to hybrid search for better recall.
    """
    results = await trigram_search(term, limit=3, only_in_stock=only_in_stock)
    if results and results[0].score >= _TRGM_CONFIDENCE:
        return results

    # Trigram match is weak — use hybrid search
    embedding = await embed_text(term)
    return await hybrid_search(term, embedding, limit=3, only_in_stock=only_in_stock)


def _extract_vs_parts(query: str) -> list[str]:
    parts = re.split(r"\s+(?:vs\.?|versus|and|or)\s+", query, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]
