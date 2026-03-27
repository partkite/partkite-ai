"""COMPARE handler — fetch both parts, let Gemini synthesize a comparison."""
from __future__ import annotations

import asyncio

from partpilot.gemini import generate
from partpilot.models import ProductResult, QueryIntent, QueryResponse
from partpilot.search.trigram import trigram_search

_COMPARE_SYSTEM = """You are an expert electronics engineer.
Compare the given parts clearly and concisely:
- Key specs differences
- Use cases where each excels
- Availability / price in India (from the product data provided)
- Recommendation based on common use cases
Format as plain text, no markdown tables."""


async def handle_compare(
    query: str,
    parts: list[str],
    limit: int,
    only_in_stock: bool,
) -> QueryResponse:
    # Fetch top results for each part in parallel
    search_terms = parts if len(parts) >= 2 else _extract_vs_parts(query)
    tasks = [trigram_search(term, limit=3, only_in_stock=only_in_stock) for term in search_terms]
    results_per_part = await asyncio.gather(*tasks)

    all_results: list[ProductResult] = []
    sections: list[str] = []
    for term, results in zip(search_terms, results_per_part):
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

    # Deduplicate
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


def _extract_vs_parts(query: str) -> list[str]:
    """Naive split on 'vs', 'versus', 'and', 'or'."""
    import re
    parts = re.split(r"\s+(?:vs\.?|versus|and|or)\s+", query, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]
