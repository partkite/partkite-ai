"""SEMANTIC + CATEGORY handler — embedding-first hybrid search."""
from __future__ import annotations

from partpilot.gemini import embed_text, generate
from partpilot.models import ProductResult, QueryIntent, QueryResponse
from partpilot.search.hybrid import hybrid_search

_SEMANTIC_SYSTEM = """You are an expert electronics engineer assistant.
The user asked a question and we found relevant products from Indian electronics stores.
Briefly explain which products best match the user's need and why (2-4 sentences).
Be specific about specs. Do not hallucinate specs not shown in the product list."""


async def handle_semantic(
    query: str,
    intent: QueryIntent,
    limit: int,
    only_in_stock: bool,
) -> QueryResponse:
    embedding = await embed_text(query)
    results = await hybrid_search(
        query, embedding,
        limit=limit,
        only_in_stock=only_in_stock,
        trgm_weight=0.2,
        vec_weight=0.8,
    )

    answer = None
    if results:
        product_list = _format_products(results[:5])
        prompt = f"User query: {query}\n\nTop matching products:\n{product_list}"
        answer = await generate(prompt, system=_SEMANTIC_SYSTEM)

    return QueryResponse(
        intent=intent,
        query=query,
        results=results,
        answer=answer,
    )


def _format_products(results: list[ProductResult]) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        cats = ", ".join(r.categories or [])
        lines.append(
            f"{i}. {r.product_name} | SKU: {r.sku or 'N/A'} | "
            f"₹{r.price} | {r.source} | In stock: {r.is_in_stock} | Categories: {cats}"
        )
    return "\n".join(lines)
