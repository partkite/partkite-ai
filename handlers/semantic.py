"""SEMANTIC + CATEGORY handler — embedding-first hybrid search."""
from __future__ import annotations

from gemini import embed_text, generate
from models import ProductResult, QueryIntent, QueryResponse
from search.hybrid import hybrid_search

_SEMANTIC_SYSTEM = """You are an expert electronics engineer assistant.
The user asked a question and we found relevant products from Indian electronics stores.
Briefly explain which products best match the user's need and why (2-4 sentences).
Be specific about specs. Do not hallucinate specs not shown in the product list.

CRITICAL INSTRUCTION: You MUST start your response with the EXACT index number of the SINGLE best matching product wrapped in brackets. Example: "[3]". Then provide your explanation."""


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
        only_in_stock=True,
        trgm_weight=0.2,
        vec_weight=0.8,
    )

    answer = None
    if results:
        import re
        top_k = results[:5]
        product_list = _format_products(top_k)
        prompt = f"User query: {query}\n\nTop matching products:\n{product_list}"
        answer = await generate(prompt, system=_SEMANTIC_SYSTEM)
        
        # Strip thinking blocks which block regex matching at start of string
        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
        
        match = re.match(r"^\s*\[(\d+)\]\s*", answer)
        if match:
            # Map the AI's 1-indexed output to our top_k list
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(top_k):
                # Find the actual product in the full results array and move it to the front
                recommended = top_k[idx]
                results.remove(recommended)
                results.insert(0, recommended)
            answer = answer[match.end():].strip()
    else:
        answer = "No matching products found. Try rephrasing your query or removing stock filters."

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
