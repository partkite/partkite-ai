"""CIRCUIT_HELP handler — Gemini answers, then optionally suggests parts."""
from __future__ import annotations

from partpilot.gemini import embed_text, generate
from partpilot.models import QueryIntent, QueryResponse
from partpilot.search.hybrid import hybrid_search

_CIRCUIT_SYSTEM = """You are an expert electronics engineer.
Answer the user's circuit/component question precisely:
- Show calculations where relevant
- Give the exact component value/type needed
- End with a short "Suggested search term:" line with the best part to search for
Keep it concise and practical."""


async def handle_circuit(
    query: str,
    limit: int,
    only_in_stock: bool,
) -> QueryResponse:
    answer = await generate(query, system=_CIRCUIT_SYSTEM)

    # Extract suggested search term from answer and find products
    search_term = _extract_search_term(answer) or query
    embedding = await embed_text(search_term)
    results = await hybrid_search(
        search_term, embedding,
        limit=limit,
        only_in_stock=only_in_stock,
    )

    return QueryResponse(
        intent=QueryIntent.CIRCUIT_HELP,
        query=query,
        results=results,
        answer=answer,
    )


def _extract_search_term(text: str) -> str | None:
    import re
    m = re.search(r"Suggested search term:\s*(.+)", text, re.IGNORECASE)
    return m.group(1).strip() if m else None
