"""
BOM handler — architecture:

1. Gemini parses raw BOM → structured items with query expansion + dedup
2. Post-parse: merge duplicate names (sum qty), null qty → 1
3. Single batch embed call for all unique queries across all items
4. Per item search:
   - Part-number items: exact trigram first; if strong match found, pin to rank 1
   - Vector search across all search_terms (top 30 each)
   - Trigram search on important_tokens
   - Union: score = (0.6*vec + 0.4*trgm) * token_match_factor
   - token_match_factor penalises results missing critical tokens (e.g. wrong value)
5. Items below confidence 0.65 → skipped
"""
from __future__ import annotations

import asyncio
import json
import logging
import re

from gemini import embed_batch, generate
from models import BOMHealth, BOMItem, BOMItemFlag, BOMRequest, ProductResult, QueryIntent, QueryResponse
from search.trigram import trigram_search
from search.vector import vector_search

log = logging.getLogger(__name__)

BOM_MAX_ITEMS         = 100
CONFIDENCE_THRESHOLD  = 0.65
VEC_WEIGHT            = 0.6
TRGM_WEIGHT           = 0.4
EXACT_MATCH_THRESHOLD = 0.85
TOKEN_PENALTY         = 0.4
_PART_NUM_RE = re.compile(r'^[A-Za-z0-9\-]+$')

_BOM_SYSTEM = """You are an expert electronics BOM (Bill of Materials) parser.

Parse the input into a JSON array. Each item must follow this exact structure:
{
  "name": "<clean, normalized, searchable component name>",
  "search_terms": [
    "<original query>",
    "<expanded variant 1>",
    "<expanded variant 2>",
    "<expanded variant 3>",
    "<expanded variant 4>"
  ],
  "important_tokens": ["<token1>", "<token2>", "<token3>"],
  "quantity": <integer>,
  "confidence": <float 0.0-1.0>
}

Rules:
- name: normalized and searchable. e.g. "10k resistor 0402 SMD" not "R1"
- If the same component appears multiple times, merge into ONE entry and SUM the quantities
- search_terms: exactly 5 variants covering synonyms, value formats, package codes
- important_tokens: 2-5 tokens that MUST appear in a matching product (values, package, type)
- quantity: always an integer — if unclear default to 1, never null
- confidence: how well the LLM understands this component — its type, value, package, and function. 1.0 = fully known, 0.0 = completely unrecognised or ambiguous. This reflects certainty about the generated name/search_terms/tokens being correct, NOT whether the part exists.
- Return ONLY the JSON array, no markdown, no explanation"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_part_number(name: str) -> bool:
    """True for short alphanumeric names like BC547, IRF540N, ESP32-WROOM-32D."""
    tokens = name.strip().split()
    return len(tokens) <= 2 and all(_PART_NUM_RE.match(t) for t in tokens)


def _token_match_factor(tokens: list[str], product: ProductResult) -> float:
    """
    Returns a multiplier in [TOKEN_PENALTY, 1.0].
    For each important token that does NOT appear in product_name or sku,
    apply TOKEN_PENALTY. Multiple misses compound.
    """
    if not tokens:
        return 1.0
    haystack = (
        (product.product_name or "").lower() + " " +
        (product.sku or "").lower()
    )
    factor = 1.0
    for tok in tokens:
        if tok.lower() not in haystack:
            factor *= TOKEN_PENALTY
    return factor


def _extract_complete_items(raw: str) -> tuple[list[dict], bool]:
    """
    Parse as much of a potentially truncated JSON array as possible.
    Returns (complete_items, was_truncated).
    """
    raw = raw.strip()
    try:
        return json.loads(raw), False
    except json.JSONDecodeError:
        pass

    items = []
    depth = 0
    start = None
    for i, ch in enumerate(raw):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    items.append(json.loads(raw[start:i + 1]))
                except json.JSONDecodeError:
                    pass
                start = None

    return items, True


def _dedup_items(items: list[dict]) -> list[dict]:
    """
    Merge items with the same normalised name, summing quantities.
    Keeps the entry with the highest confidence as the base.
    """
    merged: dict[str, dict] = {}
    for item in items:
        key = item.get("name", "").strip().lower()
        if key not in merged:
            merged[key] = dict(item)
            merged[key]["quantity"] = item.get("quantity") or 1
        else:
            existing = merged[key]
            existing["quantity"] = (existing.get("quantity") or 1) + (item.get("quantity") or 1)
            if float(item.get("confidence", 0)) > float(existing.get("confidence", 0)):
                existing["confidence"]       = item["confidence"]
                existing["search_terms"]     = item.get("search_terms", existing.get("search_terms", []))
                existing["important_tokens"] = item.get("important_tokens", existing.get("important_tokens", []))
    result = list(merged.values())
    log.info("[BOM] dedup: %d → %d items", len(items), len(result))
    return result


# ── LLM parse with truncation retry ──────────────────────────────────────────

async def _parse_bom_with_retry(text: str, max_rounds: int = 3) -> list[dict]:
    remaining    = text
    all_items:   list[dict] = []
    parsed_names: set[str]  = set()

    for round_num in range(max_rounds):
        log.info("[BOM] parse round %d — input: %s", round_num + 1, remaining[:80])

        raw = await generate(remaining, system=_BOM_SYSTEM, max_tokens=8192)
        log.debug("[BOM] round %d raw:\n%s", round_num + 1, raw)

        raw = re.sub(r"```(?:json)?|```", "", raw).strip()
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        items, truncated = _extract_complete_items(raw)

        # null qty → 1
        for it in items:
            if it.get("quantity") is None:
                it["quantity"] = 1

        new_items = [it for it in items if it.get("name") not in parsed_names]
        all_items.extend(new_items)
        for it in new_items:
            parsed_names.add(it.get("name", ""))

        log.info(
            "[BOM] round %d: %d items (%s), total: %d",
            round_num + 1, len(new_items),
            "truncated" if truncated else "complete",
            len(all_items),
        )

        if not truncated:
            break

        covered  = {n.lower() for n in parsed_names}
        leftover = [
            line for line in text.splitlines()
            if line.strip() and not any(tok in line.lower() for tok in covered)
        ]
        if not leftover:
            break
        remaining = "\n".join(leftover)
        log.info("[BOM] retrying with %d uncovered lines", len(leftover))

    return _dedup_items(all_items)


# ── BOM Health ────────────────────────────────────────────────────────────────

_HEALTH_SYSTEM = """You are an expert electronics engineer with deep knowledge of component supply chains.

Given a list of component names from a Bill of Materials, for each component return:
- nrnd: true if the part is Not Recommended for New Designs (obsolete, end-of-life, or has a clear modern replacement)
- counterfeit_risk: "high" if widely counterfeited (e.g. LM7805, IRF540N, 78xx regulators, popular MOSFETs), "medium" if occasionally faked, "low" if rarely counterfeited

Respond ONLY with valid JSON array, no markdown:
[{"name": "<exact name>", "nrnd": false, "counterfeit_risk": "low"}, ...]"""


async def _compute_flags_only(item_names: list[str]) -> BOMHealth:
    """
    Run only the Gemini-based assessment (counterfeit + NRND).
    Returns a BOMHealth with score=0 and deterministic fields=0 —
    the frontend fills those in from its own data.
    """
    flags: list[BOMItemFlag] = []
    counterfeit_score = 100

    if item_names:
        try:
            raw = await generate(
                "\n".join(item_names),
                system=_HEALTH_SYSTEM,
                max_tokens=1024,
            )
            # Strip markdown fences and any thinking/preamble — extract first JSON array
            raw = re.sub(r"```(?:json)?|```", "", raw).strip()
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            start, end = raw.find("["), raw.rfind("]")
            if start == -1 or end == -1:
                raise ValueError(f"No JSON array found in response: {raw[:200]}")
            raw = raw[start:end + 1]
            data = json.loads(raw)
            high_cf = 0
            med_cf = 0
            for entry in data:
                name = entry.get("name", "")
                nrnd = bool(entry.get("nrnd", False))
                cf = entry.get("counterfeit_risk", "low")
                flags.append(BOMItemFlag(name=name, nrnd=nrnd, counterfeit_risk=cf))
                if cf == "high":
                    high_cf += 1
                elif cf == "medium":
                    med_cf += 1
            n = len(item_names)
            deduction = (high_cf / n) * 40 + (med_cf / n) * 20
            counterfeit_score = max(0, round(100 - deduction))
        except Exception as e:
            log.warning("[BOM health] Gemini flags failed: %s — raw: %.300s", e, raw if 'raw' in dir() else "N/A")

    return BOMHealth(
        score=0,           # filled by frontend
        sourcability=0,    # filled by frontend
        price_spread=0,    # filled by frontend
        completeness=0,    # filled by frontend
        recognition=0,     # filled by frontend
        counterfeit=counterfeit_score,
        flags=flags,
    )


async def _compute_health(bom_items: list[BOMItem], parsed: list[dict]) -> BOMHealth:
    """
    Compute BOM health score from search results + Gemini knowledge.
    Deterministic scores from data; counterfeit/NRND from Gemini.
    """
    total = len(bom_items)
    if total == 0:
        return BOMHealth(score=0, sourcability=0, price_spread=0,
                         completeness=0, recognition=0, counterfeit=100)

    # ── Recognition: % of items not skipped ──────────────────────────────────
    found = sum(1 for i in bom_items if not i.skipped)
    recognition = round(found / total * 100)

    # ── Completeness: avg confidence of all parsed items ─────────────────────
    confidences = [float(p.get("confidence", 1.0)) for p in parsed]
    completeness = round(sum(confidences) / len(confidences) * 100) if confidences else 100

    # ── Sourcability: % of non-skipped items found on 3+ distributors ────────
    active = [i for i in bom_items if not i.skipped]
    if active:
        multi_dist = sum(
            1 for i in active
            if len({r.source for r in i.results}) >= 3
        )
        single_dist = sum(
            1 for i in active
            if 1 <= len({r.source for r in i.results}) < 3
        )
        not_found = sum(1 for i in active if len(i.results) == 0)
        # Weight: 3+ = full, 1-2 = half, 0 = zero
        sourcability = round(
            (multi_dist * 1.0 + single_dist * 0.5) / len(active) * 100
        )
    else:
        sourcability = 0

    # ── Price Spread: consistency of prices across distributors ──────────────
    spreads: list[float] = []
    for item in active:
        prices = [r.price for r in item.results if r.price and r.price > 0]
        if len(prices) >= 2:
            mn, mx = min(prices), max(prices)
            if mx > 0:
                spreads.append((mx - mn) / mx)  # 0 = identical, 1 = huge spread
    if spreads:
        avg_spread = sum(spreads) / len(spreads)
        price_spread = round(max(0, 100 - avg_spread * 100))
    else:
        price_spread = 85  # no multi-source data — neutral

    # ── Counterfeit + NRND: ask Gemini ───────────────────────────────────────
    item_names = [i.name for i in bom_items if not i.skipped]
    gemini_health = await _compute_flags_only(item_names)
    flags = gemini_health.flags
    counterfeit_score = gemini_health.counterfeit

    # ── Overall score: weighted average ──────────────────────────────────────
    overall = round(
        sourcability   * 0.30 +
        price_spread   * 0.15 +
        completeness   * 0.15 +
        recognition    * 0.25 +
        counterfeit_score * 0.15
    )

    return BOMHealth(
        score=overall,
        sourcability=sourcability,
        price_spread=price_spread,
        completeness=completeness,
        recognition=recognition,
        counterfeit=counterfeit_score,
        flags=flags,
    )


# ── Main handler ──────────────────────────────────────────────────────────────

async def handle_bom(req: BOMRequest) -> QueryResponse:
    log.info("[BOM] input: %s", req.text[:120])

    parsed = await _parse_bom_with_retry(req.text)

    if not parsed:
        return QueryResponse(
            intent=QueryIntent.BOM,
            query=req.text,
            answer="Could not parse the parts list. Please check the format.",
        )

    # Cap at 100 distinct items
    if len(parsed) > BOM_MAX_ITEMS:
        log.warning("[BOM] truncating %d items to %d", len(parsed), BOM_MAX_ITEMS)
        parsed = parsed[:BOM_MAX_ITEMS]

    log.info("[BOM] %d unique items after dedup", len(parsed))
    for i, item in enumerate(parsed):
        log.info(
            "[BOM] item %d: name=%r  qty=%s  confidence=%.2f  tokens=%s",
            i, item.get("name"), item.get("quantity"),
            item.get("confidence", 0), item.get("important_tokens"),
        )

    # ── Single batch embed for all unique queries across all items ────────────
    flat_queries:  list[str]        = []
    seen_queries:  dict[str, int]   = {}
    item_indices:  list[list[int]]  = []   # per item: indices into flat_queries

    for item in parsed:
        if float(item.get("confidence", 1.0)) < CONFIDENCE_THRESHOLD:
            item_indices.append([])
            continue
        name         = item.get("name", "")
        search_terms = item.get("search_terms") or [name]
        queries      = list(dict.fromkeys([name] + search_terms))
        idxs = []
        for q in queries:
            if q not in seen_queries:
                seen_queries[q] = len(flat_queries)
                flat_queries.append(q)
            idxs.append(seen_queries[q])
        item_indices.append(idxs)

    if flat_queries:
        log.info("[BOM] batch embedding %d unique queries for %d items", len(flat_queries), len(parsed))
        all_embeddings = await embed_batch(flat_queries)
        log.info("[BOM] batch embedding done")
    else:
        all_embeddings = []

    tasks = [
        _search_bom_item(
            item, req.limit_per_item, only_in_stock=True,
            embeddings=[all_embeddings[i] for i in idxs] if idxs else None,
        )
        for item, idxs in zip(parsed, item_indices)
    ]
    bom_items = await asyncio.gather(*tasks)

    # Compute health score concurrently with nothing else pending
    health = await _compute_health(list(bom_items), parsed)

    log.info("[BOM] done — %d items, health=%d", len(bom_items), health.score)
    return QueryResponse(
        intent=QueryIntent.BOM,
        query=req.text,
        bom_items=list(bom_items),
        bom_health=health,
    )


# ── Per-item search ───────────────────────────────────────────────────────────

async def _search_bom_item(
    item:        dict,
    limit:       int,
    only_in_stock: bool,
    embeddings:  list[list[float]] | None = None,
) -> BOMItem:
    name             = item.get("name", "")
    search_terms     = item.get("search_terms") or [name]
    important_tokens = item.get("important_tokens") or []
    quantity         = item.get("quantity") or 1
    confidence       = float(item.get("confidence", 1.0))

    base = BOMItem(
        part=name, name=name, qty=quantity,
        confidence=confidence,
    )

    if confidence < CONFIDENCE_THRESHOLD:
        log.info("[BOM] skipping %r (confidence=%.2f)", name, confidence)
        base.skipped = True
        return base

    log.info("[BOM] searching %r | tokens=%s", name, important_tokens)

    # Fetch max(limit*2, 10) candidates per query — ranked by closeness,
    # so no need to over-fetch; union across 5-6 queries gives rich pool
    vec_candidates = max(limit * 2, 10)

    # ── Part-number fast path: exact trigram first ────────────────────────────
    pinned: ProductResult | None = None
    if _is_part_number(name):
        exact = await trigram_search(name, limit=5, only_in_stock=only_in_stock)
        if exact and exact[0].score >= EXACT_MATCH_THRESHOLD:
            pinned = exact[0]
            log.info("[BOM] %r — exact match pinned: %r (score=%.3f)", name, pinned.product_name, pinned.score)

    # ── Vector search ─────────────────────────────────────────────────────────
    if embeddings is None:
        embeddings = await embed_batch(list(dict.fromkeys([name] + search_terms)))
        log.debug("[BOM] %r — embedded %d queries (fallback)", name, len(embeddings))

    vec_tasks = [
        vector_search(emb, limit=vec_candidates, only_in_stock=only_in_stock)
        for emb in embeddings
    ]
    vec_results_per_query: list[list[ProductResult]] = await asyncio.gather(*vec_tasks)

    vec_scores: dict[str, float] = {}
    vec_rows:   dict[str, ProductResult] = {}
    for results in vec_results_per_query:
        for r in results:
            if r.id not in vec_scores or r.score > vec_scores[r.id]:
                vec_scores[r.id] = r.score
                vec_rows[r.id]   = r

    # ── Trigram search on important_tokens ────────────────────────────────────
    keyword_query = " ".join(important_tokens) if important_tokens else name
    trgm_results  = await trigram_search(keyword_query, limit=vec_candidates, only_in_stock=only_in_stock)
    trgm_scores: dict[str, float] = {r.id: r.score for r in trgm_results}
    trgm_rows:   dict[str, ProductResult] = {r.id: r for r in trgm_results}

    log.debug("[BOM] %r — %d vec + %d trgm candidates", name, len(vec_scores), len(trgm_scores))

    # ── Union + token-penalised scoring ───────────────────────────────────────
    all_ids = set(vec_scores) | set(trgm_scores)
    scored: list[tuple[float, ProductResult]] = []

    for pid in all_ids:
        vs  = vec_scores.get(pid, 0.0)
        ks  = trgm_scores.get(pid, 0.0)
        row = vec_rows.get(pid) or trgm_rows[pid]
        base_score = VEC_WEIGHT * vs + TRGM_WEIGHT * ks
        penalty    = _token_match_factor(important_tokens, row)
        final      = base_score * penalty
        scored.append((final, row.model_copy(update={"score": round(final, 4)})))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Pin exact match to rank 1 if found
    if pinned:
        scored = [(s, r) for s, r in scored if r.id != pinned.id]
        scored.insert(0, (pinned.score, pinned))

    top = scored[:limit]
    log.info(
        "[BOM] %r — %d candidates, top %d (best=%.3f%s)",
        name, len(all_ids), len(top),
        top[0][0] if top else 0,
        " [pinned]" if pinned else "",
    )

    base.results = [r for _, r in top]
    return base
