"""
PartPilot — FastAPI entry point.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager

import asyncpg
import io
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from config import LOG_LEVEL
from db import close_pool, get_pool, init_pool
from handlers.bom import handle_bom
from handlers.circuit import handle_circuit
from handlers.compare import handle_compare
from handlers.lookup import handle_lookup
from handlers.semantic import handle_semantic
from models import BOMRequest, BOMHealth, HealthRequest, ProductResult, ProductVariant, QueryIntent, QueryRequest, QueryResponse, PartEnrichRequest, PartEnrichResponse, CategoryOverviewRequest, CategoryOverviewResponse, CategoryGroup
from router import classify
from search.trigram import _parse_raw_data, _row_to_result
from handlers.bom import _compute_flags_only

log = logging.getLogger(__name__)

# ── Timeouts ──────────────────────────────────────────────────────────────────
_QUERY_TIMEOUT = 30.0
_BOM_TIMEOUT   = 120.0

# ── BOM concurrency cap ───────────────────────────────────────────────────────
_BOM_CONCURRENCY = 5
_bom_semaphore: asyncio.Semaphore | None = None


# ── App setup ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _bom_semaphore
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    _bom_semaphore = asyncio.Semaphore(_BOM_CONCURRENCY)
    await init_pool()
    yield
    await close_pool()


app = FastAPI(
    title="PartPilot",
    description="Intelligent electronics parts search & assistant",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _handle_exc(e: Exception, context: str) -> None:
    if isinstance(e, asyncpg.TooManyConnectionsError):
        raise HTTPException(status_code=503, detail="Database busy — please retry shortly.")
    if isinstance(e, asyncio.TimeoutError):
        raise HTTPException(status_code=504, detail="Request timed out. Please try again.")
    log.error("%s error: %s", context, e, exc_info=True)
    raise HTTPException(status_code=500, detail="Internal server error.")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """
    Deep health check — verifies DB connectivity and Gemini reachability.
    Returns 200 only when both are healthy.
    """
    result: dict = {"status": "ok", "db": "ok", "gemini": "ok"}

    # DB check
    try:
        pool = get_pool()
        async with pool.acquire(timeout=3) as conn:
            await conn.fetchval("SELECT 1")
    except Exception as e:
        result["db"] = f"error: {e}"
        result["status"] = "degraded"

    # Gemini check — lightweight: just instantiate and check the client is alive
    # (avoid a real API call to save quota; connection errors surface on first real request)
    try:
        from gemini import _clients
        if not _clients:
            raise RuntimeError("No Gemini clients initialised")
    except Exception as e:
        result["gemini"] = f"error: {e}"
        result["status"] = "degraded"

    status_code = 200 if result["status"] == "ok" else 503
    from fastapi.responses import JSONResponse
    return JSONResponse(content=result, status_code=status_code)


@app.post("/api/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    async def _handle():
        intent, parts = await classify(req.query)
        match intent:
            case QueryIntent.PART_LOOKUP:
                return await handle_lookup(req.query, parts, req.limit, req.only_in_stock)
            case QueryIntent.COMPARE:
                return await handle_compare(req.query, parts, req.limit, req.only_in_stock)
            case QueryIntent.CIRCUIT_HELP:
                return await handle_circuit(req.query, req.limit, req.only_in_stock)
            case QueryIntent.SEMANTIC | QueryIntent.CATEGORY:
                return await handle_semantic(req.query, intent, req.limit, req.only_in_stock)
            case QueryIntent.BOM:
                return await handle_bom(BOMRequest(
                    text=req.query,
                    only_in_stock=req.only_in_stock,
                    limit_per_item=req.limit,
                ))
            case _:
                return await handle_semantic(req.query, QueryIntent.SEMANTIC, req.limit, req.only_in_stock)

    try:
        return await asyncio.wait_for(_handle(), timeout=_QUERY_TIMEOUT)
    except (HTTPException, asyncio.TimeoutError, asyncpg.TooManyConnectionsError):
        raise
    except Exception as e:
        _handle_exc(e, "query")


@app.post("/api/bom", response_model=QueryResponse)
async def bom(req: BOMRequest) -> QueryResponse:
    if _bom_semaphore._value == 0:
        raise HTTPException(
            status_code=429,
            detail=f"Server busy — max {_BOM_CONCURRENCY} concurrent BOM requests. Please retry shortly.",
        )

    async with _bom_semaphore:
        try:
            return await asyncio.wait_for(handle_bom(req), timeout=_BOM_TIMEOUT)
        except (HTTPException, asyncio.TimeoutError, asyncpg.TooManyConnectionsError):
            raise
        except Exception as e:
            _handle_exc(e, "bom")


@app.post("/api/bom/health", response_model=BOMHealth)
async def bom_health(req: HealthRequest) -> BOMHealth:
    """
    Recompute Gemini-based flags (counterfeit risk, NRND) for a list of part names.
    The 4 deterministic scores are computed client-side; this returns only the
    Gemini-derived fields so the frontend can merge them.
    """
    try:
        return await asyncio.wait_for(_compute_flags_only(req.names), timeout=20.0)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timed out.")
    except Exception as e:
        _handle_exc(e, "bom/health")


@app.get("/api/product/{product_id}", response_model=ProductResult)
async def get_product(product_id: str) -> ProductResult:
    """Direct product lookup by UUID — no AI, no search, pure DB fetch."""
    try:
        async with get_pool().acquire(timeout=10) as conn:
            row = await conn.fetchrow(
                """
                SELECT id, product_name, sku, price, source, product_url,
                       categories, brand, is_in_stock, description, ai_description, raw_data
                FROM scraped_data
                WHERE id = $1::uuid
                """,
                product_id,
            )
            if row is None:
                raise HTTPException(status_code=404, detail="Product not found.")

            variant_rows = await conn.fetch(
                """
                SELECT id, title, sku, price, is_available, attributes
                FROM product_variants
                WHERE scraped_data_id = $1::uuid
                ORDER BY price ASC NULLS LAST
                """,
                product_id,
            )
    except HTTPException:
        raise
    except asyncpg.TooManyConnectionsError:
        raise HTTPException(status_code=503, detail="Database busy — please retry shortly.")
    except Exception as e:
        log.error("product lookup error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error.")

    variants = [
        ProductVariant(
            id=str(v["id"]),
            title=v["title"],
            sku=v["sku"],
            price=float(v["price"]) if v["price"] is not None else None,
            is_available=v["is_available"],
            attributes=_parse_raw_data(v["attributes"]),
        )
        for v in variant_rows
    ]

    result = _row_to_result(row, score=1.0)
    result.variants = variants
    return result


# ── File extraction endpoint ──────────────────────────────────────────────────

_ALLOWED_MIME = {
    "image/jpeg", "image/png", "image/gif", "image/webp", "image/bmp", "image/tiff",
    "application/pdf",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "text/csv",
}

_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@app.post("/api/extract-file")
async def extract_file(file: UploadFile = File(...)) -> dict:
    """
    Extract text/BOM content from an uploaded file.
    Supports: images (JPEG/PNG/etc.), PDF, Excel (.xlsx/.xls), CSV.
    Returns {"text": "<extracted content>"}.
    """
    content_type = (file.content_type or "").split(";")[0].strip().lower()

    data = await file.read()
    if len(data) > _MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large (max 10 MB).")

    try:
        # ── Images → Vision API OCR ───────────────────────────────────────────
        if content_type.startswith("image/"):
            from vision import detect_text
            text = await detect_text(data, mime_type=content_type)
            if not text:
                raise HTTPException(status_code=422, detail="No text found in image.")
            return {"text": text}

        # ── PDF → Vision API (first page rendered as image via pdf2image) ─────
        if content_type == "application/pdf":
            try:
                import pdf2image  # type: ignore
                images = pdf2image.convert_from_bytes(data, dpi=200, fmt="jpeg")
            except Exception as e:
                raise HTTPException(status_code=422, detail=f"Could not read PDF: {e}")

            from vision import detect_text
            pages_text: list[str] = []
            for img in images:
                buf = io.BytesIO()
                img.save(buf, format="JPEG")
                page_text = await detect_text(buf.getvalue(), mime_type="image/jpeg")
                if page_text:
                    pages_text.append(page_text)

            text = "\n".join(pages_text).strip()
            if not text:
                raise HTTPException(status_code=422, detail="No text found in PDF.")
            return {"text": text}

        # ── Excel (.xlsx / .xls) → openpyxl / xlrd ───────────────────────────
        if content_type in (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-excel",
        ) or (file.filename or "").lower().endswith((".xlsx", ".xls")):
            try:
                import openpyxl  # type: ignore
                wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
                rows: list[str] = []
                for ws in wb.worksheets:
                    for row in ws.iter_rows(values_only=True):
                        cells = [str(c).strip() for c in row if c is not None and str(c).strip()]
                        if cells:
                            rows.append(", ".join(cells))
                text = "\n".join(rows).strip()
            except Exception as e:
                raise HTTPException(status_code=422, detail=f"Could not read Excel file: {e}")
            if not text:
                raise HTTPException(status_code=422, detail="Excel file appears empty.")
            return {"text": text}

        # ── CSV ───────────────────────────────────────────────────────────────
        if content_type == "text/csv" or (file.filename or "").lower().endswith(".csv"):
            try:
                text = data.decode("utf-8", errors="replace").strip()
            except Exception as e:
                raise HTTPException(status_code=422, detail=f"Could not read CSV: {e}")
            if not text:
                raise HTTPException(status_code=422, detail="CSV file appears empty.")
            return {"text": text}

        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {content_type}. Use image, PDF, Excel, or CSV.",
        )

    except HTTPException:
        raise
    except Exception as e:
        log.error("extract-file error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="File extraction failed.")


# ── AI description endpoint ───────────────────────────────────────────────────

_AI_DESC_SYSTEM = """You are a technical writer for an electronics component store.
Paraphrase the given product description to be clear, helpful, and engaging for electronics hobbyists and engineers.
- Preserve ALL information from the original — do not omit any specs, features, or details
- Rephrase every part of the original content; do not summarize or condense
- Use plain English, no marketing fluff
- Keep the same level of detail and length as the original
- Return only the paraphrased description, no preamble"""


@app.post("/api/product/{product_id}/ai-description")
async def generate_ai_description(product_id: str) -> dict:
    """
    Generate (or return cached) AI-paraphrased description for a product.
    On first call: generates via Gemini and saves to scraped_data.ai_description.
    On subsequent calls: returns the cached value immediately.
    """
    try:
        async with get_pool().acquire(timeout=10) as conn:
            row = await conn.fetchrow(
                "SELECT description, ai_description FROM scraped_data WHERE id = $1::uuid",
                product_id,
            )
        if row is None:
            raise HTTPException(status_code=404, detail="Product not found.")

        # Return cached if already generated
        if row["ai_description"]:
            return {"ai_description": row["ai_description"]}

        # Nothing to paraphrase
        if not row["description"] or not row["description"].strip():
            return {"ai_description": None}

        # Generate via Gemini
        from gemini import generate as gemini_generate
        ai_desc = await gemini_generate(row["description"], system=_AI_DESC_SYSTEM)
        ai_desc = ai_desc.strip()

        # Persist
        async with get_pool().acquire(timeout=10) as conn:
            await conn.execute(
                "UPDATE scraped_data SET ai_description = $1 WHERE id = $2::uuid",
                ai_desc, product_id,
            )

        return {"ai_description": ai_desc}

    except HTTPException:
        raise
    except Exception as e:
        log.error("ai-description error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate description.")


# ── Part enrichment endpoint ──────────────────────────────────────────────────

def _fix_truncated_json(raw: str) -> str:
    """Fix truncated JSON strings by closing unclosed strings, brackets, and braces in the correct order."""
    import re as _re
    raw = _re.sub(r",\s*([}\]])", r"\1", raw)
    raw = _re.sub(r"(?<![\\])'", '"', raw)
    
    escaped = False
    in_string = False
    stack = []
    
    for c in raw:
        if c == '\\' and not escaped:
            escaped = True
            continue
        elif c == '"' and not escaped:
            in_string = not in_string
            escaped = False
            continue
            
        if not in_string:
            if c == '{':
                stack.append('}')
            elif c == '[':
                stack.append(']')
            elif c == '}':
                if stack and stack[-1] == '}':
                    stack.pop()
            elif c == ']':
                if stack and stack[-1] == ']':
                    stack.pop()
        escaped = False
            
    if in_string:
        raw += '"'
        
    raw = raw.rstrip().rstrip(",")
    
    while stack:
        raw += stack.pop()
        
    return raw


def _extract_partial_json(raw: str) -> dict:
    """
    Best-effort extraction of top-level string/list fields from truncated JSON.
    Tries each top-level key individually so a truncated 'specs' doesn't kill 'good_for'.
    """
    import re as _re2
    result = {}
    # Try to extract each known field independently
    for field in ("specs", "good_for", "watch_out", "external_needed", "alts", "counterfeit_note"):
        # Find the field's value start
        m = _re2.search(rf'"{field}"\s*:\s*', raw)
        if not m:
            continue
        val_start = m.end()
        val_raw = raw[val_start:].strip()
        # Try to parse just this value by finding its natural end
        for end_offset in range(len(val_raw), 0, -1):
            try:
                val = json.loads(val_raw[:end_offset])
                result[field] = val
                break
            except (json.JSONDecodeError, ValueError):
                continue
    return result

_ENRICH_SYSTEM = """You are an expert electronics engineer. Given a component's name, description, and raw attributes, extract structured information.

Respond ONLY with valid JSON (no markdown, no preamble):
{
  "specs": {"Spec name": "value"},
  "good_for": ["use case 1", "use case 2"],
  "watch_out": ["caution 1"],
  "external_needed": "components needed or null",
  "alts": [{"name": "part", "note": "why", "price_hint": "₹X–Y or null"}],
  "counterfeit_note": "note or null"
}

Rules:
- specs: exactly 4–5 key electrical specs (voltage, current, package, frequency). Short keys, concise values.
- good_for: 2–3 practical use cases as short phrases
- watch_out: 1–2 real cautions. Empty list if none.
- external_needed: only for ICs that need external passives. null for modules.
- alts: 1–2 real alternative part numbers. Empty list if unknown.
- counterfeit_note: only if commonly counterfeited. null otherwise.
- All string values must use double quotes. No trailing commas.
"""


@app.post("/api/part/enrich", response_model=PartEnrichResponse)
async def enrich_part(req: PartEnrichRequest) -> PartEnrichResponse:
    """
    AI-extract structured specs, use cases, cautions, and alternatives for a part.
    Uses cached ai_description if available, otherwise generates fresh.
    """
    import json, re as _re
    from gemini import generate as gemini_generate

    # Build context from all available data
    attrs_text = ""
    if req.raw_data.get("attributes"):
        attrs_text = "\n".join(f"  {k} {v}" for k, v in req.raw_data["attributes"].items())
    elif req.raw_data.get("price_tiers"):
        tiers = req.raw_data["price_tiers"]
        attrs_text = "Price tiers: " + ", ".join(
            f"₹{t['price']} (qty {t['min_qty']}+)" for t in tiers
        )

    cats = [c for c in req.categories if not c.startswith("[C]") and not c.startswith("_") and len(c) > 2]
    context = f"""Part: {req.product_name}
Categories: {', '.join(cats) if cats else 'unknown'}
Description: {req.description or 'not available'}
Attributes:
{attrs_text or 'not available'}"""

    try:
        raw = await asyncio.wait_for(
            gemini_generate(context, system=_ENRICH_SYSTEM),
            timeout=20.0,
        )
        raw = _re.sub(r"```(?:json)?|```", "", raw).strip()
        raw = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL).strip()
        start = raw.find("{")
        if start != -1:
            raw = raw[start:]
        raw = _fix_truncated_json(raw)
        data = json.loads(raw)
        return PartEnrichResponse(
            specs=data.get("specs") or {},
            good_for=data.get("good_for") or [],
            watch_out=data.get("watch_out") or [],
            external_needed=data.get("external_needed"),
            alts=data.get("alts") or [],
            counterfeit_note=data.get("counterfeit_note"),
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Enrichment timed out.")
    except (json.JSONDecodeError, ValueError) as e:
        log.warning("enrich JSON parse failed (%s), attempting field extraction | raw=%r", e, locals().get("raw", ""))
        # Best-effort: extract whatever fields parsed before truncation
        data = _extract_partial_json(locals().get("raw", ""))
        return PartEnrichResponse(
            specs=data.get("specs") or {},
            good_for=data.get("good_for") or [],
            watch_out=data.get("watch_out") or [],
            external_needed=data.get("external_needed"),
            alts=data.get("alts") or [],
            counterfeit_note=data.get("counterfeit_note"),
        )
    except Exception as e:
        log.error("enrich error: %s | raw=%r", e, locals().get("raw", ""), exc_info=True)
        raise HTTPException(status_code=500, detail="Enrichment failed.")


# ── Category overview endpoint ────────────────────────────────────────────────

_CATEGORY_SYSTEM = """You are an expert electronics engineer. Given a list of products in a category, group them into meaningful sub-types.

Respond ONLY with valid JSON (no markdown, no preamble):
{
  "title": "clean category name (title case)",
  "subtitle": "short tagline e.g. '6 types stocked'",
  "groups": [
    {
      "name": "Sub-type name",
      "description": "One sentence: what it is and when to use it.",
      "product_names": ["exact product name 1", "exact product name 2", ...]
    }
  ]
}

Rules:
- Create 3–8 meaningful groups based on actual sub-types (e.g. H-Bridge ICs, Stepper Drivers, BLDC Drivers)
- Each group must contain only product names from the provided list — no invented names
- description must be practical and specific, not generic
- title should be clean (e.g. "Motor Drivers" not "types of motor drivers")
- subtitle: "{N} types stocked — click any category to see parts and prices"
"""


@app.post("/api/category/overview", response_model=CategoryOverviewResponse)
async def category_overview(req: CategoryOverviewRequest) -> CategoryOverviewResponse:
    """
    AI-group a list of products into meaningful sub-categories with descriptions.
    """
    import json, re as _re
    from gemini import generate as gemini_generate

    products_text = "\n".join(
        f"- {p.product_name} | cats: {', '.join(c for c in p.categories if not c.startswith('[C]') and not c.startswith('_'))}"
        + (f" | {p.description[:120]}..." if p.description else "")
        for p in req.products[:30]
    )
    prompt = f"Query: {req.query}\n\nProducts:\n{products_text}"

    try:
        raw = await asyncio.wait_for(
            gemini_generate(prompt, system=_CATEGORY_SYSTEM),
            timeout=45.0,
        )
        raw = _re.sub(r"```(?:json)?|```", "", raw).strip()
        raw = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL).strip()
        # Find outermost JSON object start — don't trim at rfind("}") since output may be truncated
        start = raw.find("{")
        if start != -1:
            raw = raw[start:]
        raw = _fix_truncated_json(raw)
        data = json.loads(raw)

        # Build a name→product map for fast lookup
        name_map = {p.product_name: p for p in req.products}

        groups = []
        for g in data.get("groups", []):
            pnames = g.get("product_names", [])
            # Only keep names that actually exist in our product list
            valid = [n for n in pnames if n in name_map]
            if not valid:
                # fuzzy fallback: match by prefix
                valid = [p.product_name for p in req.products
                         if any(n.lower() in p.product_name.lower() or p.product_name.lower() in n.lower()
                                for n in pnames)][:3]
            groups.append(CategoryGroup(
                name=g.get("name", "Other"),
                description=g.get("description", ""),
                count=len(valid),
                examples=valid[:3],
            ))

        return CategoryOverviewResponse(
            title=data.get("title", req.query),
            subtitle=data.get("subtitle", f"{len(groups)} types stocked"),
            groups=groups,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Category overview timed out.")
    except (json.JSONDecodeError, ValueError) as e:
        log.warning("category overview JSON parse failed (%s) | raw=%r", e, locals().get("raw", ""))
        # Return a minimal fallback grouping by cleaned category tags
        from collections import defaultdict
        fallback: dict[str, list] = defaultdict(list)
        for p in req.products:
            cats = [c for c in p.categories if not c.startswith("[C]") and not c.startswith("_") and len(c) > 2]
            key = cats[0] if cats else "Other"
            fallback[key].append(p.product_name)
        groups = [
            CategoryGroup(name=k, description="", count=len(v), examples=v[:3])
            for k, v in list(fallback.items())[:8]
        ]
        catname = req.query.replace(r"^(types of|browse|show me|list|all)\s+", "").strip().title()
        return CategoryOverviewResponse(
            title=catname,
            subtitle=f"{len(groups)} types stocked",
            groups=groups,
        )
    except Exception as e:
        log.error("category overview error: %s | raw=%r", e, locals().get("raw", ""), exc_info=True)
        raise HTTPException(status_code=500, detail="Category overview failed.")
