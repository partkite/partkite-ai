# PartPilot

Intelligent electronics parts search API for Indian component stores. Understands natural language queries, part numbers, BOMs, circuit questions, and comparisons — powered by Gemini and pgvector.

---

## What it does

Send a natural language query, get back matched products from 6 Indian electronics stores (Robu, Evelta, MakerBazar, Quartz Components, Sunrom, ElectronicSpices) with prices, stock status, and links.

Examples of what it handles:

- `"LM2596"` → exact part lookup across all stores
- `"convert 12V to 5V at 1A"` → semantic search by function
- `"LM7805 vs AMS1117"` → side-by-side comparison with Gemini analysis
- `"what resistor for a 5mm LED at 5V"` → circuit calculation + part suggestions
- `"types of motor drivers"` → category browse
- A messy BOM pasted as text → parsed and searched item by item

---

## Architecture

```
POST /api/query
      │
      ▼
  classify()          ← Gemini Flash classifies intent + extracts part names
      │
      ├─ PART_LOOKUP  → handle_lookup   → trigram → hybrid fallback
      ├─ SEMANTIC     → handle_semantic → hybrid (80% vector, 20% trigram) + LLM answer
      ├─ CATEGORY     → handle_semantic → hybrid (80% vector, 20% trigram) + LLM answer
      ├─ COMPARE      → handle_compare  → trigram per part (parallel) + Gemini comparison
      ├─ CIRCUIT_HELP → handle_circuit  → Gemini answer → hybrid search on suggested term
      └─ BOM          → handle_bom      → Gemini parse → trigram/hybrid per item (parallel)
```

---

## Search pipeline

Three search strategies, selected per intent:

### Trigram search
Uses PostgreSQL `pg_trgm` for fuzzy string matching. Best for exact part numbers and SKUs (`BC547`, `IRF540N`). Calls the `trigram_search()` stored procedure.

### Vector search
Uses `pgvector` cosine similarity against 768-dim embeddings stored in `scraped_data.embedding`. Backed by an HNSW index (`m=16, ef_construction=64`). Best for functional/descriptive queries.

### Hybrid search (RRF)
Combines both via Reciprocal Rank Fusion in the `hybrid_search()` stored procedure. Default weights: 30% trigram + 70% vector. Falls back to trigram-only if embeddings aren't populated yet.

**Strategy by intent:**

| Intent | Strategy |
|---|---|
| PART_LOOKUP | Trigram first, hybrid if results thin |
| SEMANTIC / CATEGORY | Hybrid (0.2 trgm / 0.8 vec) |
| COMPARE | Trigram per part, parallel |
| CIRCUIT_HELP | Hybrid on Gemini-suggested search term |
| BOM | Trigram first per item, hybrid supplement |

---

## Embedding pipeline

Products are embedded as: `product_name | brand | category1 | category2 | description`

Bulk embedding job (`partpilot/embeddings/generate.py`):

- Producer fetches all rows with `embedding IS NULL`, streams in batches
- N workers (default ~10), each with a dedicated `asyncpg` connection (no shared pool — avoids pgbouncer contention)
- Token bucket rate limiter at 200 RPM (70% of Gemini tier limit)
- Exponential backoff on 429/503/quota errors, up to 6 retries per batch
- Writes via `UPDATE ... FROM unnest(vecs, ids)` — single round-trip per batch
- Reconnects automatically on dropped connections

```bash
python -m partpilot.embeddings.generate
python -m partpilot.embeddings.generate --batch-size 100 --workers 10
```

New rows scraped after initial setup are embedded inline by the scraper — no need to run the bulk job again.

---

## Gemini integration

`partpilot/gemini.py` wraps all Gemini calls with:

- **Round-robin key rotation** — set `GEMINI_API_KEY=key1,key2,key3` for multiple keys
- **Automatic retry** on 429/503, rotating to the next key each attempt
- **Two embedding task types**: `RETRIEVAL_DOCUMENT` for products, `RETRIEVAL_QUERY` for search queries (asymmetric embedding — improves recall)
- **Single-turn generation** for intent classification, comparisons, circuit help, and semantic summaries

Models used:
- Chat/generation: `gemma-3-27b-it` (configurable via `GEMINI_CHAT_MODEL`)
- Embeddings: `gemini-embedding-001` at 768 dimensions (configurable via `GEMINI_EMBED_MODEL`)

---

## Database

Supabase (PostgreSQL) with:

- `pgvector` — vector storage and HNSW index
- `pg_trgm` — trigram GIN indexes on `product_name` and `sku`
- `asyncpg` connection pool (min 5, max 20) configured for pgbouncer transaction mode (`statement_cache_size=0`)
- Three stored procedures: `trigram_search()`, `vector_search()`, `hybrid_search()`

Key table: `scraped_data`

| Column | Type | Notes |
|---|---|---|
| id | uuid | PK |
| source | text | e.g. `robu.in` |
| source_id | text | site's own product ID |
| product_name | text | trigram indexed |
| sku | text | trigram indexed |
| brand | text | |
| description | text | |
| categories | text[] | |
| price | numeric | |
| is_in_stock | bool | partial index |
| embedding | vector(768) | HNSW indexed |
| raw_data | jsonb | full original payload |
| scraped_at | timestamptz | |

---

## API

Base URL: `http://localhost:8000`

---

### `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{ "status": "ok" }
```

---

### `POST /api/query`

The main endpoint. Handles all query types — intent is classified automatically.

**Parameters:**

| Field | Type | Default | Description |
|---|---|---|---|
| `query` | string | required | Natural language query, part number, or comparison |
| `only_in_stock` | bool | `false` | Filter to in-stock products only |
| `limit` | int (1–50) | `10` | Max results to return |

---

**Part lookup** — exact part number or name:

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "LM2596 buck converter IC"}'
```

```json
{
  "intent": "PART_LOOKUP",
  "query": "LM2596 buck converter IC",
  "results": [
    {
      "id": "3f2a...",
      "product_name": "LM2596S-ADJ DC-DC Buck Converter IC",
      "sku": "LM2596S-ADJ",
      "price": 35.0,
      "source": "robu.in",
      "product_url": "https://robu.in/product/lm2596s-adj/",
      "categories": ["ICs", "Power Management"],
      "brand": "Texas Instruments",
      "is_in_stock": true,
      "raw_data": {},
      "score": 0.97
    }
  ],
  "answer": null,
  "bom_items": []
}
```

---

**Semantic search** — describe what you need:

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "step down converter 12V to 5V at 2A", "only_in_stock": true}'
```

```json
{
  "intent": "SEMANTIC",
  "query": "step down converter 12V to 5V at 2A",
  "results": [ ... ],
  "answer": "The LM2596-based modules are the best match for your requirement. They support input voltages up to 40V and can deliver up to 3A output, making them suitable for 12V→5V at 2A. The XL4016 modules handle higher currents if you need headroom.",
  "bom_items": []
}
```

---

**Compare** — two or more parts:

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "LM7805 vs AMS1117 3.3V"}'
```

```json
{
  "intent": "COMPARE",
  "query": "LM7805 vs AMS1117 3.3V",
  "results": [ ... ],
  "answer": "LM7805: Fixed 5V output, up to 1.5A, requires heatsink above 1A, dropout ~2V. Best for 7V+ input rails.\n\nAMS1117-3.3: Fixed 3.3V, 1A max, low dropout (~1.3V), no heatsink needed for light loads. Best for 5V→3.3V regulation in microcontroller circuits.\n\nRecommendation: Use AMS1117-3.3 for 3.3V MCU power from a 5V rail. Use LM7805 for 5V from a 9–12V supply.",
  "bom_items": []
}
```

---

**Circuit help** — calculations and component values:

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "what resistor do I need for a red 5mm LED at 5V?"}'
```

```json
{
  "intent": "CIRCUIT_HELP",
  "query": "what resistor do I need for a red 5mm LED at 5V?",
  "results": [
    {
      "product_name": "330 Ohm Resistor (pack of 50)",
      "sku": "RES-330",
      "price": 15.0,
      "source": "robu.in",
      ...
    }
  ],
  "answer": "For a red 5mm LED (Vf ≈ 2.0V, If = 20mA):\n\nR = (Vs - Vf) / If = (5 - 2.0) / 0.020 = 150Ω\n\nUse the nearest standard value: 150Ω or 180Ω.\n\nSuggested search term: 150 ohm resistor",
  "bom_items": []
}
```

---

**Category browse:**

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "types of motor driver ICs", "limit": 20}'
```

```json
{
  "intent": "CATEGORY",
  "query": "types of motor driver ICs",
  "results": [ ... ],
  "answer": "The top motor driver ICs available are the L298N (dual H-bridge, up to 2A per channel), DRV8833 (low-voltage, 1.5A), TB6612FNG (efficient alternative to L298N), and A4988/DRV8825 for stepper motors.",
  "bom_items": []
}
```

---

### `POST /api/bom`

Dedicated BOM endpoint. Accepts a messy multi-item parts list and returns matched products per line item.

**Parameters:**

| Field | Type | Default | Description |
|---|---|---|---|
| `text` | string | required | Raw BOM text, any format |
| `only_in_stock` | bool | `false` | Filter to in-stock products only |
| `limit_per_item` | int (1–20) | `5` | Max results per BOM line item |

```bash
curl -X POST http://localhost:8000/api/bom \
  -H "Content-Type: application/json" \
  -d '{
    "text": "2x ESP32-WROOM-32D\n10x 10k resistor 0402\n5x 100nF 0402 cap\n1x AMS1117-3.3\n3x BC547 NPN transistor",
    "only_in_stock": true,
    "limit_per_item": 3
  }'
```

```json
{
  "intent": "BOM",
  "query": "2x ESP32-WROOM-32D\n10x 10k resistor...",
  "results": [],
  "answer": null,
  "bom_items": [
    {
      "part": "ESP32-WROOM-32D",
      "qty": 2,
      "spec": null,
      "results": [
        {
          "product_name": "ESP32-WROOM-32D WiFi+BT Module",
          "sku": "ESP32-WROOM-32D",
          "price": 349.0,
          "source": "robu.in",
          "is_in_stock": true,
          "score": 0.98,
          ...
        }
      ]
    },
    {
      "part": "10k resistor",
      "qty": 10,
      "spec": "0402",
      "results": [ ... ]
    },
    {
      "part": "100nF capacitor",
      "qty": 5,
      "spec": "0402",
      "results": [ ... ]
    },
    {
      "part": "AMS1117-3.3",
      "qty": 1,
      "spec": null,
      "results": [ ... ]
    },
    {
      "part": "BC547 NPN transistor",
      "qty": 3,
      "spec": null,
      "results": [ ... ]
    }
  ]
}
```

---

## Setup

```bash
cd partpilot
cp .env.example .env
# fill in .env

pip install -r requirements.txt
uvicorn partpilot.main:app --reload
```

Run the bulk embedding job once after initial scrape:

```bash
python -m partpilot.embeddings.generate --batch-size 100 --workers 10
```

Build the HNSW index (run once, takes 10–20 min for 100k+ rows):

```bash
python - << 'EOF'
import asyncio, asyncpg
from partpilot.config import POSTGRES_DSN

async def main():
    conn = await asyncpg.connect(dsn=POSTGRES_DSN, statement_cache_size=0)
    await conn.execute("SET statement_timeout = 0")
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS scraped_data_embedding_hnsw_idx
        ON public.scraped_data
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)
    print("Done")
    await conn.close()

asyncio.run(main())
EOF
```

---

## Project structure

```
partpilot/
├── main.py              # FastAPI app, lifespan, endpoints
├── router.py            # Gemini intent classifier
├── models.py            # Pydantic schemas
├── config.py            # Env vars
├── db.py                # asyncpg pool
├── gemini.py            # Gemini client with key rotation
├── handlers/
│   ├── lookup.py        # PART_LOOKUP
│   ├── semantic.py      # SEMANTIC + CATEGORY
│   ├── compare.py       # COMPARE
│   ├── circuit.py       # CIRCUIT_HELP
│   └── bom.py           # BOM
├── search/
│   ├── trigram.py       # pg_trgm search
│   ├── vector.py        # pgvector search
│   └── hybrid.py        # RRF hybrid search
└── embeddings/
    └── generate.py      # Bulk embedding job
```
