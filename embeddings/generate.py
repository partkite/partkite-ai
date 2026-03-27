"""
Parallel bulk embedding job.

Each worker owns a dedicated asyncpg connection — no shared pool, no pgbouncer
contention. Producer fetches all IDs upfront then streams row data in chunks.

Run:
  python -m partpilot.embeddings.generate
  python -m partpilot.embeddings.generate --batch-size 100 --workers 10
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import time
from datetime import datetime

import asyncpg
from google import genai
from google.genai import types as gtypes
from tqdm import tqdm

from config import GEMINI_API_KEY, GEMINI_EMBED_MODEL, POSTGRES_DSN

log = logging.getLogger(__name__)

RPM_SAFE  = 200   # 70% of tier limit — prevents cascading 429s
_SENTINEL = None

_CONN_KWARGS = dict(
    statement_cache_size=0,   # required for pgbouncer transaction mode
    command_timeout=120,
)


async def _get_conn() -> asyncpg.Connection:
    """Open a fresh direct connection with retry."""
    for attempt in range(5):
        try:
            return await asyncpg.connect(dsn=POSTGRES_DSN, **_CONN_KWARGS)
        except Exception as e:
            wait = 2 ** attempt
            tqdm.write(f"[connect] attempt {attempt+1} failed: {e} — retry in {wait}s")
            await asyncio.sleep(wait)
    raise RuntimeError("Could not connect to database after 5 attempts")


# ── Token bucket ──────────────────────────────────────────────────────────────

class TokenBucket:
    def __init__(self, rate: float, capacity: float):
        self.rate     = rate
        self.capacity = capacity
        self._tokens  = capacity
        self._last    = time.monotonic()
        self._lock    = asyncio.Lock()

    async def acquire(self):
        wait = 0.0
        async with self._lock:
            now = time.monotonic()
            self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
            self._last = now
            if self._tokens < 1:
                wait = (1 - self._tokens) / self.rate
                self._tokens -= 1
            else:
                self._tokens -= 1
        if wait > 0:
            await asyncio.sleep(wait)


# ── Worker state ──────────────────────────────────────────────────────────────

class WorkerState:
    def __init__(self, n: int):
        self.last_at: list[float | None] = [None] * n
        self.rtt:     list[float]        = [0.0]  * n
        self.status:  list[str]          = ["init"] * n
        self._lock = asyncio.Lock()

    async def update(self, wid: int, status: str, rtt: float | None = None):
        async with self._lock:
            self.status[wid] = status
            if rtt is not None:
                self.rtt[wid]     = rtt
                self.last_at[wid] = time.monotonic()

    def summary(self) -> str:
        now   = time.monotonic()
        parts = []
        for i, (last, rtt, st) in enumerate(zip(self.last_at, self.rtt, self.status)):
            if st == "init":
                continue
            ago = f"{now - last:.0f}s" if last else "?"
            parts.append(f"W{i}:{st}({ago},{rtt:.1f}s)")
        return " ".join(parts) if parts else "starting..."


# ── Document builder ──────────────────────────────────────────────────────────

def _build_doc(row: asyncpg.Record) -> str:
    parts = [row["product_name"] or ""]
    if row["brand"]:
        parts.append(row["brand"])
    if row["categories"]:
        parts.extend(row["categories"])
    if row["description"]:
        parts.append(row["description"])
    return " | ".join(p for p in parts if p)


# ── Producer ──────────────────────────────────────────────────────────────────

async def _producer(queue: asyncio.Queue, batch_size: int, n_workers: int):
    """
    Dedicated connection. Fetches all pending IDs first (fast, ~2MB),
    then streams full row data in batch_size chunks.
    """
    conn = await _get_conn()
    try:
        id_rows = await conn.fetch(
            "SELECT id FROM scraped_data WHERE embedding IS NULL ORDER BY scraped_at DESC"
        )
        all_ids = [r["id"] for r in id_rows]
        tqdm.write(f"Producer: {len(all_ids):,} IDs fetched")

        for i in range(0, len(all_ids), batch_size):
            chunk = all_ids[i : i + batch_size]
            rows = await conn.fetch(
                """
                SELECT id, product_name, description, brand, categories
                FROM scraped_data
                WHERE id = ANY($1::uuid[])
                """,
                chunk,
            )
            await queue.put(rows)
    finally:
        await conn.close()

    for _ in range(n_workers):
        await queue.put(_SENTINEL)


# ── Worker ────────────────────────────────────────────────────────────────────

async def _worker(
    wid:      int,
    client:   genai.Client,
    queue:    asyncio.Queue,
    bucket:   TokenBucket,
    bar:      tqdm,
    counters: list,
    ws:       WorkerState,
):
    from google.genai.types import Content, Part

    # Each worker owns its own persistent DB connection — zero pool contention
    conn = await _get_conn()

    try:
        while True:
            rows = await queue.get()
            if rows is _SENTINEL:
                await ws.update(wid, "done")
                queue.task_done()
                break

            docs     = [_build_doc(r) for r in rows]
            contents = [Content(parts=[Part(text=t)]) for t in docs]

            await ws.update(wid, "rate-wait")
            await bucket.acquire()
            await ws.update(wid, "embedding")

            t0 = time.monotonic()
            embeddings = None

            for attempt in range(6):
                try:
                    resp = await client.aio.models.embed_content(
                        model=GEMINI_EMBED_MODEL,
                        contents=contents,
                        config=gtypes.EmbedContentConfig(
                            task_type="RETRIEVAL_DOCUMENT",
                            output_dimensionality=768,
                        ),
                    )
                    embeddings = [e.values for e in resp.embeddings]
                    break
                except Exception as e:
                    msg  = str(e)
                    wait = 2 ** (attempt + 1)
                    ts   = datetime.now().strftime("%H:%M:%S")
                    if any(x in msg for x in ("429", "503", "quota", "RESOURCE_EXHAUSTED")):
                        tqdm.write(f"\n[{ts}] W{wid} RATE_LIMIT attempt {attempt+1}/6 — backoff {wait}s")
                        await ws.update(wid, f"backoff-{wait}s")
                        await asyncio.sleep(wait)
                    else:
                        tqdm.write(f"\n[{ts}] W{wid} EMBED ERROR: {msg[:150]}")
                        break

            rtt = time.monotonic() - t0
            await ws.update(wid, "writing", rtt=rtt)

            if embeddings is None:
                counters[1] += len(rows)
                bar.update(len(rows))
                queue.task_done()
                continue

            # Write all embeddings in one batch (fast now that HNSW index is dropped)
            ids  = [rows[i]["id"] for i in range(len(embeddings))]
            vecs = ["[" + ",".join(str(v) for v in emb) + "]" for emb in embeddings]

            for db_attempt in range(3):
                try:
                    await conn.execute(
                        """
                        UPDATE scraped_data AS sd
                        SET    embedding = v.vec::vector
                        FROM   unnest($1::text[], $2::uuid[]) AS v(vec, id)
                        WHERE  sd.id = v.id
                        """,
                        vecs, ids,
                    )
                    counters[0] += len(rows)
                    break
                except (asyncpg.PostgresConnectionError,
                        asyncpg.InterfaceError,
                        asyncio.TimeoutError,
                        OSError) as e:
                    ts = datetime.now().strftime("%H:%M:%S")
                    tqdm.write(f"\n[{ts}] W{wid} reconnecting (attempt {db_attempt+1}): {e}")
                    try:
                        await conn.close()
                    except Exception:
                        pass
                    conn = await _get_conn()
                except Exception as e:
                    ts = datetime.now().strftime("%H:%M:%S")
                    tqdm.write(f"\n[{ts}] W{wid} DB ERROR: {type(e).__name__}: {e}")
                    counters[1] += len(rows)
                    break

            bar.update(len(rows))
            await ws.update(wid, "idle")
            queue.task_done()
    finally:
        await conn.close()


# ── Status printer ────────────────────────────────────────────────────────────

async def _status_printer(ws: WorkerState, counters: list, stop: asyncio.Event, bar: tqdm):
    log_path = "partpilot/embeddings/worker_status.txt"
    while not stop.is_set():
        await asyncio.sleep(3)
        ts   = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] done={counters[0]:,} fail={counters[1]} | {ws.summary()}\n"
        with open(log_path, "a") as f:
            f.write(line)
        bar.set_postfix_str(f"done={counters[0]:,} fail={counters[1]}")


# ── Entry point ───────────────────────────────────────────────────────────────

async def run(batch_size: int = 100, n_workers: int | None = None) -> None:
    key    = GEMINI_API_KEY.split(",")[0].strip()
    client = genai.Client(api_key=key)

    req_per_sec         = RPM_SAFE / 60
    workers             = n_workers or max(1, int(req_per_sec * 3) + 2)
    assumed_rtt         = 3.0
    worker_capacity_rpm = workers * (60 / assumed_rtt)
    effective_rpm       = min(RPM_SAFE, worker_capacity_rpm)

    print(f"Batch size: {batch_size} | Workers: {workers} | "
          f"Rate cap: {RPM_SAFE} req/min | Effective: {effective_rpm:.0f} req/min "
          f"→ ~{effective_rpm * batch_size:,.0f} rows/min")

    # One throw-away connection just for the row count
    _c = await _get_conn()
    total = await _c.fetchval("SELECT COUNT(*) FROM scraped_data WHERE embedding IS NULL")
    await _c.close()

    print(f"Rows needing embeddings: {total:,}")
    if total == 0:
        print("All rows already embedded.")
        return

    eta_min = total / (effective_rpm * batch_size) if effective_rpm > 0 else 0
    print(f"ETA (at {assumed_rtt}s RTT): ~{eta_min:.1f} min")

    queue    = asyncio.Queue(maxsize=workers * 4)
    bucket   = TokenBucket(rate=req_per_sec, capacity=float(workers))
    counters = [0, 0]
    ws       = WorkerState(workers)
    stop_evt = asyncio.Event()

    with tqdm(total=total, unit="row", dynamic_ncols=True) as bar:
        producer = asyncio.create_task(_producer(queue, batch_size, workers))
        printer  = asyncio.create_task(_status_printer(ws, counters, stop_evt, bar))
        tasks    = [
            asyncio.create_task(_worker(i, client, queue, bucket, bar, counters, ws))
            for i in range(workers)
        ]
        await asyncio.gather(producer, *tasks)
        stop_evt.set()
        await printer

    print(f"\nDone. Embedded: {counters[0]:,} | Failed: {counters[1]:,}")

    # Recreate HNSW index (was dropped for fast bulk writes)
    if counters[0] > 0:
        print("Recreating HNSW index (this may take a few minutes)...")
        idx_conn = await asyncpg.connect(dsn=POSTGRES_DSN, statement_cache_size=0)  # no command_timeout — index build can take 10+ min
        try:
            await idx_conn.execute(
                """
                CREATE INDEX IF NOT EXISTS scraped_data_embedding_hnsw_idx
                ON public.scraped_data
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
                """
            )
            print("HNSW index created successfully.")
        except Exception as e:
            print(f"WARNING: Failed to create HNSW index: {e}")
            print("Run manually: CREATE INDEX scraped_data_embedding_hnsw_idx ON scraped_data USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=64);")
        finally:
            await idx_conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--workers",    type=int, default=None)
    args = parser.parse_args()
    asyncio.run(run(batch_size=args.batch_size, n_workers=args.workers))
