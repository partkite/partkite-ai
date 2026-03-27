import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

SUPABASE_URL: str = os.environ["SUPABASE_URL"]
SUPABASE_KEY: str = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
GEMINI_API_KEY: str = os.environ["GEMINI_API_KEY"]  # comma-separated for key rotation

# Postgres DSN for asyncpg (direct, bypasses supabase-py overhead for vector queries)
POSTGRES_DSN: str = os.environ["POSTGRES_DSN"]  # postgresql://user:pass@host:5432/postgres

# Models
GEMINI_CHAT_MODEL: str = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash")
GEMINI_EMBED_MODEL: str = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")
EMBED_DIM: int = 768  # gemini-embedding-001 supports 768/1536/3072 via MRL; 768 is fast + lean

# Search defaults
DEFAULT_MATCH_COUNT: int = 10
TRGM_SIMILARITY_THRESHOLD: float = 0.15  # pg_trgm threshold (lower = more results)

# API authentication — comma-separated list of valid keys
# Set API_KEYS=key1,key2,key3 in env. If unset, auth is disabled (dev mode).
_raw_keys = os.getenv("API_KEYS", "")
API_KEYS: set[str] = {k.strip() for k in _raw_keys.split(",") if k.strip()}

# Logging level — set LOG_LEVEL=DEBUG in env for verbose output
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
